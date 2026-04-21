"""
Step 1: Parse JSONL log dataset using Drain algorithm.

Input:
  - dataset/data_full.jsonl   (fields: @timestamp, level, host, module, message)

Outputs:
  - dataset/result/data_full_structured.csv
  - dataset/result/data_full_templates.csv

Memory-efficient two-pass streaming approach:
  Pass 1 – stream JSONL line-by-line to build the Drain prefix tree.
            No log IDs or DataFrames are kept in RAM.
  Pass 2 – stream JSONL again, look up each line's template via the tree,
            and write the structured CSV in fixed-size chunks.
"""

import sys
import os
import csv
import json
import hashlib
from datetime import datetime

# Add project root to path so utils can be imported
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from utils.Drain import LogParser, Node, Logcluster

# Drain hyper-params
DEPTH     = 4
ST        = 0.4       # similarity threshold
MAX_CHILD = 100

# Regex pre-processing applied to the message field before template extraction
REX = [
    r'(\d+\.){3}\d+(:\d+)?',                                    # IP addresses
    r'req-[0-9a-f\-]{36}',                                      # request UUIDs
    r'[0-9a-f]{32}',                                            # hex IDs
    r'blk_(|-)[0-9]+',                                          # block IDs
    r'(?<=[^A-Za-z0-9])(\-?\+?\d+)(?=[^A-Za-z0-9])|[0-9]+$',  # numbers
]

INPUT_DIR  = os.path.join(os.path.dirname(__file__), 'dataset')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'dataset', 'result')
JSONL_FILE = 'data_full.jsonl'
LOG_NAME   = 'data_full'   # prefix for output CSV file names

# Number of rows to buffer before flushing to the structured CSV
WRITE_CHUNK = 50_000
# Progress report interval (lines)
REPORT_EVERY = 100_000


class JSONLLogParser(LogParser):
    """Memory-efficient Drain parser for large JSONL log files.

    Overrides ``parse()`` with a two-pass streaming implementation so that
    neither the full log DataFrame nor per-template log-ID lists are ever
    kept in RAM.

    Each JSON object must have:
      @timestamp  – ISO-8601 timestamp
      level       – log level (e.g. INFO, ERROR)
      host        – source host
      module      – component / module name  (mapped to 'Component')
      message     – raw log message text     (mapped to 'Content')
    """

    def __init__(self, jsonl_path: str, **kwargs):
        super().__init__(log_format='<Content>', **kwargs)
        self._jsonl_path = jsonl_path

    # ------------------------------------------------------------------
    # Streaming parse – replaces the parent's RAM-heavy implementation
    # ------------------------------------------------------------------
    def parse(self, logName):
        print(f'Parsing file: {self._jsonl_path}')
        start_time = datetime.now()
        self.logName = logName

        os.makedirs(self.savePath, exist_ok=True)

        root_node  = Node()
        log_clusters: list[Logcluster] = []
        # Map cluster id → occurrence count (avoids storing every line ID)
        occ_count: dict[int, int] = {}

        # ── Pass 1: build Drain prefix tree ───────────────────────────
        print('[parse_logs] Pass 1: building template tree...')
        total = 0
        with open(self._jsonl_path, 'r', encoding='utf-8') as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                total += 1
                obj    = json.loads(raw)
                tokens = self.preprocess(obj.get('message', '')).strip().split()

                match = self.treeSearch(root_node, tokens)
                if match is None:
                    cluster = Logcluster(logTemplate=tokens, logIDL=[])
                    log_clusters.append(cluster)
                    occ_count[id(cluster)] = 1
                    self.addSeqToPrefixTree(root_node, cluster)
                else:
                    new_tpl = self.getTemplate(tokens, match.logTemplate)
                    if ' '.join(new_tpl) != ' '.join(match.logTemplate):
                        match.logTemplate = new_tpl
                    occ_count[id(match)] = occ_count.get(id(match), 0) + 1

                if total % REPORT_EVERY == 0:
                    print(f'  Pass 1: {total:,} lines processed...')

        print(f'[parse_logs] Pass 1 done – {total:,} lines, '
              f'{len(log_clusters)} templates found.')

        # Build template-string → (event_id, occurrence) lookup
        template_lookup: dict[str, tuple[str, int]] = {}
        for cluster in log_clusters:
            tstr = ' '.join(cluster.logTemplate)
            tid  = hashlib.md5(tstr.encode('utf-8')).hexdigest()[:8]
            template_lookup[tstr] = (tid, occ_count.get(id(cluster), 0))

        # Free the cluster list – tree nodes still needed for Pass 2
        log_clusters.clear()
        occ_count.clear()

        # ── Pass 2: assign templates, write structured CSV in chunks ──
        print('[parse_logs] Pass 2: writing structured CSV...')
        structured_path = os.path.join(self.savePath, logName + '_structured.csv')
        fieldnames = ['LineId', 'Component', 'Host', 'Level',
                      'iso_time', 'Content', 'EventId', 'EventTemplate']

        written = 0
        with open(structured_path, 'w', newline='', encoding='utf-8') as out_fh:
            writer = csv.DictWriter(out_fh, fieldnames=fieldnames)
            writer.writeheader()
            chunk: list[dict] = []

            with open(self._jsonl_path, 'r', encoding='utf-8') as fh:
                line_id = 0
                for raw in fh:
                    raw = raw.strip()
                    if not raw:
                        continue
                    line_id += 1
                    obj     = json.loads(raw)
                    content = obj.get('message', '')
                    tokens  = self.preprocess(content).strip().split()

                    match = self.treeSearch(root_node, tokens)
                    if match is not None:
                        tstr = ' '.join(match.logTemplate)
                    else:
                        tstr = content          # fallback: use raw content
                    tid = hashlib.md5(tstr.encode('utf-8')).hexdigest()[:8]

                    chunk.append({
                        'LineId':        line_id,
                        'Component':     obj.get('module', ''),
                        'Host':          obj.get('host', ''),
                        'Level':         obj.get('level', ''),
                        'iso_time':      obj.get('@timestamp', ''),
                        'Content':       content,
                        'EventId':       tid,
                        'EventTemplate': tstr,
                    })

                    if len(chunk) >= WRITE_CHUNK:
                        writer.writerows(chunk)
                        written += len(chunk)
                        chunk.clear()
                        if written % REPORT_EVERY == 0:
                            print(f'  Pass 2: {written:,} lines written...')

            if chunk:
                writer.writerows(chunk)
                written += len(chunk)

        print(f'[parse_logs] Pass 2 done – {written:,} lines written.')

        # ── Write templates CSV ────────────────────────────────────────
        templates_path = os.path.join(self.savePath, logName + '_templates.csv')
        with open(templates_path, 'w', newline='', encoding='utf-8') as tf:
            writer = csv.DictWriter(
                tf, fieldnames=['EventId', 'EventTemplate', 'Occurrences'])
            writer.writeheader()
            for tstr, (tid, occ) in template_lookup.items():
                writer.writerow(
                    {'EventId': tid, 'EventTemplate': tstr, 'Occurrences': occ})

        elapsed = datetime.now() - start_time
        print(f'[parse_logs] Parsing done. [Time taken: {elapsed}]')
        print(f'[parse_logs] Results saved to: {self.savePath}')


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    parser = JSONLLogParser(
        jsonl_path=os.path.join(INPUT_DIR, JSONL_FILE),
        indir=INPUT_DIR,
        outdir=OUTPUT_DIR,
        depth=DEPTH,
        st=ST,
        maxChild=MAX_CHILD,
        rex=REX,
        keep_para=True,
    )
    parser.parse(LOG_NAME)


if __name__ == '__main__':
    main()
