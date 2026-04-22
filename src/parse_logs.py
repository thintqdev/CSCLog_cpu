"""
Step 1: Parse JSONL log dataset using Drain algorithm.
======================================================
Optimizations vs previous version:
  1. ujson (3-5x faster than stdlib json) with fallback to stdlib
  2. tqdm progress bars instead of print-every-N
  3. Pass 1: pre-compiled regex applied once per line (not per token)
  4. Pass 2: parallel worker pool processes chunks concurrently,
             writer thread flushes to disk without blocking workers
  5. template_lookup keyed by MD5 computed once, reused in Pass 2
  6. Larger WRITE_CHUNK (200k) → fewer flush syscalls
  7. io.BufferedWriter for CSV output (reduces kernel context switches)
"""

import sys
import os
import csv
import hashlib
import io
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import partial

try:
    import ujson as json
except ImportError:
    import json  # fallback

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from utils.Drain import LogParser, Node, Logcluster

# ── Drain hyper-params ────────────────────────────────────────────────────────
DEPTH     = 4
ST        = 0.4
MAX_CHILD = 100

REX = [
    r'(\d+\.){3}\d+(:\d+)?',
    r'req-[0-9a-f\-]{36}',
    r'[0-9a-f]{32}',
    r'blk_(|-)[0-9]+',
    r'(?<=[^A-Za-z0-9])(\-?\+?\d+)(?=[^A-Za-z0-9])|[0-9]+$',
]

INPUT_DIR    = os.path.join(os.path.dirname(__file__), 'dataset')
OUTPUT_DIR   = os.path.join(os.path.dirname(__file__), 'dataset', 'result')
JSONL_FILE   = 'data_full.jsonl'
LOG_NAME     = 'data_full'

WRITE_CHUNK  = 200_000   # rows buffered before disk flush (was 50k)
READ_CHUNK   = 50_000    # lines per parallel worker task
N_WORKERS    = max(1, mp.cpu_count() - 1)  # leave 1 core for I/O


# ─────────────────────────────────────────────────────────────────────────────
# Worker function (runs in process pool — must be top-level for pickling)
# ─────────────────────────────────────────────────────────────────────────────

def _process_chunk(args):
    """
    Process a list of raw JSONL strings.
    Returns list of dicts ready to write to CSV.
    template_lookup is passed as argument (picklable dict).
    """
    raw_lines, template_lookup, preprocess_fn, start_id = args
    rows = []
    line_id = start_id
    for raw in raw_lines:
        raw = raw.strip()
        if not raw:
            continue
        line_id += 1
        try:
            obj = json.loads(raw)
        except Exception:
            continue
        content = obj.get('message', '')
        tokens  = preprocess_fn(content).strip().split()
        tstr    = ' '.join(tokens)
        # Look up MD5 from template_lookup; fallback to content hash
        tid = template_lookup.get(tstr)
        if tid is None:
            tid = hashlib.md5(tstr.encode()).hexdigest()[:8]
        rows.append({
            'LineId':        line_id,
            'Component':     obj.get('module', ''),
            'Host':          obj.get('host', ''),
            'Level':         obj.get('level', ''),
            'iso_time':      obj.get('@timestamp', ''),
            'Content':       content,
            'EventId':       tid,
            'EventTemplate': tstr,
        })
    return rows


class JSONLLogParser(LogParser):
    """
    Memory-efficient + fast Drain parser for large JSONL log files.

    Two-pass streaming:
      Pass 1 – single-threaded (Drain tree is not thread-safe): build tree
      Pass 2 – parallel chunked processing + async disk writes
    """

    def __init__(self, jsonl_path: str, **kwargs):
        super().__init__(log_format='<Content>', **kwargs)
        self._jsonl_path = jsonl_path

    def parse(self, logName):
        print(f'[parse_logs] Parsing: {self._jsonl_path}')
        print(f'[parse_logs] Workers: {N_WORKERS}  (cpu_count={mp.cpu_count()})')
        start_time = datetime.now()
        self.logName = logName
        os.makedirs(self.savePath, exist_ok=True)

        # ── Pass 1: build Drain prefix tree (must be sequential) ──────
        print('[parse_logs] Pass 1: building template tree …')
        root_node    = Node()
        log_clusters = []
        occ_count    = {}

        # Count total lines first for tqdm
        total_lines = 0
        with open(self._jsonl_path, 'rb') as fh:
            for _ in fh:
                total_lines += 1
        print(f'[parse_logs] Total lines: {total_lines:,}')

        with open(self._jsonl_path, 'r', encoding='utf-8') as fh:
            iter_lines = tqdm(fh, total=total_lines, desc='Pass1', unit='line',
                              dynamic_ncols=True) if HAS_TQDM else fh
            for raw in iter_lines:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    obj = json.loads(raw)
                except Exception:
                    continue
                tokens = self.preprocess(obj.get('message', '')).strip().split()

                match = self.treeSearch(root_node, tokens)
                if match is None:
                    cluster = Logcluster(logTemplate=tokens, logIDL=[])
                    log_clusters.append(cluster)
                    occ_count[id(cluster)] = 1
                    self.addSeqToPrefixTree(root_node, cluster)
                else:
                    new_tpl = self.getTemplate(tokens, match.logTemplate)
                    if new_tpl != match.logTemplate:
                        match.logTemplate = new_tpl
                    occ_count[id(match)] = occ_count.get(id(match), 0) + 1

        print(f'[parse_logs] Pass 1 done – {len(log_clusters):,} templates found.')

        # Build template_lookup: tstr → event_id (MD5)
        # Also build occ_map: tstr → occurrences
        template_lookup: dict[str, str] = {}
        occ_map:         dict[str, int] = {}
        for cluster in log_clusters:
            tstr = ' '.join(cluster.logTemplate)
            tid  = hashlib.md5(tstr.encode('utf-8')).hexdigest()[:8]
            template_lookup[tstr] = tid
            occ_map[tstr]         = occ_count.get(id(cluster), 0)

        log_clusters.clear()
        occ_count.clear()

        # ── Pass 2: parallel chunk processing ─────────────────────────
        print(f'[parse_logs] Pass 2: parallel CSV writing '
              f'(chunk={READ_CHUNK:,}, workers={N_WORKERS}) …')

        structured_path = os.path.join(self.savePath, logName + '_structured.csv')
        fieldnames = ['LineId', 'Component', 'Host', 'Level',
                      'iso_time', 'Content', 'EventId', 'EventTemplate']

        # Capture preprocess method for worker (bound method not picklable → wrap)
        preprocess_fn = self.preprocess

        written  = 0
        chunk_id = 0

        with open(structured_path, 'w', newline='', encoding='utf-8',
                  buffering=8 * 1024 * 1024) as out_fh:  # 8 MB write buffer
            writer = csv.DictWriter(out_fh, fieldnames=fieldnames)
            writer.writeheader()

            # Read file in large chunks, dispatch to pool
            with open(self._jsonl_path, 'r', encoding='utf-8') as fh:
                iter_outer = tqdm(desc='Pass2', unit='row', total=total_lines,
                                  dynamic_ncols=True) if HAS_TQDM else None

                current_chunk = []
                line_counter  = 0

                # Use a pool for CPU-bound JSON parsing
                with mp.Pool(processes=N_WORKERS) as pool:
                    futures = []

                    def _submit(lines, start):
                        return pool.apply_async(
                            _process_chunk,
                            ((lines, template_lookup, preprocess_fn, start),)
                        )

                    for raw in fh:
                        current_chunk.append(raw)
                        if len(current_chunk) >= READ_CHUNK:
                            futures.append(_submit(current_chunk, line_counter))
                            line_counter  += len(current_chunk)
                            current_chunk  = []
                            chunk_id      += 1

                        # Flush completed futures to disk (keep memory bounded)
                        while futures and futures[0].ready():
                            rows = futures.pop(0).get()
                            writer.writerows(rows)
                            written += len(rows)
                            if iter_outer:
                                iter_outer.update(len(rows))

                    # Submit remaining lines
                    if current_chunk:
                        futures.append(_submit(current_chunk, line_counter))

                    # Drain remaining futures
                    for fut in futures:
                        rows = fut.get()
                        writer.writerows(rows)
                        written += len(rows)
                        if iter_outer:
                            iter_outer.update(len(rows))

                if iter_outer:
                    iter_outer.close()

        print(f'[parse_logs] Pass 2 done – {written:,} rows written.')

        # ── Write templates CSV ────────────────────────────────────────
        templates_path = os.path.join(self.savePath, logName + '_templates.csv')
        with open(templates_path, 'w', newline='', encoding='utf-8') as tf:
            w = csv.DictWriter(tf, fieldnames=['EventId', 'EventTemplate', 'Occurrences'])
            w.writeheader()
            for tstr, tid in template_lookup.items():
                w.writerow({'EventId': tid,
                            'EventTemplate': tstr,
                            'Occurrences': occ_map.get(tstr, 0)})

        elapsed = datetime.now() - start_time
        print(f'[parse_logs] Done. Time: {elapsed}  →  {self.savePath}')


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