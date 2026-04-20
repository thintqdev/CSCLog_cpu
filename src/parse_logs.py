"""
Step 1: Parse JSONL log dataset using Drain algorithm.

Input:
  - dataset/data_full.jsonl   (fields: @timestamp, level, host, module, message)

Outputs:
  - dataset/result/data_full_structured.csv
  - dataset/result/data_full_templates.csv
"""

import sys
import os
import json

import pandas as pd

# Add project root to path so utils can be imported
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from utils.Drain import LogParser

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


class JSONLLogParser(LogParser):
    """Drain parser that loads log entries from a JSONL file.

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

    def load_data(self):
        records = []
        with open(self._jsonl_path, 'r', encoding='utf-8') as fh:
            for i, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                records.append({
                    'LineId':    i,
                    'Component': obj.get('module', ''),
                    'Host':      obj.get('host', ''),
                    'Level':     obj.get('level', ''),
                    'iso_time':  obj.get('@timestamp', ''),
                    'Content':   obj.get('message', ''),
                })
        self.df_log = pd.DataFrame(records)


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
    print(f'[parse_logs] Done. Results saved to: {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
