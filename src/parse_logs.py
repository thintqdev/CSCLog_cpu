"""
Step 1: Parse raw Linux.log using Drain algorithm.
Outputs:
  - dataset/result/Linux.log_structured.csv
  - dataset/result/Linux.log_templates.csv
"""

import sys
import os

# Add project root to path so utils can be imported
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from utils.Drain import LogParser

# ── Linux log format ────────────────────────────────────────────────────────
# Example line:
#   Jun  9 06:06:20 combo syslogd 1.4.1: restart.
LOG_FORMAT = '<Month> <Day> <Time> <Host> <Component>: <Content>'

# Drain hyper-params tuned for Linux syslog
DEPTH       = 4
ST          = 0.4       # similarity threshold
MAX_CHILD   = 100

# Regex pre-processing: replace common variable patterns before parsing
REX = [
    r'(\d+\.){3}\d+(:\d+)?',           # IP addresses
    r'blk_(|-)[0-9]+',                  # block IDs
    r'(/[-\w]+)+',                      # file paths
    r'(?<=[^A-Za-z0-9])(\-?\+?\d+)(?=[^A-Za-z0-9])|[0-9]+$',  # numbers
]

INPUT_DIR  = os.path.join(os.path.dirname(__file__), 'dataset')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'dataset', 'result')
LOG_FILE   = 'Linux.log'


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    parser = LogParser(
        log_format=LOG_FORMAT,
        indir=INPUT_DIR,
        outdir=OUTPUT_DIR,
        depth=DEPTH,
        st=ST,
        maxChild=MAX_CHILD,
        rex=REX,
        keep_para=True,
    )
    parser.parse(LOG_FILE)
    print(f'[parse_logs] Done. Results saved to: {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
