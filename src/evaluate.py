"""
Step 4: Evaluate a saved CSCLog checkpoint on the Linux test sets.

Usage:
    python src/evaluate.py [--checkpoint PATH] [--num_candidates K [K ...]]

Loads the best checkpoint produced by train.py and prints:
  - Top-K anomaly detection: Accuracy, Precision, Recall, F1
  - Per-K breakdown
"""

import sys
import os
import json
import re
import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    classification_report,
)
import dateutil.parser

_NAN_RE = re.compile(r'\bnan\b')


def _safe_eval(s):
    if not isinstance(s, str) or not s.strip() or s.strip().lower() == 'nan':
        return None
    return eval(_NAN_RE.sub('None', s))

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.model import CSCLog

# ── Device: GPU if available, else CPU ──────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'[evaluate] Using device: {DEVICE}')

# ── Paths ─────────────────────────────────────────────────────────────────────
RESULT_DIR    = os.path.join(os.path.dirname(__file__), 'dataset', 'result')
TEMPLATES_CSV = os.path.join(RESULT_DIR, 'data_full_templates.csv')
EMB_PATH      = os.path.join(RESULT_DIR, 'data_full_sentences_emb.json')
COM_PATH      = os.path.join(RESULT_DIR, 'data_full_component.json')
TEST_NOR_CSV  = os.path.join(RESULT_DIR, 'test_normal.csv')
TEST_ANO_CSV  = os.path.join(RESULT_DIR, 'test_anomaly.csv')
DEFAULT_CKPT  = os.path.join(RESULT_DIR, 'csclog_best.pth')


# ─────────────────────────────────────────────────────────────────────────────
# Data helper (same logic as train.py)
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_DT = dateutil.parser.parse('2000-01-01T00:00:00')


def _parse_ts(s):
    if not isinstance(s, str) or not s.strip():
        return _DEFAULT_DT
    return dateutil.parser.parse(s, yearfirst=True)


def load_test_sessions(log_path, templates_csv, emb_path, com_path, window_size):
    df      = pd.read_csv(log_path, engine='c', na_filter=False, memory_map=True)
    temp_df = pd.read_csv(templates_csv, index_col='EventId',
                          engine='c', na_filter=False, memory_map=True)
    mapping  = {idx: i for i, idx in enumerate(temp_df.index.unique())}
    emb      = json.load(open(emb_path))
    cop      = json.load(open(com_path))
    num_keys = len(mapping)
    emb_dim  = len(list(emb.values())[0])

    sessions = []
    for _, row in df.iterrows():
        seqs = _safe_eval(row['EventSequence'])
        if seqs is None:
            continue
        n = len(seqs)
        if n <= window_size:
            continue

        inp, comp, quanp, timep, labels = [], [], [], [], []
        for i in range(n - window_size):
            window = seqs[i:i + window_size]
            qp = [0] * num_keys
            for ev, _, _ in window:
                if ev in mapping:
                    qp[mapping[ev]] += 1
            quanp.append(qp)

            seq, com_l, tm_l = [], [], []
            t0 = _parse_ts(window[0][2])
            for ev, component, ts in window:
                seq.append(emb.get(ev, [0.0] * emb_dim))
                com_l.append(cop.get(component, 0))
                tm_l.append((_parse_ts(ts) - t0).seconds)
            inp.append(seq)
            comp.append(com_l)
            timep.append(tm_l)
            next_ev = seqs[i + window_size][0]
            labels.append(mapping.get(next_ev, -1))

        if inp:
            sessions.append((inp, comp, quanp, timep, labels))

    return sessions


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_sessions(normal_sessions, anomaly_sessions, model,
                      num_candidates_list, anomaly_rate=1, batch_size=256):
    """Batch all windows across sessions for fast inference."""
    model.eval()

    def run(sessions, k_list):
        if not sessions:
            return {k: [] for k in k_list}

        # Flatten all windows with their session index
        all_seq, all_com, all_quan, all_timp, all_labels, session_ids = \
            [], [], [], [], [], []
        for sid, (seq, com, quan, timp, labels) in enumerate(sessions):
            all_seq.extend(seq)
            all_com.extend(com)
            all_quan.extend(quan)
            all_timp.extend(timp)
            all_labels.extend(labels)
            session_ids.extend([sid] * len(labels))

        n = len(all_seq)
        # Per-session miss counters
        session_misses = {k: [0] * len(sessions) for k in k_list}

        from torch.utils.data import TensorDataset, DataLoader
        ds = TensorDataset(
            torch.tensor(all_seq,    dtype=torch.float),
            torch.tensor(all_com,    dtype=torch.long),
            torch.tensor(all_quan,   dtype=torch.float),
            torch.tensor(all_timp,   dtype=torch.float),
            torch.tensor(all_labels, dtype=torch.long),
            torch.tensor(session_ids, dtype=torch.long),
        )
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

        with torch.no_grad():
            for seq_b, com_b, quan_b, timp_b, lab_b, sid_b in loader:
                seq_b  = seq_b.to(DEVICE)
                com_b  = com_b.to(DEVICE)
                quan_b = quan_b.to(DEVICE)
                timp_b = timp_b.to(DEVICE)
                lab_b  = lab_b.to(DEVICE)

                out = model(seq_b, com_b, quan_b, timp_b)
                for k in k_list:
                    topk = torch.argsort(out, dim=1, descending=True)[:, :k].contiguous()
                    wrong = ~(lab_b.unsqueeze(1) == topk).any(dim=1)  # shape (B,)
                    for is_wrong, sid in zip(wrong.tolist(), sid_b.tolist()):
                        if is_wrong:
                            session_misses[k][sid] += 1

        hits = {k: [1 if session_misses[k][s] >= anomaly_rate else 0
                    for s in range(len(sessions))]
                for k in k_list}
        return hits

    nor_hits = run(normal_sessions, num_candidates_list)
    ano_hits = run(anomaly_sessions, num_candidates_list)

    print('=' * 60)
    print(f'  Normal sessions : {len(normal_sessions)}')
    print(f'  Anomaly sessions: {len(anomaly_sessions)}')
    print('=' * 60)

    for k in num_candidates_list:
        preds  = nor_hits[k] + ano_hits[k]
        labels = [0] * len(nor_hits[k]) + [1] * len(ano_hits[k])
        acc  = accuracy_score(labels, preds)
        prec, rec, f1, _ = precision_recall_fscore_support(
            labels, preds, average='macro', zero_division=0)
        print(f'TopK={k:2d} | Accuracy={acc:.4f}  Precision={prec:.4f}  '
              f'Recall={rec:.4f}  F1={f1:.4f}')

    print()
    print('-- Detailed report (TopK=1) --')
    k1_preds  = nor_hits[num_candidates_list[0]] + ano_hits[num_candidates_list[0]]
    k1_labels = [0] * len(nor_hits[num_candidates_list[0]]) + \
                [1] * len(ano_hits[num_candidates_list[0]])
    print(classification_report(k1_labels, k1_preds,
                                 target_names=['Normal', 'Anomaly'],
                                 zero_division=0))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Evaluate CSCLog checkpoint (CPU)')
    p.add_argument('--checkpoint',     type=str, default=DEFAULT_CKPT)
    p.add_argument('--num_candidates', type=int, nargs='+', default=[1, 5, 10])
    p.add_argument('--anomaly_rate',   type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()

    # Load checkpoint
    print(f'[evaluate] Loading checkpoint: {args.checkpoint}')
    ckpt = torch.load(args.checkpoint, map_location=DEVICE)
    saved_args = ckpt['args']
    window_size = saved_args['window_size']

    # Rebuild model from saved config
    model = CSCLog(
        input_size  = ckpt['emb_dim'],
        com_num     = ckpt['num_coms'],
        hidden_size = saved_args['hidden_size'],
        alpha       = saved_args['alpha'],
        pattern     = saved_args['pattern'],
        num_layers  = saved_args['num_layers'],
        num_keys    = ckpt['num_keys'],
        drop        = saved_args['drop'],
    ).to(DEVICE)
    model.load_state_dict(ckpt['model'])
    print(f'[evaluate] Model restored (epoch {ckpt["epoch"]}, '
          f'best F1={ckpt["f1"]:.3f})')

    # Load test data
    print('[evaluate] Loading test sessions …')
    normal_sessions  = load_test_sessions(TEST_NOR_CSV, TEMPLATES_CSV, EMB_PATH,
                                          COM_PATH, window_size)
    anomaly_sessions = load_test_sessions(TEST_ANO_CSV, TEMPLATES_CSV, EMB_PATH,
                                          COM_PATH, window_size)

    # Evaluate
    evaluate_sessions(normal_sessions, anomaly_sessions, model,
                      args.num_candidates, args.anomaly_rate)


if __name__ == '__main__':
    main()
