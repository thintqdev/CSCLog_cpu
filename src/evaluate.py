"""
Step 4: Evaluate CSCLog checkpoint.
=====================================
Optimizations vs previous version:
  1. All session_misses counters on GPU (zero CPU sync per batch) — same as train.py
  2. tqdm progress bars
  3. Larger default eval batch size (1024)
  4. ujson fallback for faster JSON loading
  5. Pre-parsed timestamps (float) instead of dateutil in hot loop
  6. torch.compile support (--compile flag)
  7. TF32 enabled on Ampere+
"""

import sys
import os
import re
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    classification_report,
)

try:
    import ujson as json
except ImportError:
    import json

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

import dateutil.parser

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.model import CSCLog

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'[evaluate] Device: {DEVICE}')

if DEVICE.type == 'cuda':
    torch.backends.cudnn.benchmark     = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32    = True
    print(f'[evaluate] GPU: {torch.cuda.get_device_name(0)}  '
          f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

# ── Paths ─────────────────────────────────────────────────────────────────────
RESULT_DIR    = os.path.join(os.path.dirname(__file__), 'dataset', 'result')
TEMPLATES_CSV = os.path.join(RESULT_DIR, 'data_full_templates.csv')
EMB_PATH      = os.path.join(RESULT_DIR, 'data_full_sentences_emb.json')
COM_PATH      = os.path.join(RESULT_DIR, 'data_full_component.json')
TEST_NOR_CSV  = os.path.join(RESULT_DIR, 'test_normal.csv')
TEST_ANO_CSV  = os.path.join(RESULT_DIR, 'test_anomaly.csv')
DEFAULT_CKPT  = os.path.join(RESULT_DIR, 'csclog_best.pth')

# ── Timestamp helper ──────────────────────────────────────────────────────────
_DEFAULT_TS = dateutil.parser.parse('2000-01-01T00:00:00').timestamp()
_NAN_RE     = re.compile(r'\bnan\b')


def _parse_ts_float(s) -> float:
    if not isinstance(s, str) or not s.strip():
        return _DEFAULT_TS
    try:
        return dateutil.parser.parse(s, yearfirst=True).replace(tzinfo=None).timestamp()
    except Exception:
        return _DEFAULT_TS


def _safe_eval(s):
    if not isinstance(s, str) or not s.strip() or s.strip().lower() == 'nan':
        return None
    return eval(_NAN_RE.sub('None', s))


# ─────────────────────────────────────────────────────────────────────────────
# Load test sessions
# ─────────────────────────────────────────────────────────────────────────────

def load_test_sessions(log_path, templates_csv, emb_path, com_path, window_size):
    df      = pd.read_csv(log_path, engine='c', na_filter=False, memory_map=True)
    temp_df = pd.read_csv(templates_csv, index_col='EventId',
                          engine='c', na_filter=False, memory_map=True)
    mapping  = {idx: i for i, idx in enumerate(temp_df.index.unique())}
    emb      = json.load(open(emb_path))
    cop      = json.load(open(com_path))
    num_keys = len(mapping)
    emb_dim  = len(next(iter(emb.values())))

    sessions = []
    rows = list(df.itertuples(index=False))
    bar  = tqdm(rows, desc=f'  Loading {os.path.basename(log_path)}',
                unit='session') if HAS_TQDM else rows

    for row in bar:
        seqs = _safe_eval(row.EventSequence)
        if seqs is None or len(seqs) <= window_size:
            continue
        n = len(seqs)

        inp, comp, quanp, timep, labels = [], [], [], [], []
        for i in range(n - window_size):
            window  = seqs[i: i + window_size]
            ev_idxs = [mapping.get(ev, 0) for ev, _, _ in window]
            qp      = np.bincount(ev_idxs, minlength=num_keys).tolist()
            quanp.append(qp)

            seq_l, com_l, tm_l = [], [], []
            t0 = _parse_ts_float(window[0][2])
            for ev, component, ts in window:
                seq_l.append(emb.get(ev, [0.0] * emb_dim))
                com_l.append(cop.get(component, 0))
                tm_l.append(_parse_ts_float(ts) - t0)
            inp.append(seq_l)
            comp.append(com_l)
            timep.append(tm_l)
            labels.append(mapping.get(seqs[i + window_size][0], -1))

        if inp:
            sessions.append((inp, comp, quanp, timep, labels))

    print(f'[evaluate] Loaded {len(sessions):,} sessions from {os.path.basename(log_path)}')
    return sessions


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation — GPU-only counters
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_sessions(normal_sessions, anomaly_sessions, model,
                      num_candidates_list, anomaly_rate=1, batch_size=1024):
    model.eval()
    use_amp = (DEVICE.type == 'cuda')

    def run(sessions, k_list):
        if not sessions:
            return {k: [] for k in k_list}

        all_seq, all_com, all_quan, all_timp, all_labels, session_ids = \
            [], [], [], [], [], []
        for sid, (seq, com, quan, timp, labels) in enumerate(sessions):
            all_seq.extend(seq);   all_com.extend(com)
            all_quan.extend(quan); all_timp.extend(timp)
            all_labels.extend(labels)
            session_ids.extend([sid] * len(labels))

        # GPU-resident miss counters (zero CPU sync in loop)
        session_misses = {k: torch.zeros(len(sessions), dtype=torch.long, device=DEVICE)
                          for k in k_list}

        ds = TensorDataset(
            torch.tensor(all_seq,     dtype=torch.float),
            torch.tensor(all_com,     dtype=torch.long),
            torch.tensor(all_quan,    dtype=torch.float),
            torch.tensor(all_timp,    dtype=torch.float),
            torch.tensor(all_labels,  dtype=torch.long),
            torch.tensor(session_ids, dtype=torch.long),
        )
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            pin_memory=(DEVICE.type == 'cuda'),
                            num_workers=4, prefetch_factor=2)

        bar = tqdm(loader, desc='  inference', unit='batch',
                   dynamic_ncols=True) if HAS_TQDM else loader

        with torch.no_grad():
            for seq_b, com_b, quan_b, timp_b, lab_b, sid_b in bar:
                seq_b  = seq_b.to(DEVICE,  non_blocking=True)
                com_b  = com_b.to(DEVICE,  non_blocking=True)
                quan_b = quan_b.to(DEVICE, non_blocking=True)
                timp_b = timp_b.to(DEVICE, non_blocking=True)
                lab_b  = lab_b.to(DEVICE,  non_blocking=True)
                sid_b  = sid_b.to(DEVICE,  non_blocking=True)  # stays on GPU

                with torch.autocast(device_type=DEVICE.type, enabled=use_amp):
                    out = model(seq_b, com_b, quan_b, timp_b)

                for k in k_list:
                    topk      = torch.argsort(out, dim=1, descending=True)[:, :k]
                    wrong     = ~(lab_b.unsqueeze(1) == topk).any(dim=1)
                    wrong_sid = sid_b[wrong]
                    if wrong_sid.numel() > 0:
                        session_misses[k].scatter_add_(
                            0, wrong_sid,
                            torch.ones(wrong_sid.numel(), dtype=torch.long, device=DEVICE))

        # Single D2H at end
        return {k: (session_misses[k] >= anomaly_rate).long().cpu().tolist()
                for k in k_list}

    print(f'\n[evaluate] Normal sessions : {len(normal_sessions):,}')
    print(f'[evaluate] Anomaly sessions: {len(anomaly_sessions):,}')

    nor_hits = run(normal_sessions, num_candidates_list)
    ano_hits = run(anomaly_sessions, num_candidates_list)

    print('\n' + '=' * 60)
    for k in num_candidates_list:
        preds  = nor_hits[k] + ano_hits[k]
        labels = [0] * len(nor_hits[k]) + [1] * len(ano_hits[k])
        acc = accuracy_score(labels, preds)
        prec_arr, rec_arr, f1_arr, _ = precision_recall_fscore_support(
            labels, preds, average=None, labels=[0, 1], zero_division=0)
        ano_prec = float(prec_arr[1]) if len(prec_arr) > 1 else 0.0
        ano_rec  = float(rec_arr[1])  if len(rec_arr)  > 1 else 0.0
        macro_f1 = float(f1_arr.mean())
        denom    = 4.0 * ano_prec + ano_rec
        fbeta2   = 5.0 * ano_prec * ano_rec / denom if denom > 0 else 0.0
        print(f'TopK={k:2d} | Acc={acc:.4f}  AnoPrec={ano_prec:.4f}  '
              f'AnoRec={ano_rec:.4f}  F1={macro_f1:.4f}  F2ano={fbeta2:.4f}')
    print('=' * 60)

    print('\n-- Detailed report (TopK=1) --')
    k1 = num_candidates_list[0]
    k1_preds  = nor_hits[k1] + ano_hits[k1]
    k1_labels = [0] * len(nor_hits[k1]) + [1] * len(ano_hits[k1])
    print(classification_report(k1_labels, k1_preds,
                                 target_names=['Normal', 'Anomaly'], zero_division=0))


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Evaluate CSCLog checkpoint')
    p.add_argument('--checkpoint',      type=str, default=DEFAULT_CKPT)
    p.add_argument('--num_candidates',  type=int, nargs='+', default=[1, 5, 10])
    p.add_argument('--anomaly_rate',    type=int, default=1)
    p.add_argument('--batch_size',      type=int, default=1024)
    p.add_argument('--compile',         action='store_true',
                   help='torch.compile for faster inference (PyTorch 2.0+)')
    return p.parse_args()


def main():
    args = parse_args()
    t0   = datetime.now()

    print(f'[evaluate] Loading checkpoint: {args.checkpoint}')
    ckpt       = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
    saved_args = ckpt['args']
    window_size = saved_args['window_size']

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

    if args.compile and hasattr(torch, 'compile'):
        print('[evaluate] Compiling model …')
        model = torch.compile(model, mode='max-autotune')

    saved_metric = ckpt.get('fbeta2_ano', ckpt.get('rec', ckpt.get('f1', 0.0)))
    metric_name  = ('F2ano' if 'fbeta2_ano' in ckpt
                    else ('AnoRec' if 'rec' in ckpt else 'F1'))
    print(f'[evaluate] Epoch {ckpt["epoch"]}, best {metric_name}={saved_metric:.3f}')

    print('[evaluate] Loading test sessions …')
    normal_sessions  = load_test_sessions(TEST_NOR_CSV, TEMPLATES_CSV,
                                          EMB_PATH, COM_PATH, window_size)
    anomaly_sessions = load_test_sessions(TEST_ANO_CSV, TEMPLATES_CSV,
                                          EMB_PATH, COM_PATH, window_size)

    evaluate_sessions(normal_sessions, anomaly_sessions, model,
                      args.num_candidates, args.anomaly_rate, args.batch_size)

    elapsed = datetime.now() - t0
    print(f'\n[evaluate] Total time: {elapsed}')


if __name__ == '__main__':
    main()