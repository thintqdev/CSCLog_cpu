"""
Step 3: Train CSCLog on the Linux dataset (CPU-only).

Usage:
    python src/train.py [--epochs N] [--batch_size B] [--window_size W]

Inputs  (produced by preprocess.py):
    dataset/result/train_normal.csv
    dataset/result/test_normal.csv
    dataset/result/test_anomaly.csv
    dataset/result/Linux.log_templates.csv
    dataset/result/Linux_sentences_emb.json
    dataset/result/Linux_component.json

Outputs:
    dataset/result/csclog_best.pth   – best checkpoint (highest anomaly Recall)
"""

import sys
import os
import json
import re
import argparse
import random
import collections
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import dateutil.parser

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.model import CSCLog

# ── Reproducibility ───────────────────────────────────────────────────────────

def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

seed_everything(42)

# ── Device: GPU if available, else CPU ──────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'[train] Using device: {DEVICE}')

# ── Paths ─────────────────────────────────────────────────────────────────────
RESULT_DIR   = os.path.join(os.path.dirname(__file__), 'dataset', 'result')
TEMPLATES_CSV = os.path.join(RESULT_DIR, 'data_full_templates.csv')
EMB_PATH      = os.path.join(RESULT_DIR, 'data_full_sentences_emb.json')
COM_PATH      = os.path.join(RESULT_DIR, 'data_full_component.json')
TRAIN_CSV     = os.path.join(RESULT_DIR, 'train_normal.csv')
TEST_NOR_CSV  = os.path.join(RESULT_DIR, 'test_normal.csv')
TEST_ANO_CSV  = os.path.join(RESULT_DIR, 'test_anomaly.csv')
CKPT_PATH     = os.path.join(RESULT_DIR, 'csclog_best.pth')

# ── Default hyper-params ──────────────────────────────────────────────────────
DEFAULTS = dict(
    window_size  = 9,
    batch_size   = 8,
    epochs       = 10,
    lr           = 1e-4,
    weight_decay = 1e-4,
    drop         = 0.1,
    hidden_size  = [64, 64, 64, 64, 64],   # ft, lstm, mlp, gcn, out
    alpha        = 0.8,
    pattern      = 1,
    num_layers   = 2,
    num_candidates = [1],
    anomaly_rate = 1,
)


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_DT = dateutil.parser.parse('2000-01-01T00:00:00')


def _parse_ts(s):
    if not isinstance(s, str) or not s.strip():
        return _DEFAULT_DT
    return dateutil.parser.parse(s, yearfirst=True).replace(tzinfo=None)


_NAN_RE = re.compile(r'\bnan\b')


def _safe_eval(s):
    """eval an EventSequence string, replacing bare nan tokens with None."""
    if not isinstance(s, str) or not s.strip() or s.strip().lower() == 'nan':
        return None
    return eval(_NAN_RE.sub('None', s))


class _TrainDataset(torch.utils.data.Dataset):
    """Memory-efficient sliding-window dataset.

    Keeps raw session lists in memory and encodes each window to tensors
    on-the-fly inside __getitem__, so only one batch worth of data is
    ever materialised at a time.
    """

    def __init__(self, sessions, mapping, emb, cop, emb_dim, num_keys, window_size):
        self.emb      = emb
        self.cop      = cop
        self.mapping  = mapping
        self.emb_dim  = emb_dim
        self.num_keys = num_keys
        self.ws       = window_size
        # Build a flat index of (session_events_list, start_offset) pairs.
        # Skip windows whose target EventId is None (unknown after NaN substitution).
        self.index: list = []
        for seqs in sessions:
            n = len(seqs)
            for i in range(n - window_size):
                if seqs[i + window_size][0] is not None:
                    self.index.append((seqs, i))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        seqs, i = self.index[idx]
        window  = seqs[i: i + self.ws]
        label   = self.mapping.get(seqs[i + self.ws][0], 0)

        qp = [0] * self.num_keys
        for ev, _, _ in window:
            if ev in self.mapping:
                qp[self.mapping[ev]] += 1

        inp, com, tm = [], [], []
        t0 = _parse_ts(window[0][2])
        for ev, component, ts in window:
            inp.append(self.emb.get(ev, [0.0] * self.emb_dim))
            com.append(self.cop.get(component, 0))
            tm.append((_parse_ts(ts) - t0).seconds)

        return (
            torch.tensor(inp,   dtype=torch.float),
            torch.tensor(com,   dtype=torch.long),
            torch.tensor(qp,    dtype=torch.float),
            torch.tensor(tm,    dtype=torch.float),
            torch.tensor(label, dtype=torch.long),
        )


def generate_train(train_path, templates_csv, emb_path, com_path, window_size):
    """Return a lazy Dataset for sliding-window next-event prediction."""
    train_df = pd.read_csv(train_path, engine='c', na_filter=False, memory_map=True)
    temp_df  = pd.read_csv(templates_csv, index_col='EventId',
                           engine='c', na_filter=False, memory_map=True)
    mapping  = {idx: i for i, idx in enumerate(temp_df.index.unique())}
    emb      = json.load(open(emb_path))
    cop      = json.load(open(com_path))
    num_keys = len(mapping)
    emb_dim  = len(list(emb.values())[0])

    sessions = [
        seq for seq in (
            _safe_eval(row['EventSequence'])
            for _, row in train_df.iterrows()
        )
        if seq is not None
    ]

    dataset = _TrainDataset(sessions, mapping, emb, cop, emb_dim, num_keys, window_size)
    print(f'[train] Training sequences: {len(dataset)}, '
          f'emb_dim={emb_dim}, num_keys={num_keys}, num_coms={len(cop)}')
    return dataset, emb_dim, num_keys, len(cop)


def generate_test(log_path, templates_csv, emb_path, com_path, window_size):
    """Load test CSV as per-session lists (same format as main.ipynb)."""
    df      = pd.read_csv(log_path, engine='c', na_filter=False, memory_map=True)
    temp_df = pd.read_csv(templates_csv, index_col='EventId',
                          engine='c', na_filter=False, memory_map=True)
    mapping = {idx: i for i, idx in enumerate(temp_df.index.unique())}
    emb     = json.load(open(emb_path))
    cop     = json.load(open(com_path))
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

    print(f'[train] Test sessions loaded: {len(sessions)} from {log_path}')
    return sessions


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_topk(normal_sessions, anomaly_sessions, model, num_candidates_list,
                  anomaly_rate=1, batch_size=256):
    """Batch all windows across sessions for fast inference."""
    model.eval()

    def session_hit(sessions, k_list):
        if not sessions:
            return {k: [] for k in k_list}

        all_seq, all_com, all_quan, all_timp, all_labels, session_ids = \
            [], [], [], [], [], []
        for sid, (seq, com, quan, timp, labels) in enumerate(sessions):
            all_seq.extend(seq)
            all_com.extend(com)
            all_quan.extend(quan)
            all_timp.extend(timp)
            all_labels.extend(labels)
            session_ids.extend([sid] * len(labels))

        session_misses = {k: [0] * len(sessions) for k in k_list}

        ds = TensorDataset(
            torch.tensor(all_seq,     dtype=torch.float),
            torch.tensor(all_com,     dtype=torch.long),
            torch.tensor(all_quan,    dtype=torch.float),
            torch.tensor(all_timp,    dtype=torch.float),
            torch.tensor(all_labels,  dtype=torch.long),
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
                    topk  = torch.argsort(out, dim=1, descending=True)[:, :k].contiguous()
                    wrong = ~(lab_b.unsqueeze(1) == topk).any(dim=1)  # shape (B,)
                    for is_wrong, sid in zip(wrong.tolist(), sid_b.tolist()):
                        if is_wrong:
                            session_misses[k][sid] += 1

        return {k: [1 if session_misses[k][s] >= anomaly_rate else 0
                    for s in range(len(sessions))]
                for k in k_list}

    nor_hits = session_hit(normal_sessions, num_candidates_list)
    ano_hits = session_hit(anomaly_sessions, num_candidates_list)

    results = {}
    for k in num_candidates_list:
        preds  = nor_hits[k] + ano_hits[k]
        labels = [0] * len(nor_hits[k]) + [1] * len(ano_hits[k])
        acc = accuracy_score(labels, preds)
        prec_arr, rec_arr, f1_arr, _ = precision_recall_fscore_support(
            labels, preds, average=None, labels=[0, 1], zero_division=0)
        # Minimise False Negatives: track recall on the anomaly class (class 1)
        ano_rec    = float(rec_arr[1]) if len(rec_arr) > 1 else 0.0
        macro_prec = float(prec_arr.mean())
        macro_f1   = float(f1_arr.mean())
        results[k] = (acc, macro_prec, ano_rec, macro_f1)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(args):
    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    train_dataset, emb_dim, num_keys, num_coms = generate_train(
        TRAIN_CSV, TEMPLATES_CSV, EMB_PATH, COM_PATH, args.window_size)
    dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                            shuffle=True, pin_memory=False)

    normal_sessions  = generate_test(TEST_NOR_CSV, TEMPLATES_CSV, EMB_PATH,
                                     COM_PATH, args.window_size)
    anomaly_sessions = generate_test(TEST_ANO_CSV, TEMPLATES_CSV, EMB_PATH,
                                     COM_PATH, args.window_size)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = CSCLog(
        input_size  = emb_dim,
        com_num     = num_coms,
        hidden_size = args.hidden_size,
        alpha       = args.alpha,
        pattern     = args.pattern,
        num_layers  = args.num_layers,
        num_keys    = num_keys,
        drop        = args.drop,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f'[train] Model parameters: {total_params:,}')

    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2)
    criterion = nn.CrossEntropyLoss()

    best_rec, best_epoch = 0.0, 0

    # ------------------------------------------------------------------
    # Epoch loop
    # ------------------------------------------------------------------
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for step, (seq, com, quan, timp, label) in enumerate(dataloader):
            seq   = seq.to(DEVICE)
            com   = com.to(DEVICE)
            quan  = quan.to(DEVICE)
            timp  = timp.to(DEVICE)
            label = label.to(DEVICE)

            optimizer.zero_grad()
            out  = model(seq, com, quan, timp)
            loss = criterion(out, label)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        avg_loss = np.mean(train_losses)
        print(f'Epoch [{epoch}/{args.epochs}]  loss={avg_loss:.4f}')

        # Evaluate
        res = evaluate_topk(normal_sessions, anomaly_sessions, model,
                            args.num_candidates, args.anomaly_rate)
        for k, (acc, prec, rec, f1) in res.items():
            print(f'  TopK={k} | Acc={acc:.3f}  Prec={prec:.3f}  '
                  f'AnoRec={rec:.3f}  F1={f1:.3f}')
            if rec > best_rec:
                best_rec = rec
                best_epoch = epoch
                state = {
                    'model':     model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch':     epoch,
                    'rec':       best_rec,
                    'args':      vars(args),
                    'emb_dim':   emb_dim,
                    'num_keys':  num_keys,
                    'num_coms':  num_coms,
                }
                torch.save(state, CKPT_PATH)
                print(f'  [train] → New best AnoRec={best_rec:.3f}, checkpoint saved.')
        best_k_rec = max(v[2] for v in res.values())
        scheduler.step(best_k_rec)

    print(f'\n[train] Best epoch: {best_epoch}  Best AnoRec: {best_rec:.3f}')
    print(f'[train] Checkpoint: {CKPT_PATH}')


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Train CSCLog (CPU)')
    p.add_argument('--window_size',    type=int,   default=DEFAULTS['window_size'])
    p.add_argument('--batch_size',     type=int,   default=DEFAULTS['batch_size'])
    p.add_argument('--epochs',         type=int,   default=DEFAULTS['epochs'])
    p.add_argument('--lr',             type=float, default=DEFAULTS['lr'])
    p.add_argument('--weight_decay',   type=float, default=DEFAULTS['weight_decay'])
    p.add_argument('--drop',           type=float, default=DEFAULTS['drop'])
    p.add_argument('--alpha',          type=float, default=DEFAULTS['alpha'])
    p.add_argument('--pattern',        type=int,   default=DEFAULTS['pattern'])
    p.add_argument('--num_layers',     type=int,   default=DEFAULTS['num_layers'])
    p.add_argument('--anomaly_rate',   type=int,   default=DEFAULTS['anomaly_rate'])
    p.add_argument('--hidden_size',    type=int,   nargs=5,
                   default=DEFAULTS['hidden_size'],
                   metavar=('FT', 'LSTM', 'MLP', 'GCN', 'OUT'))
    p.add_argument('--num_candidates', type=int,   nargs='+',
                   default=DEFAULTS['num_candidates'])
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
