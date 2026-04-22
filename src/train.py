"""
Step 3: Train CSCLog – RAM-SAFE VERSION
============================================================================
Key fix: generate_test stores ONLY integer indices (ev_wins, com_wins, labels).
NO embeddings, NO QP matrices stored in memory.
Everything is computed on-the-fly per batch during evaluation.

RAM usage per session:
  - ev_wins:  (W, ws) int32
  - com_wins: (W, ws) int32
  - labels:   (W,)    int64
  - tm:       (W, ws) float32   [tiny, kept for model input]
  Total: ~W * (ws*12 + ws*4 + 8) bytes — ~100x less than storing embeddings
"""

import sys
import os
import json
import re
import argparse
import random
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import dateutil.parser

warnings.filterwarnings('ignore', message='.*torch-scatter.*')
warnings.filterwarnings('ignore', message='.*torch-cluster.*')
warnings.filterwarnings('ignore', message='.*torch-spline-conv.*')
warnings.filterwarnings('ignore', message='.*torch-sparse.*')

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print('[train] tqdm not found, using plain progress')

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.model import CSCLog

def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_everything(42)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'[train] Using device: {DEVICE}')

if DEVICE.type == 'cuda':
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    props = torch.cuda.get_device_properties(0)
    total_vram_gb = props.total_memory / 1e9
    print(f'[train] GPU: {props.name}  VRAM: {total_vram_gb:.1f} GB')

    if total_vram_gb < 12:
        print('[train] WARNING: Low VRAM detected, using conservative settings')
        SAFE_BATCH = 128
    elif total_vram_gb < 16:
        SAFE_BATCH = 256
    else:
        SAFE_BATCH = 512
else:
    SAFE_BATCH = 64

RESULT_DIR    = os.path.join(os.path.dirname(__file__), 'dataset', 'result')
TEMPLATES_CSV = os.path.join(RESULT_DIR, 'data_full_templates.csv')
EMB_PATH      = os.path.join(RESULT_DIR, 'data_full_sentences_emb.json')
COM_PATH      = os.path.join(RESULT_DIR, 'data_full_component.json')
TRAIN_CSV     = os.path.join(RESULT_DIR, 'train_normal.csv')
TEST_NOR_CSV  = os.path.join(RESULT_DIR, 'test_normal.csv')
TEST_ANO_CSV  = os.path.join(RESULT_DIR, 'test_anomaly.csv')
CKPT_PATH     = os.path.join(RESULT_DIR, 'csclog_best.pth')

DEFAULTS = dict(
    window_size       = 9,
    batch_size        = SAFE_BATCH,
    grad_accum        = 2,
    epochs            = 25,
    lr                = 5e-5,       # Reduced from 2e-4 to prevent NaN
    warmup_epochs     = 2,
    weight_decay      = 1e-4,       # Reduced weight decay too
    drop              = 0.2,
    hidden_size       = [128, 128, 128, 128, 128],
    alpha             = 0.8,
    pattern           = 1,
    num_layers        = 2,
    num_candidates    = [1],
    anomaly_rate      = 1,
    patience          = 6,
    compile_model     = False,
    eval_batch_size   = 512,
)

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


class _TrainDataset(Dataset):
    def __init__(self, sessions, mapping, emb, cop, emb_dim, num_keys, window_size):
        print('[dataset] Building lazy index...')
        ws = window_size

        emb_matrix = np.zeros((num_keys, emb_dim), dtype=np.float32)
        for ev, idx in mapping.items():
            if ev in emb:
                emb_matrix[idx] = emb[ev]
        self.emb_matrix = emb_matrix
        self.num_keys   = num_keys
        self.ws         = ws

        self._ev  = []
        self._com = []
        self._ts  = []

        index_rows = []
        lbl_list   = []

        for sid, seqs in enumerate(sessions):
            n = len(seqs)
            ev_idxs = np.array(
                [mapping.get(ev) if mapping.get(ev) is not None else 0
                 for ev, _, _ in seqs], dtype=np.int32)
            com_idxs = np.array(
                [cop.get(comp, 0) for _, comp, _ in seqs], dtype=np.int32)
            ts_vals = np.array(
                [_parse_ts_float(ts) for _, _, ts in seqs], dtype=np.float32)

            self._ev.append(ev_idxs)
            self._com.append(com_idxs)
            self._ts.append(ts_vals)

            for i in range(n - ws):
                lbl_idx = mapping.get(seqs[i + ws][0])
                if lbl_idx is None:
                    continue
                index_rows.append((sid, i))
                lbl_list.append(lbl_idx)

        self.index   = np.array(index_rows, dtype=np.int32)
        self.lbl_arr = np.array(lbl_list,   dtype=np.int64)

        mb_used = (self.emb_matrix.nbytes +
                   sum(arr.nbytes for arr in self._ev + self._com + self._ts) +
                   self.index.nbytes + self.lbl_arr.nbytes) / 1024**2

        print(f'[dataset] Built {len(self.lbl_arr):,} windows from '
              f'{len(self._ev):,} sessions (~{mb_used:.1f} MB)')

    def __len__(self):
        return len(self.lbl_arr)

    def __getitem__(self, idx):
        sid, start = int(self.index[idx, 0]), int(self.index[idx, 1])
        ws  = self.ws
        ev  = self._ev[sid][start: start + ws]
        com = self._com[sid][start: start + ws]
        ts  = self._ts[sid][start: start + ws]

        seq = self.emb_matrix[ev]
        tm  = (ts - ts[0]).astype(np.float32)
        qp  = np.bincount(ev.astype(np.intp),
                          minlength=self.num_keys).astype(np.float32)
        lbl = self.lbl_arr[idx]

        return (torch.from_numpy(seq),
                torch.from_numpy(com).long(),
                torch.from_numpy(qp),
                torch.from_numpy(tm),
                torch.tensor(lbl, dtype=torch.long))


def _load_artifacts(templates_csv, emb_path, com_path):
    temp_df  = pd.read_csv(templates_csv, index_col='EventId',
                           engine='c', na_filter=False, memory_map=True)
    mapping  = {idx: i for i, idx in enumerate(temp_df.index.unique())}
    emb      = json.load(open(emb_path))
    cop      = json.load(open(com_path))
    num_keys = len(mapping)
    emb_dim  = len(next(iter(emb.values())))
    return mapping, emb, cop, num_keys, emb_dim


def _build_emb_matrix(mapping, emb, num_keys, emb_dim):
    emb_matrix = np.zeros((num_keys, emb_dim), dtype=np.float32)
    for ev, idx in mapping.items():
        if ev in emb:
            emb_matrix[idx] = emb[ev]
    return emb_matrix


def generate_train(train_path, templates_csv, emb_path, com_path, window_size):
    mapping, emb, cop, num_keys, emb_dim = _load_artifacts(
        templates_csv, emb_path, com_path)
    train_df = pd.read_csv(train_path, engine='c', na_filter=False, memory_map=True)
    sessions = [s for s in (_safe_eval(r['EventSequence'])
                            for _, r in train_df.iterrows()) if s is not None]
    dataset = _TrainDataset(sessions, mapping, emb, cop, emb_dim, num_keys, window_size)
    print(f'[train] Training: {len(dataset):,} samples, '
          f'emb_dim={emb_dim}, keys={num_keys}, coms={len(cop)}')
    return dataset, emb_dim, num_keys, len(cop)


def generate_test(log_path, templates_csv, emb_path, com_path, window_size):
    mapping, emb, cop, num_keys, emb_dim = _load_artifacts(
        templates_csv, emb_path, com_path)

    df = pd.read_csv(log_path, engine='c', na_filter=False, memory_map=True)
    print(f'[train] Loading {len(df):,} test sessions from {os.path.basename(log_path)}...')

    raw_seqs = [_safe_eval(r) for r in df['EventSequence']]

    sessions   = []
    total_wins = 0

    iterator = tqdm(raw_seqs, desc='  Processing', leave=False, dynamic_ncols=True) \
               if HAS_TQDM else raw_seqs

    for seqs in iterator:
        if seqs is None or len(seqs) <= window_size:
            continue

        n = len(seqs)
        n_windows = n - window_size
        if n_windows <= 0:
            continue

        events   = [ev   for ev, _, _ in seqs]
        comps    = [comp for _, comp, _ in seqs]

        ev_idxs  = np.fromiter((mapping.get(ev, 0) for ev in events),
                               dtype=np.int32, count=n)
        com_idxs = np.fromiter((cop.get(c, 0) for c in comps),
                               dtype=np.int32, count=n)
        ts_vals  = np.arange(n, dtype=np.float32)

        ev_wins  = sliding_window_view(ev_idxs,  window_size)[:n_windows].copy()
        com_wins = sliding_window_view(com_idxs, window_size)[:n_windows].copy()
        ts_wins  = sliding_window_view(ts_vals,  window_size)[:n_windows]

        tm = (ts_wins - ts_wins[:, :1]).astype(np.float32).copy()

        labels = np.array(
            [mapping.get(events[i + window_size], -1) for i in range(n_windows)],
            dtype=np.int64)

        sessions.append((ev_wins, com_wins, tm, labels))
        total_wins += n_windows

    mb = sum(
        s[0].nbytes + s[1].nbytes + s[2].nbytes + s[3].nbytes
        for s in sessions
    ) / 1024**2
    print(f'[train] Test: {len(sessions):,} sessions, '
          f'{total_wins:,} windows, RAM={mb:.0f} MB')
    return sessions, num_keys, emb_dim, mapping, emb, cop


def evaluate_topk(normal_sessions, anomaly_sessions, model,
                  num_candidates_list, anomaly_rate=1,
                  batch_size=512, use_amp=False,
                  emb_matrix=None, num_keys=None):
    assert emb_matrix is not None and num_keys is not None, \
        'emb_matrix and num_keys must be provided'

    model.eval()
    emb_t = torch.from_numpy(emb_matrix).to(DEVICE)

    def session_hit(sessions, k_list):
        if not sessions:
            return {k: [] for k in k_list}

        all_ev   = np.concatenate([s[0] for s in sessions], axis=0)
        all_com  = np.concatenate([s[1] for s in sessions], axis=0)
        all_timp = np.concatenate([s[2] for s in sessions], axis=0)
        all_lbl  = np.concatenate([s[3] for s in sessions], axis=0)
        sid_arr  = np.concatenate([
            np.full(len(s[3]), sid, dtype=np.int64)
            for sid, s in enumerate(sessions)
        ], axis=0)

        valid_mask = all_lbl >= 0
        all_ev   = all_ev[valid_mask]
        all_com  = all_com[valid_mask]
        all_timp = all_timp[valid_mask]
        sid_arr  = sid_arr[valid_mask]
        all_lbl  = all_lbl[valid_mask]

        if len(all_lbl) == 0:
            return {k: [0] * len(sessions) for k in k_list}

        n_sessions    = len(sessions)
        session_misses = {k: torch.zeros(n_sessions, dtype=torch.long, device=DEVICE)
                          for k in k_list}

        N = len(all_ev)
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            B   = end - start

            ev_b   = torch.from_numpy(
                all_ev[start:end].astype(np.int64)).to(DEVICE)
            com_b  = torch.from_numpy(
                all_com[start:end]).long().to(DEVICE)
            timp_b = torch.from_numpy(
                all_timp[start:end]).float().to(DEVICE)
            lab_b  = torch.from_numpy(
                all_lbl[start:end]).long().to(DEVICE)
            sid_b  = torch.from_numpy(
                sid_arr[start:end]).to(DEVICE)

            seq_b = emb_t[ev_b]

            qp_b = torch.zeros(B, num_keys, dtype=torch.float32, device=DEVICE)
            qp_b.scatter_add_(
                1,
                ev_b.reshape(B, -1),
                torch.ones(B, ev_b.shape[1], dtype=torch.float32, device=DEVICE)
            )

            with torch.no_grad():
                if use_amp:
                    with torch.autocast(device_type='cuda'):
                        out = model(seq_b, com_b, qp_b, timp_b)
                else:
                    out = model(seq_b, com_b, qp_b, timp_b)

            for k in k_list:
                topk      = torch.argsort(out, dim=1, descending=True)[:, :k]
                wrong     = ~(lab_b.unsqueeze(1) == topk).any(dim=1)
                wrong_sid = sid_b[wrong]
                if wrong_sid.numel() > 0:
                    session_misses[k].scatter_add_(
                        0, wrong_sid,
                        torch.ones(wrong_sid.numel(), dtype=torch.long, device=DEVICE))

        return {k: (session_misses[k] >= anomaly_rate).long().cpu().tolist()
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
        ano_prec = float(prec_arr[1]) if len(prec_arr) > 1 else 0.0
        ano_rec  = float(rec_arr[1])  if len(rec_arr)  > 1 else 0.0
        macro_f1 = float(f1_arr.mean())
        denom    = 4.0 * ano_prec + ano_rec
        fbeta2   = 5.0 * ano_prec * ano_rec / denom if denom > 0 else 0.0
        results[k] = (acc, ano_prec, ano_rec, macro_f1, fbeta2)
    return results


def _make_scheduler(optimizer, warmup_epochs, total_epochs, steps_per_epoch):
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps  = total_epochs  * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.05, 0.5 * (1.0 + np.cos(np.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train(args):
    train_dataset, emb_dim, num_keys, num_coms = generate_train(
        TRAIN_CSV, TEMPLATES_CSV, EMB_PATH, COM_PATH, args.window_size)

    n_workers = 4 if torch.cuda.is_available() else 2

    dataloader = DataLoader(
        train_dataset,
        batch_size         = args.batch_size,
        shuffle            = True,
        pin_memory         = (DEVICE.type == 'cuda'),
        num_workers        = n_workers,
        persistent_workers = True,
        prefetch_factor    = 2,
        drop_last          = True,
    )

    print(f'[train] DataLoader: batch={args.batch_size}, '
          f'accum={args.grad_accum} → effective={args.batch_size * args.grad_accum}, '
          f'workers={n_workers}')

    normal_sessions,  num_keys_t, emb_dim_t, mapping, emb, cop = generate_test(
        TEST_NOR_CSV, TEMPLATES_CSV, EMB_PATH, COM_PATH, args.window_size)
    anomaly_sessions, *_ = generate_test(
        TEST_ANO_CSV, TEMPLATES_CSV, EMB_PATH, COM_PATH, args.window_size)

    print(f'[train] Test data ready: {len(normal_sessions)} normal, '
          f'{len(anomaly_sessions)} anomaly sessions')

    emb_matrix = _build_emb_matrix(mapping, emb, num_keys, emb_dim)

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
    print(f'[train] Model: {total_params:,} parameters')

    optimizer = optim.AdamW(model.parameters(),
                            lr=args.lr, weight_decay=args.weight_decay)
    use_amp   = (DEVICE.type == 'cuda')
    criterion = nn.CrossEntropyLoss()

    steps_per_epoch = len(dataloader)
    scheduler = _make_scheduler(optimizer, args.warmup_epochs, args.epochs,
                                 steps_per_epoch)

    best_fbeta, best_epoch = 0.0, 0
    patience_counter = 0
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        if HAS_TQDM:
            bar = tqdm(dataloader, desc=f'Epoch {epoch}/{args.epochs}',
                       leave=False, dynamic_ncols=True)
        else:
            bar = dataloader
            print(f'\nEpoch {epoch}/{args.epochs}:')

        try:
            for acc_step, (seq, com, quan, timp, label) in enumerate(bar):
                seq   = seq.to(DEVICE,   non_blocking=True)
                com   = com.to(DEVICE,   non_blocking=True)
                quan  = quan.to(DEVICE,  non_blocking=True)
                timp  = timp.to(DEVICE,  non_blocking=True)
                label = label.to(DEVICE, non_blocking=True)

                if use_amp:
                    with torch.autocast(device_type='cuda'):
                        out  = model(seq, com, quan, timp)
                        loss = criterion(out, label) / args.grad_accum
                else:
                    out  = model(seq, com, quan, timp)
                    loss = criterion(out, label) / args.grad_accum

                if not torch.isfinite(loss):
                    print(f'\n[train] WARNING: non-finite loss at step {acc_step}, skipping batch')
                    print(f'  seq NaN={seq.isnan().any().item()} Inf={seq.isinf().any().item()}')
                    print(f'  out NaN={out.isnan().any().item()} Inf={out.isinf().any().item()}')
                    print(f'  label min={label.min().item()} max={label.max().item()} num_keys={num_keys}')
                    # Check model parameters for NaN
                    for name, param in model.named_parameters():
                        if param.isnan().any() or param.isinf().any():
                            print(f'  CORRUPT PARAM: {name}')
                    optimizer.zero_grad(set_to_none=True)
                    continue

                loss.backward()

                if (acc_step + 1) % args.grad_accum == 0:
                    # Clip gradients BEFORE checking for NaN (prevents explosion)
                    total_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    # Check if gradients are finite after clipping
                    grad_finite = True
                    for param in model.parameters():
                        if param.grad is not None:
                            if not torch.isfinite(param.grad).all():
                                grad_finite = False
                                break
                    
                    if not grad_finite:
                        print(f'\n[train] WARNING: Non-finite gradients at step {acc_step}, skipping update')
                        optimizer.zero_grad(set_to_none=True)
                        continue
                    
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                epoch_loss += loss.item() * args.grad_accum

                if HAS_TQDM:
                    mem = torch.cuda.memory_allocated() / 1e9 if DEVICE.type == 'cuda' else 0
                    bar.set_postfix(loss=f'{loss.item()*args.grad_accum:.4f}',
                                   mem=f'{mem:.1f}GB')
                elif (acc_step + 1) % 50 == 0:
                    print(f'  Step {acc_step+1}/{steps_per_epoch}: '
                          f'loss={loss.item()*args.grad_accum:.4f}')

        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print(f'\n[train] OOM at epoch {epoch}! '
                      f'Try reducing --batch_size or --hidden_size')
                raise
            else:
                raise

        # Handle leftover gradients at epoch end
        if steps_per_epoch % args.grad_accum != 0:
            total_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if torch.isfinite(torch.tensor(total_norm)):
                optimizer.step()
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        avg_loss = epoch_loss / steps_per_epoch

        if DEVICE.type == 'cuda':
            peak_mem = torch.cuda.max_memory_allocated() / 1e9
            torch.cuda.reset_peak_memory_stats()
            mem_str = f'  peak={peak_mem:.2f}GB'
        else:
            mem_str = ''

        print(f'Epoch {epoch}/{args.epochs}: loss={avg_loss:.4f}  '
              f'patience={patience_counter}/{args.patience}{mem_str}')

        res = evaluate_topk(
            normal_sessions, anomaly_sessions, model,
            args.num_candidates, args.anomaly_rate,
            batch_size = args.eval_batch_size,
            use_amp    = use_amp,
            emb_matrix = emb_matrix,
            num_keys   = num_keys,
        )

        for k, (acc, prec, rec, f1, fbeta) in res.items():
            print(f'  TopK={k}: Acc={acc:.3f} Prec={prec:.3f} '
                  f'Rec={rec:.3f} F1={f1:.3f} F2={fbeta:.3f}')

        best_k = max(v[4] for v in res.values())
        if best_k > best_fbeta:
            best_fbeta = best_k
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                'model':      model.state_dict(),
                'optimizer':  optimizer.state_dict(),
                'epoch':      epoch,
                'fbeta2_ano': best_fbeta,
                'args':       vars(args),
                'emb_dim':    emb_dim,
                'num_keys':   num_keys,
                'num_coms':   num_coms,
            }, CKPT_PATH)
            print(f'  ✓ Best F2={best_fbeta:.3f} saved')
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f'\nEarly stop (no improvement for {args.patience} epochs)')
            break

    print(f'\nBest: epoch {best_epoch}, F2={best_fbeta:.3f}')
    print(f'Saved: {CKPT_PATH}')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--window_size',     type=int,   default=DEFAULTS['window_size'])
    p.add_argument('--batch_size',      type=int,   default=DEFAULTS['batch_size'])
    p.add_argument('--grad_accum',      type=int,   default=DEFAULTS['grad_accum'])
    p.add_argument('--epochs',          type=int,   default=DEFAULTS['epochs'])
    p.add_argument('--lr',              type=float, default=DEFAULTS['lr'])
    p.add_argument('--warmup_epochs',   type=int,   default=DEFAULTS['warmup_epochs'])
    p.add_argument('--weight_decay',    type=float, default=DEFAULTS['weight_decay'])
    p.add_argument('--drop',            type=float, default=DEFAULTS['drop'])
    p.add_argument('--alpha',           type=float, default=DEFAULTS['alpha'])
    p.add_argument('--pattern',         type=int,   default=DEFAULTS['pattern'])
    p.add_argument('--num_layers',      type=int,   default=DEFAULTS['num_layers'])
    p.add_argument('--patience',        type=int,   default=DEFAULTS['patience'])
    p.add_argument('--eval_batch_size', type=int,   default=DEFAULTS['eval_batch_size'])
    p.add_argument('--hidden_size',     type=int,   nargs=5,
                   default=DEFAULTS['hidden_size'])
    p.add_argument('--num_candidates',  type=int,   nargs='+',
                   default=DEFAULTS['num_candidates'])
    p.add_argument('--anomaly_rate',    type=int,   default=DEFAULTS['anomaly_rate'])
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)