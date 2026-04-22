"""
Step 2: Preprocess structured log → embeddings + sessions.
===========================================================
Optimizations vs previous version:
  1. BERT word_vec: batched tokenization + GPU inference (was 1 word at a time)
  2. build_sessions: fully vectorized with pandas (was iterrows over 4M rows)
     - pd.to_datetime once, vectorized diff, np.searchsorted for boundaries
  3. dateutil.parse removed from hot path entirely
  4. feature_select: numpy-accelerated TF-IDF matrix (was pure Python dicts)
  5. tqdm progress bars throughout
  6. BERT moved to GPU automatically if available
  7. Larger BERT batch size (128 words per forward pass)
"""

import sys
import os
import json
import math
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, BertModel

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

# ── Paths ─────────────────────────────────────────────────────────────────────
RESULT_DIR     = os.path.join(os.path.dirname(__file__), 'dataset', 'result')
OUTPUT_DIR     = RESULT_DIR
MODEL_PATH     = os.path.join(PROJECT_ROOT, 'model', 'bert')
BERT_HF_NAME   = 'bert-base-uncased'
_WEIGHT_FILES  = ('pytorch_model.bin', 'model.safetensors')

STRUCTURED_CSV = os.path.join(RESULT_DIR, 'data_full_structured.csv')
TEMPLATES_CSV  = os.path.join(RESULT_DIR, 'data_full_templates.csv')
EMB_OUTPUT     = os.path.join(OUTPUT_DIR, 'data_full_sentences_emb.json')
COM_OUTPUT     = os.path.join(OUTPUT_DIR, 'data_full_component.json')
TRAIN_CSV      = os.path.join(OUTPUT_DIR, 'train_normal.csv')
VAL_NOR_CSV    = os.path.join(OUTPUT_DIR, 'test_normal.csv')
VAL_ANO_CSV    = os.path.join(OUTPUT_DIR, 'test_anomaly.csv')

# ── Hyper-params ──────────────────────────────────────────────────────────────
WINDOW_SIZE    = 9
TRAIN_RATIO    = 0.7
GAP_SECONDS    = 60          # session boundary gap
BERT_BATCH     = 128         # words per BERT forward pass
ANOMALY_LEVELS = {'error', 'critical', 'err', 'crit', 'fatal', 'alert', 'emerg'}

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'[preprocess] Device: {DEVICE}')

# ─────────────────────────────────────────────────────────────────────────────
# 1. Text helpers
# ─────────────────────────────────────────────────────────────────────────────

STOPWORDS = {'in', 'on', 'with', 'by', 'for', 'at', 'about', 'under', 'of', 'to', 'from'}
_CLEAN_RE = re.compile(r'[^\w\u4e00-\u9fff]+')


def get_keys(sentence: str) -> list[str]:
    if not isinstance(sentence, str) or not sentence.strip():
        return []
    return [x for x in _CLEAN_RE.sub(' ', sentence.lower()).split()
            if x not in STOPWORDS]


def feature_select(list_words: list[list[str]]) -> list[tuple[str, float]]:
    """Vectorized TF-IDF using numpy instead of Python dicts."""
    # Vocabulary
    all_words = sorted({w for wl in list_words for w in wl})
    w2i       = {w: i for i, w in enumerate(all_words)}
    V, D      = len(all_words), len(list_words)

    if V == 0 or D == 0:
        return []

    # Term frequency (global)
    counts = np.zeros(V, dtype=np.float32)
    for wl in list_words:
        for w in wl:
            counts[w2i[w]] += 1
    tf = counts / counts.sum()

    # Document frequency
    doc_freq = np.zeros(V, dtype=np.float32)
    for wl in list_words:
        seen = set(wl)
        for w in seen:
            doc_freq[w2i[w]] += 1
    idf = np.log(D / (doc_freq + 1))

    tfidf = tf * idf
    order = np.argsort(tfidf)[::-1]
    return [(all_words[i], float(tfidf[i])) for i in order]


# ─────────────────────────────────────────────────────────────────────────────
# 2. Batched BERT embedding (GPU)
# ─────────────────────────────────────────────────────────────────────────────

def _bert_path() -> str:
    if any(os.path.exists(os.path.join(MODEL_PATH, f)) for f in _WEIGHT_FILES):
        print(f'[preprocess] Loading BERT from local path: {MODEL_PATH}')
        return MODEL_PATH
    print(f'[preprocess] BERT weights not found locally, downloading from HuggingFace …')
    return BERT_HF_NAME


def word_vec_batched(keys: list[tuple[str, float]],
                     tokenizer, model, batch_size: int = BERT_BATCH) -> dict:
    """
    Encode all (word, weight) pairs in batches on GPU.
    Returns {word: weighted_embedding_tensor} (tensors on CPU).
    """
    model.eval()
    words   = [w for w, _ in keys]
    weights = [wt for _, wt in keys]
    result  = {}

    bar = tqdm(range(0, len(words), batch_size), desc='  BERT batches',
               unit='batch') if HAS_TQDM else range(0, len(words), batch_size)

    with torch.no_grad():
        for start in bar:
            batch_words = words[start: start + batch_size]
            batch_wts   = weights[start: start + batch_size]

            encoded = tokenizer(
                batch_words, return_tensors='pt',
                padding=True, truncation=True, max_length=32
            )
            encoded = {k: v.to(DEVICE) for k, v in encoded.items()}
            out     = model(**encoded)                    # last_hidden_state: (B, L, H)
            # Mean pool over token dim → (B, H), then weight by TF-IDF score
            vecs    = out.last_hidden_state.mean(dim=1)   # (B, H)

            for i, (word, wt) in enumerate(zip(batch_words, batch_wts)):
                result[word] = (vecs[i] * wt).cpu()

    return result


def sentence_vec_batched(sentences: dict, keys: list[tuple[str, float]],
                         tokenizer, model) -> dict:
    """Aggregate word vectors into sentence vectors (vectorized)."""
    encode_keys = word_vec_batched(keys, tokenizer, model)

    # Stack all word vectors into a matrix for fast lookup
    vocab_words  = list(encode_keys.keys())
    vocab_matrix = torch.stack([encode_keys[w] for w in vocab_words])  # (V, H)
    w2i          = {w: i for i, w in enumerate(vocab_words)}

    emb_dim  = vocab_matrix.shape[1]
    result   = {}

    bar = tqdm(sentences.items(), desc='  sentence vecs',
               total=len(sentences), unit='template') if HAS_TQDM else sentences.items()

    for event_id, word_list in bar:
        idxs = [w2i[w] for w in word_list if w in w2i]
        if idxs:
            vec = vocab_matrix[idxs].sum(dim=0)   # sum of TF-IDF weighted vecs
        else:
            vec = torch.zeros(emb_dim)
        result[event_id] = vec.tolist()

    return result


def build_embeddings():
    print('[preprocess] Building sentence embeddings …')
    templates = pd.read_csv(TEMPLATES_CSV)

    sentences = {
        row['EventId']: get_keys(str(row['EventTemplate']))
        for _, row in templates.iterrows()
        if isinstance(row['EventId'], str) and row['EventId']
    }
    sentences = {k: v for k, v in sentences.items() if v}
    print(f'[preprocess] Templates with tokens: {len(sentences):,}')

    keys = feature_select(list(sentences.values()))
    print(f'[preprocess] Vocabulary size: {len(keys):,}')

    bert_path = _bert_path()
    tokenizer = AutoTokenizer.from_pretrained(bert_path)
    bert      = BertModel.from_pretrained(bert_path).to(DEVICE)
    bert.eval()

    if DEVICE.type == 'cuda':
        print(f'[preprocess] BERT on GPU: {torch.cuda.get_device_name(0)}')

    emb = sentence_vec_batched(sentences, keys, tokenizer, bert)

    with open(EMB_OUTPUT, 'w') as f:
        json.dump(emb, f)
    print(f'[preprocess] Embeddings saved → {EMB_OUTPUT}')
    return emb


# ─────────────────────────────────────────────────────────────────────────────
# 3. Component map
# ─────────────────────────────────────────────────────────────────────────────

def build_component_map(structured: pd.DataFrame) -> dict:
    print('[preprocess] Building component map …')
    # Vectorized component normalization
    components = (structured['Component']
                  .dropna()
                  .astype(str)
                  .str.split('[').str[0]
                  .str.split(':').str[0]
                  .str.strip()
                  .unique())
    com_map = {c: i for i, c in enumerate(sorted(set(components)))}
    with open(COM_OUTPUT, 'w') as f:
        json.dump(com_map, f)
    print(f'[preprocess] Component map: {len(com_map)} components → {COM_OUTPUT}')
    return com_map


# ─────────────────────────────────────────────────────────────────────────────
# 4. Build sessions — fully vectorized (no iterrows)
# ─────────────────────────────────────────────────────────────────────────────

def build_sessions(structured: pd.DataFrame, com_map: dict) -> list:
    """
    Vectorized session building using pandas.
    Avoids iterrows() over 4M rows — processes entire DataFrame at once.

    Session boundary: gap between consecutive timestamps > GAP_SECONDS.
    Returns list of (session_events, is_anomalous).
    """
    print('[preprocess] Building sessions (vectorized) …')
    df = structured.copy()

    # Normalize component key (vectorized)
    fallback_com = sorted(com_map.keys())[0]
    raw_com = (df['Component']
               .fillna('')
               .astype(str)
               .str.split('[').str[0]
               .str.split(':').str[0]
               .str.strip())
    df['com_key'] = raw_com.where(raw_com.isin(com_map), fallback_com)

    # Parse timestamps vectorized — much faster than dateutil in a loop
    print('[preprocess]   Parsing timestamps …')
    df['ts'] = pd.to_datetime(df['iso_time'], errors='coerce', utc=False)
    # Fill NaT with a safe default
    df['ts'] = df['ts'].fillna(pd.Timestamp('2000-01-01'))

    # Anomaly flag per row (vectorized)
    df['is_ano'] = df['Level'].fillna('').str.lower().str.strip().isin(ANOMALY_LEVELS)

    # Session boundary detection
    # A new session starts where: gap > GAP_SECONDS OR first row
    ts_series = df['ts']
    diff_secs = ts_series.diff().dt.total_seconds().fillna(0)
    new_session = (diff_secs > GAP_SECONDS)
    session_ids = new_session.cumsum()  # integer session id per row

    print(f'[preprocess]   Detected {session_ids.max() + 1:,} raw session boundaries …')

    # Aggregate sessions
    df['session_id'] = session_ids

    # Build event tuples: (EventId, com_key, iso_time_str)
    # Keep iso_time as string to match downstream expectation
    df['event_tuple'] = list(zip(df['EventId'], df['com_key'], df['iso_time']))

    sessions = []
    bar = (tqdm(df.groupby('session_id', sort=False),
                desc='  grouping sessions', unit='session')
           if HAS_TQDM
           else df.groupby('session_id', sort=False))

    for _, grp in bar:
        if len(grp) <= WINDOW_SIZE:
            continue
        event_list = grp['event_tuple'].tolist()
        is_anomaly = grp['is_ano'].any()
        sessions.append((event_list, bool(is_anomaly)))

    n_ano = sum(1 for _, a in sessions if a)
    print(f'[preprocess] Sessions: {len(sessions):,}  '
          f'(anomalous={n_ano:,}, normal={len(sessions)-n_ano:,})')
    return sessions


# ─────────────────────────────────────────────────────────────────────────────
# 5. Split and save
# ─────────────────────────────────────────────────────────────────────────────

def split_and_save(sessions: list):
    normal_sessions  = [s for s, is_ano in sessions if not is_ano]
    anomaly_sessions = [s for s, is_ano in sessions if is_ano]

    n_train        = int(len(normal_sessions) * TRAIN_RATIO)
    train_sessions = normal_sessions[:n_train]
    test_normal    = normal_sessions[n_train:]
    test_anomaly   = anomaly_sessions

    def to_df(session_list):
        return pd.DataFrame({'EventSequence': [str(s) for s in session_list]})

    to_df(train_sessions).to_csv(TRAIN_CSV,   index=False)
    to_df(test_normal).to_csv(VAL_NOR_CSV,    index=False)
    to_df(test_anomaly).to_csv(VAL_ANO_CSV,   index=False)

    total_test = len(test_normal) + len(test_anomaly)
    ratio = len(test_anomaly) / total_test if total_test else 0
    print(f'[preprocess] train_normal : {len(train_sessions):,} → {TRAIN_CSV}')
    print(f'[preprocess] test_normal  : {len(test_normal):,}    → {VAL_NOR_CSV}')
    print(f'[preprocess] test_anomaly : {len(test_anomaly):,}   → {VAL_ANO_CSV}')
    print(f'[preprocess] Anomaly ratio in test: {ratio:.1%}')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Embeddings (GPU-accelerated BERT)
    if not os.path.exists(EMB_OUTPUT):
        build_embeddings()
    else:
        print(f'[preprocess] Embeddings exist, skipping. ({EMB_OUTPUT})')

    # 2. Load structured log
    print('[preprocess] Loading structured CSV …')
    structured = pd.read_csv(STRUCTURED_CSV, engine='c',
                             na_filter=False, memory_map=True)
    print(f'[preprocess] Loaded {len(structured):,} log lines.')

    # 3. Component map
    com_map = build_component_map(structured)

    # 4. Sessions → split
    sessions = build_sessions(structured, com_map)
    split_and_save(sessions)

    print('[preprocess] Done.')


if __name__ == '__main__':
    main()