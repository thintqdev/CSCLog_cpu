"""
Step 2: Preprocess structured Linux log into:
  1. Sentence embeddings (BERT-based TF-IDF weighted)     → _sentences_emb.json
  2. Component index mapping                               → _component.json
  3. Session-windowed train / test CSV files               → train_normal.csv,
                                                             test_normal.csv,
                                                             test_anomaly.csv

Linux.log has no ground-truth anomaly labels, so we treat every session as
normal during training.  A heuristic is applied to mark rare sessions as
anomalous for testing purposes so the full evaluation pipeline can run.

Run AFTER parse_logs.py.
"""

import sys
import os
import json
import math
import operator
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, BertModel

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

# ── Paths ────────────────────────────────────────────────────────────────────
RESULT_DIR  = os.path.join(os.path.dirname(__file__), 'dataset', 'result')
OUTPUT_DIR  = RESULT_DIR                                  # same folder
MODEL_PATH      = os.path.join(PROJECT_ROOT, 'model', 'bert')
BERT_HF_NAME    = 'bert-base-uncased'
_WEIGHT_FILES   = ('pytorch_model.bin', 'model.safetensors')

STRUCTURED_CSV = os.path.join(RESULT_DIR, 'data_full_structured.csv')
TEMPLATES_CSV  = os.path.join(RESULT_DIR, 'data_full_templates.csv')

EMB_OUTPUT  = os.path.join(OUTPUT_DIR, 'data_full_sentences_emb.json')
COM_OUTPUT  = os.path.join(OUTPUT_DIR, 'data_full_component.json')
TRAIN_CSV   = os.path.join(OUTPUT_DIR, 'train_normal.csv')
VAL_NOR_CSV = os.path.join(OUTPUT_DIR, 'test_normal.csv')
VAL_ANO_CSV = os.path.join(OUTPUT_DIR, 'test_anomaly.csv')

# ── Hyper-params ─────────────────────────────────────────────────────────────
WINDOW_SIZE        = 9      # must match train.py
TRAIN_RATIO        = 0.7
# Log levels that indicate an anomalous event
ANOMALY_LEVELS     = {'error', 'critical', 'err', 'crit', 'fatal', 'alert', 'emerg'}


# ─────────────────────────────────────────────────────────────────────────────
# 1. Embedding helpers (from utils/sentence_embding.py)
# ─────────────────────────────────────────────────────────────────────────────

STOPWORDS = {'in', 'on', 'with', 'by', 'for', 'at', 'about', 'under', 'of', 'to', 'from'}


def get_keys(sentence) -> list:
    if not isinstance(sentence, str) or not sentence.strip():
        return []
    line = sentence.lower()
    line = re.sub(r'[^\w\u4e00-\u9fff]+', ' ', line)
    return [x for x in line.split() if x not in STOPWORDS]


def feature_select(list_words):
    doc_frequency = defaultdict(int)
    for word_list in list_words:
        for w in word_list:
            doc_frequency[w] += 1

    total = sum(doc_frequency.values())
    word_tf = {w: doc_frequency[w] / total for w in doc_frequency}

    doc_num = len(list_words)
    word_doc = defaultdict(int)
    for w in doc_frequency:
        for wl in list_words:
            if w in wl:
                word_doc[w] += 1
    word_idf = {w: math.log(doc_num / (word_doc[w] + 1)) for w in doc_frequency}

    word_tfidf = {w: word_tf[w] * word_idf[w] for w in doc_frequency}
    return sorted(word_tfidf.items(), key=operator.itemgetter(1), reverse=True)


def word_vec(keys, tokenizer, model):
    encode_keys = {}
    model.eval()
    with torch.no_grad():
        for word, weight in keys:
            encoded = tokenizer(word, return_tensors='pt')
            out = model(**encoded)
            encode_keys[word] = torch.mul(
                out[0].mean(dim=1, keepdim=False)[0], weight
            )
    return encode_keys


def sentence_vec(sentences: dict, keys, tokenizer, model) -> dict:
    encode_keys = word_vec(keys, tokenizer, model)
    first_ = list(encode_keys.values())[0]
    encode_sentence = {}
    for event_id, word_list in sentences.items():
        vec = torch.zeros_like(first_)
        for w in word_list:
            if w in encode_keys:
                vec += encode_keys[w]
        encode_sentence[event_id] = vec.tolist()
    return encode_sentence


# ─────────────────────────────────────────────────────────────────────────────
# 2. Build sentence embeddings
# ─────────────────────────────────────────────────────────────────────────────

def _bert_path() -> str:
    """Return local MODEL_PATH if weights are present, else fall back to HuggingFace Hub."""
    if any(os.path.exists(os.path.join(MODEL_PATH, f)) for f in _WEIGHT_FILES):
        return MODEL_PATH
    print(f'[preprocess] Weights not found in {MODEL_PATH}, '
          f'downloading "{BERT_HF_NAME}" from HuggingFace Hub …')
    return BERT_HF_NAME


def build_embeddings():
    print('[preprocess] Building sentence embeddings …')
    templates = pd.read_csv(TEMPLATES_CSV)

    sentences = {
        row['EventId']: get_keys(row['EventTemplate'])
        for _, row in templates.iterrows()
        if isinstance(row['EventId'], str) and row['EventId']
    }
    # Drop entries that produced no tokens (avoids downstream divide-by-zero)
    sentences = {k: v for k, v in sentences.items() if v}
    keys = feature_select(list(sentences.values()))

    tokenizer = AutoTokenizer.from_pretrained(_bert_path())
    bert = BertModel.from_pretrained(_bert_path())
    bert.eval()

    emb = sentence_vec(sentences, keys, tokenizer, bert)
    with open(EMB_OUTPUT, 'w') as f:
        json.dump(emb, f)
    print(f'[preprocess] Embeddings saved → {EMB_OUTPUT}')
    return emb


# ─────────────────────────────────────────────────────────────────────────────
# 3. Build component index
# ─────────────────────────────────────────────────────────────────────────────

def build_component_map(structured: pd.DataFrame) -> dict:
    print('[preprocess] Building component map …')
    components = structured['Component'].dropna().unique().tolist()
    # strip trailing colon / version noise
    components = list({c.split('[')[0].split(':')[0].strip() for c in components})
    com_map = {c: i for i, c in enumerate(sorted(components))}
    with open(COM_OUTPUT, 'w') as f:
        json.dump(com_map, f)
    print(f'[preprocess] Component map saved → {COM_OUTPUT}  ({len(com_map)} components)')
    return com_map


# ─────────────────────────────────────────────────────────────────────────────
# 4. Build sessions and split train / test
#
# Linux.log is a single contiguous syslog file; we group by Host+Component
# bursts separated by >60 s as "sessions".  Each session row is stored as
# EventSequence  =  list of (EventId, component_key, ISO-timestamp) tuples.
# ─────────────────────────────────────────────────────────────────────────────

def build_sessions(structured: pd.DataFrame, com_map: dict, gap_seconds: int = 60):
    """
    Group consecutive log lines that belong to the same component-host pair
    within `gap_seconds` into one session.
    Returns a list of sessions; each session is a list of
    (EventId, component_key, iso_timestamp).
    """
    # iso_time column is already present (written by parse_logs.py from @timestamp)
    structured = structured.copy()

    # Normalise component key to match com_map
    def norm_com(c):
        key = str(c).split('[')[0].split(':')[0].strip()
        return key if key in com_map else list(com_map.keys())[0]

    structured['com_key'] = structured['Component'].apply(norm_com)

    sessions      = []
    current_session = []
    current_has_anomaly = False
    prev_ts = None

    for _, row in structured.iterrows():
        ts = row['iso_time']
        # detect session boundary: gap > gap_seconds or missing event
        if prev_ts is not None:
            try:
                from dateutil.parser import parse as dtparse
                delta = (dtparse(ts) - dtparse(prev_ts)).seconds
            except Exception:
                delta = 0
            if delta > gap_seconds:
                if len(current_session) > WINDOW_SIZE:
                    sessions.append((current_session, current_has_anomaly))
                current_session = []
                current_has_anomaly = False

        level = str(row.get('Level', '')).lower().strip()
        if level in ANOMALY_LEVELS:
            current_has_anomaly = True

        current_session.append((row['EventId'], row['com_key'], ts))
        prev_ts = ts

    if len(current_session) > WINDOW_SIZE:
        sessions.append((current_session, current_has_anomaly))

    n_anomaly = sum(1 for _, is_ano in sessions if is_ano)
    print(f'[preprocess] Total sessions: {len(sessions)}  '
          f'(anomalous by level: {n_anomaly}, '
          f'normal: {len(sessions) - n_anomaly})')
    return sessions


def split_and_save(sessions: list):
    """
    Split sessions into train_normal, test_normal, test_anomaly.

    Anomaly label: a session is anomalous if it contains at least one log
    line whose Level is ERROR, CRITICAL, or equivalent (see ANOMALY_LEVELS).
    This is semantically correct for operational log data.

    Split strategy:
      - Training: first TRAIN_RATIO of *normal* sessions only
        (CSCLog is trained on normal behaviour exclusively)
      - Test normal:  remaining normal sessions
      - Test anomaly: all anomalous sessions (any split position)
    """
    normal_sessions  = [s for s, is_ano in sessions if not is_ano]
    anomaly_sessions = [s for s, is_ano in sessions if is_ano]

    n_train = int(len(normal_sessions) * TRAIN_RATIO)
    train_sessions = normal_sessions[:n_train]
    test_normal    = normal_sessions[n_train:]
    test_anomaly   = anomaly_sessions

    def to_df(session_list):
        return pd.DataFrame([{'EventSequence': str(s)} for s in session_list])

    to_df(train_sessions).to_csv(TRAIN_CSV,   index=False)
    to_df(test_normal).to_csv(VAL_NOR_CSV,    index=False)
    to_df(test_anomaly).to_csv(VAL_ANO_CSV,   index=False)

    total_test = len(test_normal) + len(test_anomaly)
    ratio = len(test_anomaly) / total_test if total_test else 0
    print(f'[preprocess] train_normal:  {len(train_sessions)} sessions → {TRAIN_CSV}')
    print(f'[preprocess] test_normal:   {len(test_normal)} sessions → {VAL_NOR_CSV}')
    print(f'[preprocess] test_anomaly:  {len(test_anomaly)} sessions → {VAL_ANO_CSV}')
    print(f'[preprocess] Test anomaly ratio: {ratio:.1%}')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Sentence embeddings
    if not os.path.exists(EMB_OUTPUT):
        build_embeddings()
    else:
        print(f'[preprocess] Embeddings already exist, skipping. ({EMB_OUTPUT})')

    # 2. Load structured log
    print('[preprocess] Loading structured log …')
    structured = pd.read_csv(STRUCTURED_CSV)
    print(f'[preprocess] Loaded {len(structured)} log lines.')

    # 3. Component map
    com_map = build_component_map(structured)

    # 4. Sessions → split
    sessions = build_sessions(structured, com_map)
    split_and_save(sessions)

    print('[preprocess] Done.')


if __name__ == '__main__':
    main()
