# CSCLog – Setup & Run Guide

## Requirements

- Python 3.9.x (tested on 3.9.13)
- pip 22+
- (Optional) CUDA-capable GPU for faster training

### Install dependencies

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu

pip3 install torch-geometric

pip3 install numpy pandas scikit-learn transformers python-dateutil regex
```

> If you have a GPU replace the index URL with the matching CUDA variant, e.g.  
> `--index-url https://download.pytorch.org/whl/cu121`

### Pre-trained BERT model

The BERT tokeniser / weights must be present at `model/bert/`.  
The directory should contain at minimum:

```
model/bert/
    config.json
    tokenizer.json
    tokenizer_config.json
    vocab.txt
    pytorch_model.bin      ← download separately
```

Download from [bert-base-uncased on Hugging Face](https://huggingface.co/bert-base-uncased/tree/main)
or run:

```bash
python3 - <<'EOF'
from transformers import AutoTokenizer, BertModel
tok = AutoTokenizer.from_pretrained('bert-base-uncased')
mdl = BertModel.from_pretrained('bert-base-uncased')
tok.save_pretrained('model/bert')
mdl.save_pretrained('model/bert')
EOF
```

---

## Dataset

Place the JSONL log file at:

```
src/dataset/data_full.jsonl
```

Each line must be a JSON object with the following fields:

| Field        | Description                    | Example                        |
|--------------|--------------------------------|--------------------------------|
| `@timestamp` | ISO-8601 timestamp             | `2026-04-20T03:56:15.062Z`     |
| `level`      | Log level                      | `INFO`                         |
| `host`       | Source host name               | `controller`                   |
| `module`     | Component / module name        | `nova.api.openstack.requestlog`|
| `message`    | Raw log message text           | `OPTIONS / => ...`             |

---

## Pipeline

```
data_full.jsonl
      │
      ▼
 parse_logs.py          →  data_full_structured.csv
                            data_full_templates.csv
      │
      ▼
 preprocess.py          →  data_full_sentences_emb.json
                            data_full_component.json
                            train_normal.csv
                            test_normal.csv
                            test_anomaly.csv
      │
      ▼
  train.py              →  csclog_best.pth
      │
      ▼
 evaluate.py            →  prints metrics
```

All outputs are written to `src/dataset/result/`.

### Step 1 – Parse logs

```bash
python src/parse_logs.py
```

### Step 2 – Preprocess

```bash
python src/preprocess.py
```

### Step 3 – Train

```bash
python src/train.py [--epochs 10] [--batch_size 16] [--window_size 9]
```

| Argument           | Default | Description                      |
|--------------------|---------|----------------------------------|
| `--epochs`         | 10      | Number of training epochs        |
| `--batch_size`     | 16      | Mini-batch size                  |
| `--window_size`    | 9       | Sliding window length            |
| `--lr`             | 1e-3    | Learning rate                    |
| `--num_candidates` | 1       | Top-K for detection (can be multiple, e.g. `1 5 10`) |
| `--anomaly_rate`   | 1       | Min misses to flag a session as anomaly |

### Step 4 – Evaluate

```bash
python src/evaluate.py [--checkpoint src/dataset/result/csclog_best.pth] \
                       [--num_candidates 1 5 10]
```

### Alternative – Jupyter notebook

```bash
jupyter notebook main.ipynb
```

---

## Directory Structure

```
CSCLog/
├── main.ipynb
├── SETUP.md
├── README.md
├── model/
│   └── bert/               ← BERT weights
├── src/
│   ├── parse_logs.py       ← Step 1
│   ├── preprocess.py       ← Step 2
│   ├── train.py            ← Step 3
│   ├── evaluate.py         ← Step 4
│   ├── model.py
│   └── dataset/
│       ├── data_full.jsonl ← input
│       └── result/         ← generated outputs
└── utils/
    ├── Drain.py
    ├── pytorchtools.py
    └── sentence_embding.py
```
