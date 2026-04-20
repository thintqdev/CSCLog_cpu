#!/usr/bin/env bash
# run_pipeline.sh – Full CSCLog pipeline runner
# Usage: bash run_pipeline.sh [--skip-install] [--epochs N] [--batch_size B] [--window_size W]
# Run from the project root (where this script lives).

set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────────────
SKIP_INSTALL=false
EPOCHS=10
BATCH_SIZE=16
WINDOW_SIZE=9
NUM_CANDIDATES="1"

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-install)  SKIP_INSTALL=true ;;
        --epochs)        EPOCHS="$2";       shift ;;
        --batch_size)    BATCH_SIZE="$2";   shift ;;
        --window_size)   WINDOW_SIZE="$2";  shift ;;
        --num_candidates) NUM_CANDIDATES="$2"; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
    shift
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Colours ───────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()    { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
die()     { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }
step()    { echo; echo -e "${GREEN}══════════════════════════════════════════${NC}"; \
            echo -e "${GREEN} $*${NC}"; \
            echo -e "${GREEN}══════════════════════════════════════════${NC}"; }

# ── Pre-flight checks ─────────────────────────────────────────────────────────
command -v python >/dev/null 2>&1 || die "python not found – please install Python 3.9"

JSONL="src/dataset/data_full.jsonl"
[[ -f "$JSONL" ]] || die "Dataset not found: $JSONL"

BERT_DIR="model/bert"
[[ -f "$BERT_DIR/config.json" ]] || \
    warn "BERT config not found at $BERT_DIR/config.json – preprocess.py may fail."

# ── Install dependencies ──────────────────────────────────────────────────────
if [[ "$SKIP_INSTALL" == false ]]; then
    step "Step 0 – Installing dependencies"
    pip install --quiet \
        numpy pandas scikit-learn transformers python-dateutil regex

    # Detect CUDA availability to choose the right torch index URL
    if python -c "import torch; torch.cuda.is_available()" 2>/dev/null; then
        TORCH_INDEX="https://download.pytorch.org/whl/torch_stable.html"
    else
        TORCH_INDEX="https://download.pytorch.org/whl/torch_stable.html"
    fi

    pip install --quiet torch==1.12.0+cpu torchvision==0.13.0+cpu \
        -f "$TORCH_INDEX" || warn "torch install failed – may already be installed"

    pip install --quiet \
        torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric \
        -f https://data.pyg.org/whl/torch-1.12.0+cpu.html \
        || warn "torch-geometric install failed – may already be installed"

    info "Dependencies installed."
fi

# ── Step 1: Parse logs ────────────────────────────────────────────────────────
step "Step 1 – Parsing JSONL logs (Drain)"
python src/parse_logs.py
info "parse_logs.py completed."

# ── Step 2: Preprocess ───────────────────────────────────────────────────────
step "Step 2 – Preprocessing (embeddings + sessions)"
python src/preprocess.py
info "preprocess.py completed."

# ── Step 3: Train ────────────────────────────────────────────────────────────
step "Step 3 – Training CSCLog"
python src/train.py \
    --epochs        "$EPOCHS"        \
    --batch_size    "$BATCH_SIZE"    \
    --window_size   "$WINDOW_SIZE"   \
    --num_candidates $NUM_CANDIDATES
info "train.py completed."

# ── Step 4: Evaluate ─────────────────────────────────────────────────────────
step "Step 4 – Evaluating checkpoint"
python src/evaluate.py \
    --num_candidates $NUM_CANDIDATES
info "evaluate.py completed."

echo
info "Pipeline finished successfully. Outputs in src/dataset/result/"
