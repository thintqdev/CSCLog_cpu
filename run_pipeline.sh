#!/usr/bin/env bash
# run_pipeline.sh – Full CSCLog pipeline runner
# Usage: bash run_pipeline.sh [OPTIONS]
#
# Options:
#   --skip-install          Skip pip install step
#   --epochs N              Number of training epochs      (default: 25)
#   --batch_size B          Per-GPU mini-batch size        (default: 512)
#   --grad_accum G          Gradient accumulation steps    (default: 2)
#   --window_size W         Sliding window size            (default: 9)
#   --num_candidates K...   TopK candidates for eval       (default: 1)
#   --eval_batch_size B     Batch size during evaluation   (default: 1024)
#   --patience P            Early-stopping patience        (default: 6)
#   --compile               Enable torch.compile (PyTorch 2.0+)
#   --cpu                   Force CPU even if GPU present
#
# Run from the project root (where this script lives).

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
SKIP_INSTALL=false
EPOCHS=25
BATCH_SIZE=512
GRAD_ACCUM=2
WINDOW_SIZE=9
NUM_CANDIDATES="1"
EVAL_BATCH_SIZE=1024
PATIENCE=6
COMPILE=false
FORCE_CPU=false

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-install)     SKIP_INSTALL=true ;;
        --epochs)           EPOCHS="$2";           shift ;;
        --batch_size)       BATCH_SIZE="$2";       shift ;;
        --grad_accum)       GRAD_ACCUM="$2";       shift ;;
        --window_size)      WINDOW_SIZE="$2";      shift ;;
        --num_candidates)   NUM_CANDIDATES="$2";   shift ;;
        --eval_batch_size)  EVAL_BATCH_SIZE="$2";  shift ;;
        --patience)         PATIENCE="$2";         shift ;;
        --compile)          COMPILE=true ;;
        --cpu)              FORCE_CPU=true ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
    shift
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Colours ───────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; CYAN='\033[0;36m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
die()   { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }
step()  { echo; echo -e "${CYAN}══════════════════════════════════════════${NC}";
          echo -e "${CYAN} $*${NC}";
          echo -e "${CYAN}══════════════════════════════════════════${NC}"; }

# ── Detect GPU ────────────────────────────────────────────────────────────────
USE_GPU=false
if [[ "$FORCE_CPU" == false ]] && command -v nvidia-smi &>/dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo 0)
    if [[ "$GPU_COUNT" -gt 0 ]]; then
        USE_GPU=true
        info "Detected $GPU_COUNT GPU(s):"
        nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader \
            | while IFS= read -r line; do info "  GPU $line"; done
    fi
fi

if [[ "$USE_GPU" == false ]]; then
    warn "No GPU detected – running on CPU. Training will be slow."
    # Reduce defaults to something reasonable on CPU
    BATCH_SIZE=64
    GRAD_ACCUM=1
    EVAL_BATCH_SIZE=256
fi

# Print effective config
step "Configuration"
echo -e "  Epochs          : ${EPOCHS}"
echo -e "  Batch size      : ${BATCH_SIZE}  (grad_accum=${GRAD_ACCUM} → effective=$(( BATCH_SIZE * GRAD_ACCUM )))"
echo -e "  Window size     : ${WINDOW_SIZE}"
echo -e "  Num candidates  : ${NUM_CANDIDATES}"
echo -e "  Eval batch size : ${EVAL_BATCH_SIZE}"
echo -e "  Patience        : ${PATIENCE}"
echo -e "  torch.compile   : ${COMPILE}"
echo -e "  Device          : $( [[ "$USE_GPU" == true ]] && echo "CUDA (GPU)" || echo "CPU" )"

# ── Pre-flight checks ─────────────────────────────────────────────────────────
command -v python3 >/dev/null 2>&1 || die "python3 not found – please install Python 3.9+"

JSONL="src/dataset/data_full.jsonl"
[[ -f "$JSONL" ]] || die "Dataset not found: $JSONL"

BERT_DIR="model/bert"
[[ -f "$BERT_DIR/config.json" ]] || \
    warn "BERT config not found at $BERT_DIR/config.json – preprocess.py will download from HuggingFace."

# ── Timing helper ─────────────────────────────────────────────────────────────
_T0=$(date +%s)
elapsed() { echo $(( $(date +%s) - _T0 )); }
fmt_time() {
    local s=$1
    printf '%02dh %02dm %02ds' $(( s/3600 )) $(( (s%3600)/60 )) $(( s%60 ))
}

# ── Step 0: Install dependencies ─────────────────────────────────────────────
if [[ "$SKIP_INSTALL" == false ]]; then
    step "Step 0 – Installing dependencies"
    pip3 install --quiet numpy pandas scikit-learn transformers python-dateutil regex tqdm

    if [[ "$USE_GPU" == true ]]; then
        info "Installing PyTorch with CUDA support…"
        pip3 install --quiet torch torchvision \
            --index-url https://download.pytorch.org/whl/cu121
    else
        info "Installing PyTorch (CPU only)…"
        pip3 install --quiet torch torchvision \
            --index-url https://download.pytorch.org/whl/cpu
    fi

    pip3 install --quiet torch-geometric
    info "Dependencies installed."
fi

# ── Step 1: Parse logs ────────────────────────────────────────────────────────
step "Step 1 – Parsing JSONL logs (Drain)"
T1=$(date +%s)
python3 src/parse_logs.py
info "parse_logs.py completed in $(fmt_time $(( $(date +%s) - T1 )))."

# ── Step 2: Preprocess ───────────────────────────────────────────────────────
step "Step 2 – Preprocessing (embeddings + sessions)"
T2=$(date +%s)
python3 src/preprocess.py
info "preprocess.py completed in $(fmt_time $(( $(date +%s) - T2 )))."

# ── Step 3: Train ─────────────────────────────────────────────────────────────
step "Step 3 – Training CSCLog"
T3=$(date +%s)

TRAIN_ARGS=(
    --epochs          "$EPOCHS"
    --batch_size      "$BATCH_SIZE"
    --grad_accum      "$GRAD_ACCUM"
    --window_size     "$WINDOW_SIZE"
    --num_candidates  $NUM_CANDIDATES
    --eval_batch_size "$EVAL_BATCH_SIZE"
    --patience        "$PATIENCE"
)
[[ "$COMPILE" == true ]] && TRAIN_ARGS+=(--compile)

python3 src/train.py "${TRAIN_ARGS[@]}"
info "train.py completed in $(fmt_time $(( $(date +%s) - T3 )))."

# ── Step 4: Evaluate ─────────────────────────────────────────────────────────
step "Step 4 – Evaluating checkpoint"
T4=$(date +%s)

EVAL_ARGS=(--num_candidates $NUM_CANDIDATES)
[[ "$COMPILE" == true ]] && EVAL_ARGS+=(--compile)

python3 src/evaluate.py "${EVAL_ARGS[@]}"
info "evaluate.py completed in $(fmt_time $(( $(date +%s) - T4 )))."

# ── Done ──────────────────────────────────────────────────────────────────────
echo
info "Pipeline finished in $(fmt_time $(elapsed)). Outputs in src/dataset/result/"