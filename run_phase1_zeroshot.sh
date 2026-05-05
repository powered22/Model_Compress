#!/usr/bin/env bash
# run_phase1_zeroshot.sh
# ──────────────────────────────────────────────────────────────────────────────
# Phase 1 launcher: zeroshot experiments across all 4 tasks.
#
# Usage:
#   bash run_phase1_zeroshot.sh                      # default: llama3 q2_K, all tasks
#   bash run_phase1_zeroshot.sh --model mistral:7b-instruct-v0.2-q2_K
#   bash run_phase1_zeroshot.sh --tasks weather,extreme  # subset of tasks
#   MODES=zeroshot,fewshot bash run_phase1_zeroshot.sh   # also run fewshot
#
# Env vars (can override from outside):
#   MODELS       comma-separated Ollama model tags (default: llama3:8b-instruct-q2_K)
#   MODES        comma-separated modes (default: zeroshot)
#   OLLAMA_HOST  Ollama server URL (default: http://localhost:11434)
#   LOG_DIR      output directory (default: results_log)
#
# To run on a second GPU with a separate Ollama instance:
#   CUDA_VISIBLE_DEVICES=1 OLLAMA_HOST=http://localhost:11435 ollama serve &
#   OLLAMA_HOST=http://localhost:11435 MODELS=mistral:7b-instruct-v0.2-q2_K \
#       bash run_phase1_zeroshot.sh --tasks weather,extreme &
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────────────
MODELS="${MODELS:-llama3:8b-instruct-q2_K}"
MODES="${MODES:-zeroshot}"
OLLAMA_HOST="${OLLAMA_HOST:-http://localhost:11434}"
LOG_DIR="${LOG_DIR:-results_log}"
MAX_SAMPLES="${MAX_SAMPLES:-100}"
TASKS="weather,extreme,energy"

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)   MODELS="$2";  shift 2 ;;
        --tasks)   TASKS="$2";   shift 2 ;;
        --modes)   MODES="$2";   shift 2 ;;
        --log-dir) LOG_DIR="$2"; shift 2 ;;
        --help|-h)
            sed -n '3,28p' "$0" | sed 's/^# //'
            exit 0
            ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

export MODELS MODES OLLAMA_HOST LOG_DIR MAX_SAMPLES

# ── Helpers ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()    { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# ── Pre-flight checks ─────────────────────────────────────────────────────────
info "Phase 1 — Zeroshot experiments"
info "  MODELS:      $MODELS"
info "  MODES:       $MODES"
info "  TASKS:       $TASKS"
info "  OLLAMA_HOST: $OLLAMA_HOST"
info "  LOG_DIR:     $LOG_DIR"
info "  MAX_SAMPLES: $MAX_SAMPLES"
echo ""

# Check Ollama is reachable
if ! curl -s --max-time 3 "${OLLAMA_HOST}/api/tags" > /dev/null 2>&1; then
    error "Ollama is not reachable at ${OLLAMA_HOST}"
    error "Start it with:  ollama serve"
    error "Or with GPU:    CUDA_VISIBLE_DEVICES=0 ollama serve"
    exit 1
fi
info "Ollama is reachable at ${OLLAMA_HOST}"

# Check each requested model is pulled
IFS=',' read -ra MODEL_LIST <<< "$MODELS"
for model in "${MODEL_LIST[@]}"; do
    model="$(echo "$model" | tr -d ' ')"
    if curl -s "${OLLAMA_HOST}/api/tags" | grep -q "\"${model}\""; then
        info "  Model present: $model"
    else
        warn "  Model NOT found: $model"
        warn "  Pull it with:  ollama pull $model"
        warn "  Continuing — experiment will fail at runtime for this model."
    fi
done
echo ""

mkdir -p "$LOG_DIR"

# ── Task runners ──────────────────────────────────────────────────────────────
run_weather() {
    info "━━━ TASK: weather forecasting ━━━"
    python -m experiments.run_all_weather_exp
    info "━━━ weather done ━━━"
    echo ""
}

run_extreme() {
    info "━━━ TASK: extreme weather detection ━━━"
    # For zeroshot only, 5 folds is excessive — use 3 to save time
    local kf="${K_FOLDS:-5}"
    K_FOLDS="$kf" python -m experiments.run_all_extreme_exp
    info "━━━ extreme done ━━━"
    echo ""
}

run_energy() {
    info "━━━ TASK: energy (forecast + extreme) ━━━"
    python -m experiments.run_all_energy_exp
    info "━━━ energy done ━━━"
    echo ""
}

# ── Dispatch ──────────────────────────────────────────────────────────────────
START_TIME=$(date +%s)

IFS=',' read -ra TASK_LIST <<< "$TASKS"
for task in "${TASK_LIST[@]}"; do
    task="$(echo "$task" | tr -d ' ')"
    case "$task" in
        weather) run_weather ;;
        extreme) run_extreme ;;
        energy)  run_energy  ;;
        *)
            warn "Unknown task '$task'. Valid: weather, extreme, energy"
            ;;
    esac
done

END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))
info "All tasks finished in $((ELAPSED/60))m $((ELAPSED%60))s"
info "Results written to: $LOG_DIR/"
