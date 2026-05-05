#!/bin/bash
# create_models.sh
#
# Pull all quantization variants for the three model families used in
# the quantization-performance trade-off study (Opsi 1).
#
# Usage:
#   bash create_models.sh
#
# Requires: Ollama ≥ 0.3 running locally (ollama serve)
#
# Quantization levels (from most compressed to full precision):
#   q2_K   — 2-bit  (~2.5 GB) : fastest inference, lowest accuracy
#   q4_K_M — 4-bit  (~4.5 GB) : recommended balance (good accuracy / speed)
#   q8_0   — 8-bit  (~8.5 GB) : near-lossless quality, slower
#   f16    — 16-bit (~16 GB)  : full precision baseline (GPU recommended)
#
# Note: Exact Ollama tag names may differ. Run `ollama search <name>`
# to verify availability before pulling.

set -e

echo "======================================================================"
echo "Pulling Llama 3 8B Instruct — all quantization levels"
echo "======================================================================"
ollama pull llama3:8b-instruct-q2_K
ollama pull llama3:8b-instruct-q4_K_M
ollama pull llama3:8b-instruct-q8_0
# ollama pull llama3:8b-instruct-f16   # ~16 GB — enable if VRAM allows

echo ""
echo "======================================================================"
echo "Pulling Mistral 7B Instruct v0.2 — all quantization levels"
echo "======================================================================"
ollama pull mistral:7b-instruct-v0.2-q2_K
ollama pull mistral:7b-instruct-v0.2-q4_K_M
ollama pull mistral:7b-instruct-v0.2-q8_0
# ollama pull mistral:7b-instruct-v0.2-f16

echo ""
echo "======================================================================"
echo "Pulling Qwen 2.5 7B Instruct — all quantization levels"
echo "======================================================================"
ollama pull qwen2.5:7b-instruct-q2_K
ollama pull qwen2.5:7b-instruct-q4_K_M
ollama pull qwen2.5:7b-instruct-q8_0
# ollama pull qwen2.5:7b-instruct-f16

echo ""
echo "======================================================================"
echo "All models pulled successfully."
echo ""
echo "Verify with: ollama list"
echo ""
echo "Disk space used (approx, per family):"
echo "  q2_K  : ~2.5 GB x 3 = ~7.5 GB"
echo "  q4_K_M: ~4.5 GB x 3 = ~13.5 GB"
echo "  q8_0  : ~8.5 GB x 3 = ~25.5 GB"
echo "  Total (without f16): ~46.5 GB"
echo "======================================================================"

# ── Legacy model creation from local Modelfiles (kept for backward compat) ──
# These were used in earlier experiments and can be re-enabled if needed.
#
# ollama create gemma_9b_q4:latest -f ./models/benchmark/gemma_9b_q4
# ollama create gemma_9b_q8:latest -f ./models/benchmark/gemma_9b_q8
# ollama create llama_8b_q4:latest -f ./models/benchmark/llama_8b_q4
# ollama create llama_8b_q8:latest -f ./models/benchmark/llama_8b_q8
# ollama create dolphin_8b_q4:latest -f ./models/benchmark/dolphin_8b_q4
# ollama create dolphin_8b_q8:latest -f ./models/benchmark/dolphin_8b_q8
# ollama create qwen_7b_q4:latest -f ./models/benchmark/qwen_7b_q4
# ollama create qwen_7b_q8:latest -f ./models/benchmark/qwen_7b_q8
# ollama create cogito_8b_q4:latest -f ./models/benchmark/cogito_8b_q4
# ollama create cogito_8b_q8:latest -f ./models/benchmark/cogito_8b_q8
# ollama create qwen_8b_q4:latest -f ./models/benchmark/qwen_8b_q4
# ollama create qwen_8b_q8:latest -f ./models/benchmark/qwen_8b_q8
# ollama create granite_8b_q4:latest -f ./models/benchmark/granite_8b_q4 --quantize q4_k_m
# ollama create granite_8b_q8:latest -f ./models/benchmark/granite_8b_q8
