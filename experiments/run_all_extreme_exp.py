# run_all_extreme_experiments.py
"""
Entry point for extreme weather detection experiments.

Experiment matrix:
  Models  : Llama / Mistral / Qwen at multiple quantization levels
            (q2_K | q4_K_M | q8_0 | f16)
  Modes   : zeroshot | fewshot | ragfs
  CV      : stratified k-fold (few-shot and ragfs); single run (zeroshot)

RAG-FS mode (ragfs):
  Within each fold, embeds the fold's train pool with sentence-transformers,
  then for each test datum retrieves the N_SHOTS_RAG most similar training
  examples to build a dynamic few-shot prompt.

Run from project root:
    python -m experiments.run_all_extreme_exp
"""

import torch
import csv
import os
from datetime import datetime
from experiments.kfold_extreme import run_multiseed_extreme
from experiments.utils import get_logging


# ─────────────────────────────────────────────────────────────────────────────
# Configuration — edit these before running
# ─────────────────────────────────────────────────────────────────────────────

JSONL_PATH   = os.environ.get("EXTREME_DATA", "data/weather_extreme/100_dataset.jsonl")
LOG_DIR      = os.environ.get("LOG_DIR",       "results_log")
CONCURRENCY  = None
MAX_SAMPLES  = int(os.environ.get("MAX_SAMPLES", "100"))
N_SHOTS      = int(os.environ.get("N_SHOTS", "4"))
SEEDS        = [int(s) for s in os.environ.get("SEEDS", "42,123,999").split(",")]

# RAG-FS configuration — retrieved examples per test datum
N_SHOTS_RAG = int(os.environ.get("N_SHOTS_RAG", "4"))
RAG_MODEL   = "all-MiniLM-L6-v2"

# ── Quantization comparison matrix ───────────────────────────────────────────
# Override via env var: MODELS=llama3:8b-instruct-q2_K,mistral:7b-instruct-v0.2-q2_K
# Available quant variants: q2_K | q4_K_M | q8_0 | f16
#   llama3:8b-instruct-{quant}
#   mistral:7b-instruct-v0.2-{quant}
#   qwen2.5:7b-instruct-{quant}
MODELS = os.environ.get(
    "MODELS",
    "llama3:8b-instruct-q2_K",
).split(",")

# ── Prompting modes ───────────────────────────────────────────────────────────
# Override via env var: MODES=zeroshot  or  MODES=zeroshot,fewshot,ragfs
# zeroshot : no examples, instructions only
# fewshot  : stratified static examples selected from train fold
# ragfs    : per-datum retrieval from fold's train pool (Opsi 2)
MODES = os.environ.get("MODES", "zeroshot,fewshot,ragfs").split(",")

FORMAT_DBASE = "sentence"

# Metrics to include in the summary table
REPORT_METRICS = ["accuracy", "precision", "recall", "f1",
                  "type_accuracy", "strict", "lenient"]


# ─────────────────────────────────────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    log = get_logging(
        name="run_all_extreme",
        log_dir=LOG_DIR,
        log_prefix=f"run_all_extreme_{timestamp}",
    )

    log.info(f"Starting extreme weather experiment matrix")
    log.info(f"  Dataset:     {JSONL_PATH}")
    log.info(f"  Models:      {MODELS}")
    log.info(f"  Modes:       {MODES}")
    log.info(f"  K-folds:     {K_FOLDS} (few-shot / ragfs)")
    log.info(f"  N-shots:     {N_SHOTS} (fewshot) / {N_SHOTS_RAG} (ragfs)")
    log.info(f"  Format:      {FORMAT_DBASE}")

    all_results = []

    # ── Outer loop: models ────────────────────────────────────────────────────
    for model in MODELS:
        log.info(f"\n{'='*60}")
        log.info(f"MODEL: {model}")
        log.info(f"  Quant level: {_get_quant_level(model)}")
        log.info(f"{'='*60}")

        # ── Inner loop: prompting modes ───────────────────────────────────────
        for mode in MODES:
            log.info(f"\n  Mode: {mode}")
            log.info(f"  {'─'*40}")

            try:
                # N_SHOTS_RAG is passed for ragfs; N_SHOTS for static fewshot.
                # run_multiseed_extreme uses n_shots for both — the retriever in
                # ragfs mode will use it as k for retrieve().
                n_shots_for_mode = N_SHOTS_RAG if mode == "ragfs" else N_SHOTS

                scores = run_multiseed_extreme(
                    jsonl_path=JSONL_PATH,
                    model=model,
                    prompting_mode=mode,
                    format_dbase=FORMAT_DBASE,
                    concurrency=CONCURRENCY,
                    n_shots=n_shots_for_mode,
                    seeds=SEEDS,
                    log_dir=LOG_DIR,
                    max_samples=MAX_SAMPLES,
                )

                all_results.append({
                    "model":  model,
                    "quant":  _get_quant_level(model),
                    "family": _get_model_family(model),
                    "mode":   mode,
                    "scores": scores,
                })

                log.info(f"\n  [{model} | {mode}] Results:")
                for metric in REPORT_METRICS:
                    mean = scores.get(metric, 0)
                    std  = scores.get(f"{metric}_std", None)
                    if std is not None:
                        log.info(f"    {metric:20s}: {mean:.4f} ± {std:.4f}")
                    else:
                        log.info(f"    {metric:20s}: {mean:.4f}")

            except Exception as e:
                log.info(f"  [ERROR] {model} | {mode}: {e}")
                import traceback
                log.info(traceback.format_exc())
                all_results.append({
                    "model":  model,
                    "quant":  _get_quant_level(model),
                    "family": _get_model_family(model),
                    "mode":   mode,
                    "scores": {},
                    "error":  str(e),
                })

    _print_summary_table(all_results, log)

    csv_path = os.path.join(LOG_DIR, f"summary_extreme_weather_{timestamp}.csv")
    _save_summary_csv(all_results, csv_path)
    log.info(f"\nSummary saved to: {csv_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Model name helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_quant_level(model: str) -> str:
    for quant in ("f16", "q8_0", "q6_K", "q5_K_M", "q4_K_M", "q4_K_S",
                  "q3_K_M", "q2_K"):
        if quant.lower() in model.lower():
            return quant
    return "unknown"


def _get_model_family(model: str) -> str:
    return model.split(":")[0]


# ─────────────────────────────────────────────────────────────────────────────
# Summary helpers
# ─────────────────────────────────────────────────────────────────────────────

def _print_summary_table(all_results: list, log):
    log.info(f"\n\n{'='*90}")
    log.info("FINAL SUMMARY TABLE — Extreme Weather Detection")
    log.info(f"{'='*90}")

    header = f"{'Model':<32} {'Quant':<10} {'Mode':<10} " + " ".join(
        f"{m[:8]:>10}" for m in REPORT_METRICS
    )
    log.info(header)
    log.info("─" * len(header))

    for entry in all_results:
        model  = entry["model"]
        quant  = entry.get("quant", "")
        mode   = entry["mode"]
        scores = entry.get("scores", {})

        if "error" in entry:
            row = f"{model:<32} {quant:<10} {mode:<10}  ERROR: {entry['error']}"
        else:
            vals = []
            for metric in REPORT_METRICS:
                mean = scores.get(metric, 0)
                std  = scores.get(f"{metric}_std", None)
                cell = f"{mean:.3f}" if std is None else f"{mean:.3f}±{std:.2f}"
                vals.append(f"{cell:>10}")
            row = f"{model:<32} {quant:<10} {mode:<10} " + " ".join(vals)

        log.info(row)

    log.info(f"{'='*90}")


def _save_summary_csv(all_results: list, csv_path: str):
    fieldnames = ["model", "family", "quant", "mode"] + [
        col
        for metric in REPORT_METRICS
        for col in [metric, f"{metric}_std"]
    ] + ["tp", "fp", "fn", "tn"]

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for entry in all_results:
            if "error" in entry:
                continue
            scores = entry.get("scores", {})
            row = {
                "model":  entry["model"],
                "family": entry.get("family", ""),
                "quant":  entry.get("quant", ""),
                "mode":   entry["mode"],
            }
            for metric in REPORT_METRICS:
                row[metric]          = round(scores.get(metric, 0), 4)
                row[f"{metric}_std"] = round(scores.get(f"{metric}_std", 0), 4)
            for cm_key in ["tp", "fp", "fn", "tn"]:
                row[cm_key] = scores.get(cm_key, 0)
            writer.writerow(row)


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"\n[CUDA] GPU detected: {gpu_name} ({gpu_mem:.1f} GB VRAM)")
        print(f"[CUDA] LLM concurrency will be increased automatically")
        print(f"[CUDA] Make sure Ollama server is running with GPU support\n")
    else:
        print("\n[CPU] No GPU detected — running on CPU\n")

    main()
