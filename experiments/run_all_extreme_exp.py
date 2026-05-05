# run_all_extreme_experiments.py
"""
Runs the full extreme weather experiment matrix:
  - All models x all prompting modes
  - Zero-shot: single run on all data
  - Few-shot:  5-fold cross-validation

Results are collected into a summary table printed at the end
and saved to results_log/summary_extreme_weather.csv
"""
import torch
import csv
import os
from datetime import datetime
from experiments.kfold_extreme import run_kfold_extreme
from experiments.utils import get_logging


# ─────────────────────────────────────────────────────────────────────────────
# Configuration — edit these before running
# ─────────────────────────────────────────────────────────────────────────────

JSONL_PATH   = "data/weather_extreme/10_dataset.jsonl"   # swap to 100_dataset when ready
LOG_DIR      = "results_log"
CONCURRENCY  = None
SEED         = 42

# For 10-sample test: k=3, n_shots=2
# For 100-sample real run: k=5, n_shots=4
K_FOLDS  = 3
N_SHOTS  = 2

MODELS = [
    "llama3:8b-instruct-q2_K",
    #"qwen2.5:7b-instruct-q2_K",
    #"mistral:7b-instruct-v0.2-q2_K",
]

MODES = [
    "zeroshot",
    "fewshot",
]

FORMAT_DBASE = "sentence"   # "sentence" or "jsonl"

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

    log.info(f"Starting full experiment matrix")
    log.info(f"  Dataset:     {JSONL_PATH}")
    log.info(f"  Models:      {MODELS}")
    log.info(f"  Modes:       {MODES}")
    log.info(f"  K-folds:     {K_FOLDS} (few-shot only)")
    log.info(f"  N-shots:     {N_SHOTS}")
    log.info(f"  Format:      {FORMAT_DBASE}")

    all_results = []   # collects (model, mode, scores) for summary table

    # ── Outer loop: models ────────────────────────────────────────────────────
    for model in MODELS:
        log.info(f"\n{'='*60}")
        log.info(f"MODEL: {model}")
        log.info(f"{'='*60}")

        # ── Inner loop: prompting modes ───────────────────────────────────────
        for mode in MODES:
            log.info(f"\n  Mode: {mode}")
            log.info(f"  {'─'*40}")

            try:
                scores = run_kfold_extreme(
                    jsonl_path=JSONL_PATH,
                    model=model,
                    prompting_mode=mode,
                    format_dbase=FORMAT_DBASE,
                    concurrency=CONCURRENCY,
                    k=K_FOLDS,
                    n_shots=N_SHOTS,
                    seed=SEED,
                    log_dir=LOG_DIR,
                )

                all_results.append({
                    "model":   model,
                    "mode":    mode,
                    "scores":  scores,
                })

                # Print per-run summary
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
                all_results.append({
                    "model":  model,
                    "mode":   mode,
                    "scores": {},
                    "error":  str(e),
                })

    # ── Print summary table to console ───────────────────────────────────────
    _print_summary_table(all_results, log)

    # ── Save summary to CSV ───────────────────────────────────────────────────
    csv_path = os.path.join(LOG_DIR, f"summary_extreme_weather_{timestamp}.csv")
    _save_summary_csv(all_results, csv_path)
    log.info(f"\nSummary saved to: {csv_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Summary helpers
# ─────────────────────────────────────────────────────────────────────────────

def _print_summary_table(all_results: list, log):
    """Print a clean results table to the log."""
    log.info(f"\n\n{'='*80}")
    log.info("FINAL SUMMARY TABLE")
    log.info(f"{'='*80}")

    header = f"{'Model':<15} {'Mode':<12} " + " ".join(
        f"{m[:8]:>10}" for m in REPORT_METRICS
    )
    log.info(header)
    log.info("─" * len(header))

    for entry in all_results:
        model  = entry["model"]
        mode   = entry["mode"]
        scores = entry.get("scores", {})

        if "error" in entry:
            row = f"{model:<15} {mode:<12}  ERROR: {entry['error']}"
        else:
            vals = []
            for metric in REPORT_METRICS:
                mean = scores.get(metric, 0)
                std  = scores.get(f"{metric}_std", None)
                # For zero-shot there is no std (single run)
                cell = f"{mean:.3f}" if std is None else f"{mean:.3f}±{std:.2f}"
                vals.append(f"{cell:>10}")
            row = f"{model:<15} {mode:<12} " + " ".join(vals)

        log.info(row)

    log.info(f"{'='*80}")


def _save_summary_csv(all_results: list, csv_path: str):
    """Save results to CSV for easy import into Excel / LaTeX table."""
    fieldnames = ["model", "mode"] + [
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
            row = {"model": entry["model"], "mode": entry["mode"]}
            for metric in REPORT_METRICS:
                row[metric]              = round(scores.get(metric, 0), 4)
                row[f"{metric}_std"]     = round(scores.get(f"{metric}_std", 0), 4)
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

