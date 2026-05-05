# run_all_weather_experiments.py
"""
Entry point for short-term weather forecasting experiments.

Mirrors run_all_extreme_experiments.py but for the weather task.
Uses fixed held-out few-shot selection (one example per horizon)
instead of k-fold CV, because the weather task has no class labels
to stratify over.

Run from project root:
    python -m run_all_weather_experiments
"""

import os
import csv
import json
from datetime import datetime

from experiments.runner import WeatherExperiment, PersistenceExperiment
from experiments.utils import get_logging
from experiments.kfold_weather import (
    fixed_fewshot_weather_split,
    format_weather_fewshot_prompt,
    verify_split,
    load_jsonl,
)
import torch
from experiments.baselines.lstm_evaluator import LSTMEvaluator


# ─────────────────────────────────────────────────────────────────────────────
# Configuration — edit these before running
# ─────────────────────────────────────────────────────────────────────────────

JSONL_PATH   = "data/weather/100_test_data.jsonl"   # swap to full dataset when ready
LOG_DIR      = "results_log"
CONCURRENCY  = None    # None → resolved from MODEL_DEFAULTS in runner.py
SEED         = 42

MODELS = [
    "llama3:8b-instruct-q2_K",
    # "qwen2.5:7b-instruct-q2_K",
    # "mistral:7b-instruct-v0.2-q2_K",
]

MODES = [
    "zeroshot",
    "fewshot",
]

FORMAT_DBASE = "sentence"   # "sentence" or "jsonl"
TASK         = "weather"

# Metrics to include in the summary table
REPORT_METRICS = [
    "l1loss",               # penalised MAE (missing ts → 0.0)
    "l1loss_matched",       # matched MAE (only on aligned timestamps)
    "timestamp_coverage",   # % of GT timestamps present in prediction
    "coverage",             # % of GT numeric fields populated
    "skill_l1loss",         # skill score vs persistence (penalised)
    "skill_l1loss_matched", # skill score vs persistence (matched)
    "bertscore",
    "cosinesim",
    "bleu",
    "rouge_l",
]

# Path where few-shot prompt file will be written
FEWSHOT_PROMPT_PATH = os.path.join(
    "initial_prompts", "weather", "fewshot-weather.txt"
)

# Zero-shot prompt path — your existing file
ZEROSHOT_PROMPT_PATH = os.path.join(
    "initial_prompts", "weather", "zeroshot-weather.txt"
)

# LSTM baseline configuration
# Set LSTM_CHECKPOINT to None to skip LSTM evaluation
LSTM_CHECKPOINT  = "results_log/lstm_best_h10_seq6_hidden128_layers2.pt"
LSTM_TRAIN_JSONL = "data/weather/train.jsonl"   # same train data used to fit scalers
LSTM_SEQ_LEN     = 6
LSTM_H_MAX       = 10


# ─────────────────────────────────────────────────────────────────────────────
# Prompt file writers
# ─────────────────────────────────────────────────────────────────────────────

def _write_zeroshot_prompt():
    """
    Write a minimal zero-shot instruction file if it does not already exist.
    If you already have a zeroshot-weather.txt, this is skipped.
    """
    if os.path.exists(ZEROSHOT_PROMPT_PATH):
        return   # keep your existing file

    os.makedirs(os.path.dirname(ZEROSHOT_PROMPT_PATH), exist_ok=True)
    content = (
        "You are a weather forecasting assistant.\n"
        "Given the weather observations below, forecast the requested "
        "future hours.\n"
        "Report all variables for each forecast hour in the same format "
        "as the observations.\n"
        "Match the timestamp format exactly.\n"
    )
    with open(ZEROSHOT_PROMPT_PATH, "w", encoding="utf-8") as f:
        f.write(content)


def _write_fewshot_prompt(few_shot_examples: list):
    """
    Format and write the few-shot prompt file.
    Called once before running all few-shot experiments.
    """
    os.makedirs(os.path.dirname(FEWSHOT_PROMPT_PATH), exist_ok=True)
    prompt_text = format_weather_fewshot_prompt(few_shot_examples)
    with open(FEWSHOT_PROMPT_PATH, "w", encoding="utf-8") as f:
        f.write(prompt_text)
    return FEWSHOT_PROMPT_PATH


# ─────────────────────────────────────────────────────────────────────────────
# Single experiment runner
# ─────────────────────────────────────────────────────────────────────────────

def run_weather_experiment(
    data_list: list,
    model: str,
    mode: str,
    log,
    persistence_scores: dict = None,
) -> dict:
    """
    Run one WeatherExperiment (one model, one mode) and return scores.

    For zero-shot: data_list = full dataset
    For few-shot:  data_list = test set only (few-shot examples already removed)
    """
    # Map mode → prompting_mode string (matches your initial_prompts filenames)
    prompting_mode = mode   # "zeroshot" or "fewshot"

    log.info(f"\n  Running: model={model}, mode={mode}, "
             f"n_samples={len(data_list)}")

    exp = WeatherExperiment(
        data_list=data_list,
        log_prefix=log,
        prompting_mode=prompting_mode,
        model=model,
        task=TASK,
        format_dbase=FORMAT_DBASE,
        concurrency=CONCURRENCY,   # None → uses MODEL_DEFAULTS
        timeout=None,              # None → uses MODEL_DEFAULTS
        persistence_scores=persistence_scores,
    )
    exp.run(logging=True)

    # Warn if samples were skipped
    n_expected  = len(data_list)
    n_processed = len(exp.results)
    if n_processed < n_expected:
        log.info(f"  [WARNING] {n_processed}/{n_expected} samples processed "
                 f"— check for timeouts")

    return exp.metrics.get(subject=TASK)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    log = get_logging(
        name="run_all_weather",
        log_dir=LOG_DIR,
        log_prefix=f"run_all_weather_{timestamp}",
    )

    log.info("="*60)
    log.info("Short-term Weather Forecasting Experiment")
    log.info("="*60)
    log.info(f"  Dataset:  {JSONL_PATH}")
    log.info(f"  Models:   {MODELS}")
    log.info(f"  Modes:    {MODES}")
    log.info(f"  Format:   {FORMAT_DBASE}")

    # ── Load full dataset ─────────────────────────────────────────────────────
    all_data = load_jsonl(JSONL_PATH)
    log.info(f"  Loaded {len(all_data)} samples")

    # ── Prepare splits ────────────────────────────────────────────────────────
    # Few-shot split: done ONCE, shared across all models
    # This ensures all models are evaluated on the same test set
    few_shot_examples, test_data = fixed_fewshot_weather_split(all_data)
    verify_split(few_shot_examples, test_data)

    # Write prompt files
    _write_zeroshot_prompt()
    fewshot_path = _write_fewshot_prompt(few_shot_examples)
    log.info(f"\n  Few-shot prompt written to: {fewshot_path}")
    log.info(f"  Few-shot examples:  {len(few_shot_examples)} "
             f"(one per horizon)")
    log.info(f"  Zero-shot test set: {len(all_data)} samples (full dataset)")
    log.info(f"  Few-shot test set:  {len(test_data)} samples")

    # ── Run persistence baseline ONCE ────────────────────────────────────────
    # Persistence uses the full dataset (same as zero-shot — no contamination
    # risk since it makes no LLM calls and uses no examples)
    log.info("\n" + "─"*60)
    log.info("Running persistence baseline...")
    persistence_exp    = PersistenceExperiment(test_data, log)
    persistence_scores = persistence_exp.run(logging=True)
    log.info("Persistence baseline complete.")

    all_results = []

    # ── Outer loop: models ────────────────────────────────────────────────────
    for model in MODELS:
        log.info(f"\n{'='*60}")
        log.info(f"MODEL: {model}")
        log.info(f"{'='*60}")

        # ── Inner loop: modes ─────────────────────────────────────────────────
        for mode in MODES:
            log.info(f"\n  Mode: {mode}")
            log.info(f"  {'─'*40}")

            try:
                # Zero-shot uses full dataset
                # Few-shot uses test_data (few-shot examples removed)
                data_for_mode = test_data

                scores = run_weather_experiment(
                    data_list=data_for_mode,
                    model=model,
                    mode=mode,
                    log=log,
                    persistence_scores=persistence_scores,
                )

                all_results.append({
                    "model":  model,
                    "mode":   mode,
                    "scores": scores,
                })

                log.info(f"\n  [{model} | {mode}] Key Results:")
                for metric in ["l1loss", "coverage", "skill_l1loss"]:
                    val = scores.get(metric)
                    if val is not None:
                        log.info(f"    {metric:20s}: {val:.4f}")

            except Exception as e:
                log.info(f"  [ERROR] {model} | {mode}: {e}")
                all_results.append({
                    "model":  model,
                    "mode":   mode,
                    "scores": {},
                    "error":  str(e),
                })

    # ── Add persistence to results table ─────────────────────────────────────
    all_results.insert(0, {
        "model":  "persistence",
        "mode":   "baseline",
        "scores": persistence_scores,
    })

    # ── Print and save summary ────────────────────────────────────────────────
    _print_summary_table(all_results, log)

    csv_path = os.path.join(
        LOG_DIR, f"summary_weather_{timestamp}.csv"
    )
    _save_summary_csv(all_results, csv_path)
    log.info(f"\nSummary saved to: {csv_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Summary helpers
# ─────────────────────────────────────────────────────────────────────────────

def _print_summary_table(all_results: list, log):
    log.info(f"\n\n{'='*80}")
    log.info("FINAL SUMMARY TABLE — Short-term Weather Forecasting")
    log.info(f"{'='*80}")

    header = (f"{'Model':<35} {'Mode':<12} " +
              " ".join(f"{m[:10]:>12}" for m in REPORT_METRICS))
    log.info(header)
    log.info("─" * len(header))

    for entry in all_results:
        model  = entry["model"]
        mode   = entry["mode"]
        scores = entry.get("scores", {})

        if "error" in entry:
            row = f"{model:<35} {mode:<12}  ERROR: {entry['error']}"
        else:
            vals = []
            for metric in REPORT_METRICS:
                val = scores.get(metric)
                cell = f"{val:.4f}" if isinstance(val, float) else "N/A"
                vals.append(f"{cell:>12}")
            row = f"{model:<35} {mode:<12} " + " ".join(vals)

        log.info(row)

    log.info(f"{'='*80}")


def _save_summary_csv(all_results: list, csv_path: str):
    fieldnames = ["model", "mode"] + REPORT_METRICS

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for entry in all_results:
            if "error" in entry:
                continue
            scores = entry.get("scores", {})
            row    = {"model": entry["model"], "mode": entry["mode"]}
            for metric in REPORT_METRICS:
                val        = scores.get(metric)
                row[metric] = round(val, 4) if isinstance(val, float) else ""
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