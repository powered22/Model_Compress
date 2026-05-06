    # run_all_weather_experiments.py
"""
Entry point for short-term weather forecasting experiments.

Experiment matrix:
  Models  : Llama / Mistral / Qwen at multiple quantization levels
            (q2_K | q4_K_M | q8_0 | f16)
  Modes   : zeroshot | fewshot | ragfs
  Baselines: Persistence, LSTM

RAG-FS mode (ragfs):
  Embeds train.jsonl once with sentence-transformers, then for each test
  datum retrieves the N_SHOTS_RAG most similar training examples to build
  a dynamic few-shot prompt.  This is compared against static fewshot and
  zeroshot across all quantization levels.

Run from project root:
    python -m experiments.run_all_weather_exp
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
    format_ragfs_weather_prompt,
    verify_split,
    load_jsonl,
)
import torch
from experiments.baselines.lstm_evaluator import LSTMEvaluator


# ─────────────────────────────────────────────────────────────────────────────
# Configuration — edit these before running
# ─────────────────────────────────────────────────────────────────────────────

JSONL_PATH   = os.environ.get("WEATHER_DATA",  "data/weather/100_test_data.jsonl")
TRAIN_JSONL  = os.environ.get("WEATHER_TRAIN", "data/weather/train.jsonl")
LOG_DIR      = os.environ.get("LOG_DIR",        "results_log")
CONCURRENCY  = None    # None → resolved from MODEL_DEFAULTS in runner.py
SEED         = int(os.environ.get("SEED", "42"))
MAX_SAMPLES  = int(os.environ.get("MAX_SAMPLES", "100"))

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
# fewshot  : fixed static examples (one per horizon, selected once)
# ragfs    : dynamic retrieval from train.jsonl per test datum (Opsi 2)
MODES = os.environ.get("MODES", "zeroshot,fewshot,ragfs").split(",")

FORMAT_DBASE = "sentence"   # "sentence" or "jsonl"
TASK         = "weather"

# RAG-FS configuration
N_SHOTS_RAG  = 4    # number of examples to retrieve per test datum
RAG_MODEL    = "all-MiniLM-L6-v2"   # sentence-transformers model name

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

# Zero-shot prompt path
ZEROSHOT_PROMPT_PATH = os.path.join(
    "initial_prompts", "weather", "zeroshot-weather.txt"
)

# LSTM baseline configuration
LSTM_CHECKPOINT  = "results_log/lstm_best_h10_seq6_hidden128_layers2.pt"
LSTM_TRAIN_JSONL = "data/weather/train.jsonl"
LSTM_SEQ_LEN     = 6
LSTM_H_MAX       = 10


# ─────────────────────────────────────────────────────────────────────────────
# Prompt file writers
# ─────────────────────────────────────────────────────────────────────────────

def _write_zeroshot_prompt():
    if os.path.exists(ZEROSHOT_PROMPT_PATH):
        return
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
    rag_retriever=None,
) -> dict:
    """
    Run one WeatherExperiment (one model × one mode) and return scores.

    For ragfs, rag_retriever must be pre-fitted before calling this function.
    """
    log.info(f"\n  Running: model={model}, mode={mode}, "
             f"n_samples={len(data_list)}")

    exp = WeatherExperiment(
        data_list=data_list,
        log_prefix=log,
        prompting_mode=mode,
        model=model,
        task=TASK,
        format_dbase=FORMAT_DBASE,
        concurrency=CONCURRENCY,
        timeout=None,
        persistence_scores=persistence_scores,
        rag_retriever=rag_retriever,   # None for zeroshot/fewshot
        n_shots=N_SHOTS_RAG,
    )
    exp.run(logging=True)

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
    log.info(f"  Dataset:      {JSONL_PATH}")
    log.info(f"  Train set:    {TRAIN_JSONL}")
    log.info(f"  Models:       {MODELS}")
    log.info(f"  Modes:        {MODES}")
    log.info(f"  Format:       {FORMAT_DBASE}")
    log.info(f"  N_SHOTS_RAG:  {N_SHOTS_RAG}")

    # ── Load datasets ─────────────────────────────────────────────────────────
    all_data   = load_jsonl(JSONL_PATH)[:MAX_SAMPLES]
    train_data = load_jsonl(TRAIN_JSONL) if os.path.exists(TRAIN_JSONL) else []
    log.info(f"  Loaded {len(all_data)} test samples, {len(train_data)} train samples")

    # ── Prepare static few-shot split ─────────────────────────────────────────
    few_shot_examples, test_data = fixed_fewshot_weather_split(all_data)
    verify_split(few_shot_examples, test_data)

    _write_zeroshot_prompt()
    fewshot_path = _write_fewshot_prompt(few_shot_examples)
    log.info(f"\n  Few-shot prompt written to: {fewshot_path}")
    log.info(f"  Few-shot examples:  {len(few_shot_examples)}")
    log.info(f"  Static test set:    {len(test_data)} samples")

    # ── Build RAG retriever once (shared across all models) ───────────────────
    rag_retriever = None
    if "ragfs" in MODES:
        if not train_data:
            log.info(f"\n  [WARNING] RAG-FS requested but {TRAIN_JSONL} not found. "
                     f"Falling back to full test set as pool.")
            rag_pool = all_data
        else:
            rag_pool = train_data

        log.info(f"\n  Building RAG retriever on {len(rag_pool)} examples ...")
        from experiments.rag_retriever import RAGRetriever
        rag_retriever = RAGRetriever(model_name=RAG_MODEL)
        rag_retriever.fit(rag_pool, text_key="observation")
        log.info(f"  RAG retriever ready (pool_size={rag_retriever.pool_size})")

    # ── Persistence baseline ──────────────────────────────────────────────────
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
        log.info(f"  Quant level: {_get_quant_level(model)}")
        log.info(f"{'='*60}")

        for mode in MODES:
            log.info(f"\n  Mode: {mode}")
            log.info(f"  {'─'*40}")

            try:
                # ragfs uses the full test_data with dynamic retrieval
                # zeroshot uses full test_data (no prompt examples to exclude)
                # fewshot uses test_data (few-shot examples excluded)
                data_for_mode = test_data
                retriever_for_mode = rag_retriever if mode == "ragfs" else None

                scores = run_weather_experiment(
                    data_list=data_for_mode,
                    model=model,
                    mode=mode,
                    log=log,
                    persistence_scores=persistence_scores,
                    rag_retriever=retriever_for_mode,
                )

                all_results.append({
                    "model":  model,
                    "quant":  _get_quant_level(model),
                    "family": _get_model_family(model),
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

    # ── Add persistence to results table ──────────────────────────────────────
    all_results.insert(0, {
        "model":  "persistence",
        "quant":  "baseline",
        "family": "baseline",
        "mode":   "baseline",
        "scores": persistence_scores,
    })

    _print_summary_table(all_results, log)

    csv_path = os.path.join(LOG_DIR, f"summary_weather_{timestamp}.csv")
    _save_summary_csv(all_results, csv_path)
    log.info(f"\nSummary saved to: {csv_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Model name helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_quant_level(model: str) -> str:
    """Extract quantization suffix from model name (e.g. 'q4_K_M', 'f16')."""
    for quant in ("f16", "q8_0", "q6_K", "q5_K_M", "q4_K_M", "q4_K_S",
                  "q3_K_M", "q2_K"):
        if quant.lower() in model.lower():
            return quant
    return "unknown"


def _get_model_family(model: str) -> str:
    """Extract model family from Ollama tag (e.g. 'llama3', 'mistral', 'qwen2.5')."""
    name = model.split(":")[0]
    return name


# ─────────────────────────────────────────────────────────────────────────────
# Summary helpers
# ─────────────────────────────────────────────────────────────────────────────

def _print_summary_table(all_results: list, log):
    log.info(f"\n\n{'='*90}")
    log.info("FINAL SUMMARY TABLE — Short-term Weather Forecasting")
    log.info(f"{'='*90}")

    header = (f"{'Model':<32} {'Quant':<10} {'Mode':<10} " +
              " ".join(f"{m[:10]:>12}" for m in REPORT_METRICS))
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
                val = scores.get(metric)
                cell = f"{val:.4f}" if isinstance(val, float) else "N/A"
                vals.append(f"{cell:>12}")
            row = f"{model:<32} {quant:<10} {mode:<10} " + " ".join(vals)

        log.info(row)

    log.info(f"{'='*90}")


def _save_summary_csv(all_results: list, csv_path: str):
    fieldnames = ["model", "family", "quant", "mode"] + REPORT_METRICS

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
