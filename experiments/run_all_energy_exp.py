# run_all_energy_experiments.py
"""
Entry point for energy load forecasting and extreme energy classification.

Experiment matrix:
  Models  : Llama / Mistral / Qwen at multiple quantization levels
            (q2_K | q4_K_M | q8_0 | f16)
  Modes   : zeroshot | fewshot | ragfs
  Tasks   : energy_task (48-step load forecast) + energy_extreme (binary + peak MW)

RAG-FS mode (ragfs):
  Embeds train data once with sentence-transformers, then for each test datum
  retrieves the N_SHOTS_RAG most similar training examples to build a dynamic
  few-shot prompt.  The text used for similarity is the pre-built "prompt" field
  which already encodes history, region, season, and threshold information.

Run from project root:
    python -m experiments.run_all_energy_exp
"""

import os
import csv
from datetime import datetime
import torch

from experiments.utils import get_logging
from experiments.energy_utils import (
    load_energy_data,
    get_label_extreme,
    fixed_fewshot_energy_split,
    fixed_fewshot_energy_extreme_split,
    format_energy_fewshot_prompt,
    format_energy_extreme_fewshot_prompt,
)
from experiments.energy_runner import (
    EnergyPersistenceBaseline,
    EnergyThresholdBaseline,
    EnergyForecastExperiment,
    EnergyExtremeExperiment,
)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration — edit these before running
# ─────────────────────────────────────────────────────────────────────────────

# Dataset paths (override via env var)
FORECAST_DATA_PATH  = os.environ.get("ENERGY_FORECAST_DATA",  "data/energy_task/energy_forecast_new.json")
EXTREME_DATA_PATH   = os.environ.get("ENERGY_EXTREME_DATA",   "data/energy_extreme/energy_extreme_new.json")

# Training data for RAG-FS retrieval
TRAIN_FORECAST_PATH = os.environ.get("ENERGY_FORECAST_TRAIN", "data/energy_task/train.json")
TRAIN_EXTREME_PATH  = os.environ.get("ENERGY_EXTREME_TRAIN",  "data/energy_extreme/train.json")

LOG_DIR = os.environ.get("LOG_DIR", "results_log")
SEED    = int(os.environ.get("SEED", "42"))

# Limit number of test samples (keeps experiments tractable with large JSON files)
MAX_SAMPLES = int(os.environ.get("MAX_SAMPLES", "100"))

# RAG-FS configuration
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
# fewshot  : static examples selected once (by season for forecast,
#            by class for extreme)
# ragfs    : per-datum retrieval from train data (Opsi 2)
MODES = os.environ.get("MODES", "zeroshot,fewshot,ragfs").split(",")

# Metrics included in summary CSV
FORECAST_METRICS = [
    "mae_penalised",
    "mae_matched",
    "step_coverage",
    "extraction_rate",
    "skill_penalised",
    "skill_matched",
]

EXTREME_METRICS = [
    "accuracy", "precision", "recall", "f1",
    "peak_mae",
    "lenient", "consistency",
    "extraction_rate",
]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    log = get_logging(
        name="run_all_energy",
        log_dir=LOG_DIR,
        log_prefix=f"run_all_energy_{ts}",
    )

    log.info("=" * 60)
    log.info("Energy Forecasting + Extreme Energy Experiment")
    log.info("=" * 60)
    log.info(f"  Forecast data:        {FORECAST_DATA_PATH}")
    log.info(f"  Extreme data:         {EXTREME_DATA_PATH}")
    log.info(f"  Train forecast data:  {TRAIN_FORECAST_PATH}")
    log.info(f"  Train extreme data:   {TRAIN_EXTREME_PATH}")
    log.info(f"  Models:               {MODELS}")
    log.info(f"  Modes:                {MODES}")
    log.info(f"  N_SHOTS_RAG:          {N_SHOTS_RAG}")

    forecast_results = []
    extreme_results  = []

    # ── Load datasets ─────────────────────────────────────────────────────────
    forecast_data = load_energy_data(FORECAST_DATA_PATH)[:MAX_SAMPLES]
    extreme_data  = load_energy_data(EXTREME_DATA_PATH)[:MAX_SAMPLES]
    log.info(f"  Loaded {len(forecast_data)} forecast samples, "
             f"{len(extreme_data)} extreme samples "
             f"(capped at MAX_SAMPLES={MAX_SAMPLES})")

    # Load train data for RAG-FS (fall back to test set if not found)
    train_forecast = _load_or_fallback(TRAIN_FORECAST_PATH, forecast_data, log,
                                       "train forecast")
    train_extreme  = _load_or_fallback(TRAIN_EXTREME_PATH,  extreme_data,  log,
                                       "train extreme")

    # ── Build RAG retrievers once (shared across all models) ──────────────────
    rag_forecast_retriever = None
    rag_extreme_retriever  = None

    if "ragfs" in MODES:
        from experiments.rag_retriever import RAGRetriever

        log.info(f"\n  Building RAG retriever for energy forecast "
                 f"({len(train_forecast)} examples) ...")
        rag_forecast_retriever = RAGRetriever(model_name=RAG_MODEL)
        rag_forecast_retriever.fit(train_forecast, text_key="prompt")

        log.info(f"  Building RAG retriever for energy extreme "
                 f"({len(train_extreme)} examples) ...")
        rag_extreme_retriever = RAGRetriever(model_name=RAG_MODEL)
        rag_extreme_retriever.fit(train_extreme, text_key="prompt")

        log.info("  RAG retrievers ready.")

    # ── Class distribution for extreme task ───────────────────────────────────
    n_pos = sum(1 for d in extreme_data if get_label_extreme(d)[0])
    n_neg = len(extreme_data) - n_pos
    log.info(f"\n  Extreme class distribution: "
             f"True={n_pos} ({n_pos/len(extreme_data):.1%}), "
             f"False={n_neg} ({n_neg/len(extreme_data):.1%})")

    # ── Static few-shot splits (done once, shared across models) ──────────────
    fewshot_forecast, test_forecast = fixed_fewshot_energy_split(forecast_data)
    fewshot_extreme,  test_extreme  = fixed_fewshot_energy_extreme_split(extreme_data)

    fewshot_forecast_prompt = format_energy_fewshot_prompt(fewshot_forecast)
    fewshot_extreme_prompt  = format_energy_extreme_fewshot_prompt(fewshot_extreme)

    zeroshot_forecast_prompt = (
        "You are an electricity load forecasting assistant.\n"
        "Output exactly 48 comma-separated numbers. No explanation.\n"
    )
    zeroshot_extreme_prompt = (
        "You are an electricity grid analyst.\n"
        "Answer YES or NO. If YES, also provide peak load estimate in MW.\n"
    )

    # ─────────────────────────────────────────────────────────────────────────
    # TASK 1: ENERGY LOAD FORECASTING
    # ─────────────────────────────────────────────────────────────────────────

    log.info("\n" + "=" * 60)
    log.info("TASK 1: Energy Load Forecasting")
    log.info("=" * 60)

    log.info("\n  Running persistence baseline...")
    pers_exp    = EnergyPersistenceBaseline(forecast_data, log)
    pers_scores = pers_exp.run(logging=True)
    forecast_results.append({
        "model": "persistence", "quant": "baseline",
        "family": "baseline",  "mode": "baseline",
        "scores": pers_scores,
    })

    for model in MODELS:
        log.info(f"\n  {'='*50}")
        log.info(f"  MODEL: {model}  [quant={_get_quant_level(model)}]")
        log.info(f"  {'='*50}")

        for mode in MODES:
            log.info(f"\n    Mode: {mode}")
            try:
                data_for_mode   = forecast_data if mode == "zeroshot" else test_forecast
                retriever       = rag_forecast_retriever if mode == "ragfs" else None
                static_prompt   = (zeroshot_forecast_prompt if mode == "zeroshot"
                                   else fewshot_forecast_prompt)

                exp = EnergyForecastExperiment(
                    data=data_for_mode,
                    log_prefix=log,
                    model=model,
                    prompting_mode=mode,
                    fewshot_prompt=static_prompt,
                    persistence_mae=pers_scores.get("mae_penalised"),
                    rag_retriever=retriever,
                    n_shots=N_SHOTS_RAG,
                )
                scores = exp.run(logging=True)
                forecast_results.append({
                    "model":  model,
                    "quant":  _get_quant_level(model),
                    "family": _get_model_family(model),
                    "mode":   mode,
                    "scores": scores,
                })

            except Exception as e:
                log.info(f"    [ERROR] {model} | {mode}: {e}")
                import traceback
                log.info(traceback.format_exc())
                forecast_results.append({
                    "model":  model,
                    "quant":  _get_quant_level(model),
                    "family": _get_model_family(model),
                    "mode":   mode,
                    "scores": {}, "error": str(e),
                })

    # ─────────────────────────────────────────────────────────────────────────
    # TASK 2: EXTREME ENERGY CLASSIFICATION
    # ─────────────────────────────────────────────────────────────────────────

    log.info("\n" + "=" * 60)
    log.info("TASK 2: Extreme Energy Classification")
    log.info("=" * 60)

    log.info("\n  Running threshold rule baseline...")
    thresh_exp    = EnergyThresholdBaseline(extreme_data, log)
    thresh_scores = thresh_exp.run(logging=True)
    extreme_results.append({
        "model": "threshold_rule", "quant": "baseline",
        "family": "baseline",     "mode": "baseline",
        "scores": thresh_scores,
    })

    for model in MODELS:
        log.info(f"\n  {'='*50}")
        log.info(f"  MODEL: {model}  [quant={_get_quant_level(model)}]")
        log.info(f"  {'='*50}")

        for mode in MODES:
            log.info(f"\n    Mode: {mode}")
            try:
                data_for_mode = extreme_data if mode == "zeroshot" else test_extreme
                retriever     = rag_extreme_retriever if mode == "ragfs" else None
                static_prompt = (zeroshot_extreme_prompt if mode == "zeroshot"
                                 else fewshot_extreme_prompt)

                exp = EnergyExtremeExperiment(
                    data=data_for_mode,
                    log_prefix=log,
                    model=model,
                    prompting_mode=mode,
                    fewshot_prompt=static_prompt,
                    rag_retriever=retriever,
                    n_shots=N_SHOTS_RAG,
                )
                scores = exp.run(logging=True)
                extreme_results.append({
                    "model":  model,
                    "quant":  _get_quant_level(model),
                    "family": _get_model_family(model),
                    "mode":   mode,
                    "scores": scores,
                })

            except Exception as e:
                log.info(f"    [ERROR] {model} | {mode}: {e}")
                import traceback
                log.info(traceback.format_exc())
                extreme_results.append({
                    "model":  model,
                    "quant":  _get_quant_level(model),
                    "family": _get_model_family(model),
                    "mode":   mode,
                    "scores": {}, "error": str(e),
                })

    # ── Summary tables ────────────────────────────────────────────────────────
    _print_table(forecast_results, FORECAST_METRICS, "Energy Load Forecasting", log)
    _print_table(extreme_results,  EXTREME_METRICS,  "Extreme Energy Classification", log)

    _save_csv(forecast_results, FORECAST_METRICS,
              os.path.join(LOG_DIR, f"summary_energy_forecast_{ts}.csv"))
    _save_csv(extreme_results,  EXTREME_METRICS,
              os.path.join(LOG_DIR, f"summary_energy_extreme_{ts}.csv"),
              extra_cols=["tp", "fp", "fn", "tn"])

    log.info(f"\nAll results saved to {LOG_DIR}/")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_or_fallback(path: str, fallback: list, log, label: str) -> list:
    if os.path.exists(path):
        data = load_energy_data(path)
        log.info(f"  Loaded {len(data)} {label} samples from {path}")
        return data
    log.info(f"  [WARNING] {path} not found — using test set as {label} pool for RAG-FS")
    return fallback


def _get_quant_level(model: str) -> str:
    for quant in ("f16", "q8_0", "q6_K", "q5_K_M", "q4_K_M", "q4_K_S",
                  "q3_K_M", "q2_K"):
        if quant.lower() in model.lower():
            return quant
    return "unknown"


def _get_model_family(model: str) -> str:
    return model.split(":")[0]


def _print_table(results: list, metrics: list, title: str, log):
    log.info(f"\n{'='*90}")
    log.info(f"SUMMARY: {title}")
    log.info(f"{'='*90}")
    header = (f"{'Model':<32} {'Quant':<10} {'Mode':<10} " +
              " ".join(f"{m[:12]:>13}" for m in metrics))
    log.info(header)
    log.info("─" * len(header))

    for entry in results:
        model  = entry["model"]
        quant  = entry.get("quant", "")
        mode   = entry["mode"]
        scores = entry.get("scores", {})

        if "error" in entry:
            log.info(f"{model:<32} {quant:<10} {mode:<10}  ERROR: {entry['error']}")
            continue

        vals = []
        for m in metrics:
            v = scores.get(m)
            if v is None:
                vals.append(f"{'N/A':>13}")
            elif isinstance(v, float):
                vals.append(f"{v:>13.4f}")
            else:
                vals.append(f"{str(v):>13}")
        log.info(f"{model:<32} {quant:<10} {mode:<10} " + " ".join(vals))

    log.info(f"{'='*90}")


def _save_csv(results: list, metrics: list, path: str, extra_cols: list = None):
    fieldnames = ["model", "family", "quant", "mode"] + metrics + (extra_cols or [])
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for entry in results:
            if "error" in entry:
                continue
            scores = entry.get("scores", {})
            row = {
                "model":  entry["model"],
                "family": entry.get("family", ""),
                "quant":  entry.get("quant", ""),
                "mode":   entry["mode"],
            }
            for m in metrics + (extra_cols or []):
                v = scores.get(m)
                row[m] = round(v, 4) if isinstance(v, float) else (v or "")
            writer.writerow(row)
    print(f"Saved: {path}")


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
