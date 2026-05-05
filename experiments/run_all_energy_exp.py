# run_all_energy_experiments.py
"""
Entry point for energy forecasting and extreme energy experiments.
Mirrors run_all_weather_experiments.py and run_all_extreme_experiments.py
so all three entry points produce the same summary CSV format.

Run from project root:
    python -m run_all_energy_experiments
"""

import os
import csv
from datetime import datetime
import torch

from experiments.utils import get_logging
from experiments.energy_utils import (
    load_energy_data,
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
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

# Dataset paths — update when you have more data
FORECAST_DATA_PATH = "data/energy_task/sample_energy_forecast.json"
EXTREME_DATA_PATH  = "data/energy_extreme/sample_energy_extreme.json"

LOG_DIR  = "results_log"
SEED     = 42

MODELS = [
    "llama3:8b-instruct-q2_K",
    # "qwen2.5:7b-instruct-q2_K",
    # "mistral:7b-instruct-v0.2-q2_K",
]

MODES = ["zeroshot", "fewshot"]

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
    log.info(f"  Forecast data:  {FORECAST_DATA_PATH}")
    log.info(f"  Extreme data:   {EXTREME_DATA_PATH}")
    log.info(f"  Models:         {MODELS}")
    log.info(f"  Modes:          {MODES}")

    forecast_results = []
    extreme_results  = []

    # ─────────────────────────────────────────────────────────────────────────
    # TASK 1: ENERGY FORECAST
    # ─────────────────────────────────────────────────────────────────────────

    log.info("\n" + "=" * 60)
    log.info("TASK 1: Energy Load Forecasting")
    log.info("=" * 60)

    forecast_data = load_energy_data(FORECAST_DATA_PATH)
    log.info(f"  Loaded {len(forecast_data)} samples")

    # Split — done once, shared across all models
    fewshot_forecast, test_forecast = fixed_fewshot_energy_split(forecast_data)
    fewshot_forecast_prompt = format_energy_fewshot_prompt(fewshot_forecast)

    # Zeroshot prompt — minimal instruction
    zeroshot_forecast_prompt = (
        "You are an electricity load forecasting assistant.\n"
        "Output exactly 48 comma-separated numbers. No explanation.\n"
    )

    # Persistence baseline — runs once on full dataset
    log.info("\n  Running persistence baseline...")
    pers_exp    = EnergyPersistenceBaseline(forecast_data, log)
    pers_scores = pers_exp.run(logging=True)

    forecast_results.append({
        "model": "persistence", "mode": "baseline", "scores": pers_scores,
    })

    # LLM experiments
    for model in MODELS:
        log.info(f"\n  Model: {model}")
        for mode in MODES:
            log.info(f"    Mode: {mode}")
            data_for_mode  = forecast_data if mode == "zeroshot" else test_forecast
            prompt_for_mode = (zeroshot_forecast_prompt if mode == "zeroshot"
                               else fewshot_forecast_prompt)
            try:
                exp = EnergyForecastExperiment(
                    data=data_for_mode,
                    log_prefix=log,
                    model=model,
                    prompting_mode=mode,
                    fewshot_prompt=prompt_for_mode,
                    persistence_mae=pers_scores.get("mae_penalised"),
                )
                scores = exp.run(logging=True)
                forecast_results.append({"model": model, "mode": mode, "scores": scores})

            except Exception as e:
                log.info(f"    [ERROR] {model} | {mode}: {e}")
                forecast_results.append({
                    "model": model, "mode": mode, "scores": {}, "error": str(e)})

    # ─────────────────────────────────────────────────────────────────────────
    # TASK 2: EXTREME ENERGY
    # ─────────────────────────────────────────────────────────────────────────

    log.info("\n" + "=" * 60)
    log.info("TASK 2: Extreme Energy Classification")
    log.info("=" * 60)

    extreme_data = load_energy_data(EXTREME_DATA_PATH)
    log.info(f"  Loaded {len(extreme_data)} samples")

    # Check class distribution
    n_pos = sum(1 for d in extreme_data if get_label_extreme_bool(d))
    n_neg = len(extreme_data) - n_pos
    log.info(f"  Class distribution: True={n_pos} ({n_pos/len(extreme_data):.1%}), "
             f"False={n_neg} ({n_neg/len(extreme_data):.1%})")

    # Split
    fewshot_extreme, test_extreme = fixed_fewshot_energy_extreme_split(extreme_data)
    fewshot_extreme_prompt = format_energy_extreme_fewshot_prompt(fewshot_extreme)

    zeroshot_extreme_prompt = (
        "You are an electricity grid analyst.\n"
        "Answer YES or NO. If YES, also provide peak load estimate in MW.\n"
    )

    # Threshold rule baseline — runs once on full dataset
    log.info("\n  Running threshold rule baseline...")
    thresh_exp    = EnergyThresholdBaseline(extreme_data, log)
    thresh_scores = thresh_exp.run(logging=True)

    extreme_results.append({
        "model": "threshold_rule", "mode": "baseline", "scores": thresh_scores,
    })

    # LLM experiments
    for model in MODELS:
        log.info(f"\n  Model: {model}")
        for mode in MODES:
            log.info(f"    Mode: {mode}")
            data_for_mode   = extreme_data if mode == "zeroshot" else test_extreme
            prompt_for_mode = (zeroshot_extreme_prompt if mode == "zeroshot"
                               else fewshot_extreme_prompt)
            try:
                exp = EnergyExtremeExperiment(
                    data=data_for_mode,
                    log_prefix=log,
                    model=model,
                    prompting_mode=mode,
                    fewshot_prompt=prompt_for_mode,
                )
                scores = exp.run(logging=True)
                extreme_results.append({"model": model, "mode": mode, "scores": scores})

            except Exception as e:
                log.info(f"    [ERROR] {model} | {mode}: {e}")
                extreme_results.append({
                    "model": model, "mode": mode, "scores": {}, "error": str(e)})

    # ─────────────────────────────────────────────────────────────────────────
    # Summary tables
    # ─────────────────────────────────────────────────────────────────────────

    _print_table(forecast_results, FORECAST_METRICS,
                 "Energy Load Forecasting", log)
    _print_table(extreme_results, EXTREME_METRICS,
                 "Extreme Energy Classification", log)

    _save_csv(forecast_results, FORECAST_METRICS,
              os.path.join(LOG_DIR, f"summary_energy_forecast_{ts}.csv"))
    _save_csv(extreme_results, EXTREME_METRICS,
              os.path.join(LOG_DIR, f"summary_energy_extreme_{ts}.csv"),
              extra_cols=["tp", "fp", "fn", "tn"])

    log.info(f"\nAll results saved to {LOG_DIR}/")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_label_extreme_bool(datum: dict) -> bool:
    from experiments.energy_utils import get_label_extreme
    has, _ = get_label_extreme(datum)
    return has


def _print_table(results: list, metrics: list, title: str, log):
    log.info(f"\n{'='*70}")
    log.info(f"SUMMARY: {title}")
    log.info(f"{'='*70}")
    header = f"{'Model':<35} {'Mode':<12} " + " ".join(
        f"{m[:12]:>13}" for m in metrics)
    log.info(header)
    log.info("─" * len(header))

    for entry in results:
        if "error" in entry:
            log.info(f"{entry['model']:<35} {entry['mode']:<12}  ERROR: {entry['error']}")
            continue
        scores = entry.get("scores", {})
        vals   = []
        for m in metrics:
            v = scores.get(m)
            if v is None:
                vals.append(f"{'N/A':>13}")
            elif isinstance(v, float):
                vals.append(f"{v:>13.4f}")
            else:
                vals.append(f"{str(v):>13}")
        log.info(f"{entry['model']:<35} {entry['mode']:<12} " + " ".join(vals))

    log.info(f"{'='*70}")


def _save_csv(results: list, metrics: list, path: str, extra_cols: list = None):
    fieldnames = ["model", "mode"] + metrics + (extra_cols or [])
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for entry in results:
            if "error" in entry:
                continue
            scores = entry.get("scores", {})
            row = {"model": entry["model"], "mode": entry["mode"]}
            for m in metrics + (extra_cols or []):
                v = scores.get(m)
                row[m] = round(v, 4) if isinstance(v, float) else (v or "")
            writer.writerow(row)
    print(f"Saved: {path}")


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