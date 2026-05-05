# experiments/energy_runner.py
"""
Experiment runners for energy forecasting and extreme energy tasks.
Mirrors WeatherExperiment / WeatherExtremeExperiment interface so
results feed into the same summary CSV pipeline.
"""
from __future__ import annotations

import asyncio
import csv
import json
import os
import time
from datetime import datetime
from typing import List, Dict, Optional

from tqdm.asyncio import tqdm_asyncio

from experiments.energy_metrics import EnergyForecastMetrics, EnergyExtremeMetrics
from experiments.energy_utils import (
    extract_forecast_numbers,
    extract_extreme_response,
    persistence_forecast,
    threshold_rule_baseline,
    get_label_extreme,
    get_threshold,
    get_season,
    ENERGY_HORIZON,
)
from experiments.prompts import talk_to_llm
from experiments.utils import Logger, get_logging
from experiments.runner import MODEL_DEFAULTS, _DEFAULT_CONCURRENCY, _DEFAULT_TIMEOUT


# ─────────────────────────────────────────────────────────────────────────────
# CSV column definitions (matching weather CSV structure)
# ─────────────────────────────────────────────────────────────────────────────

CSV_ENERGY_FORECAST = [
    "Key", "Region", "Season", "Day",
    "Prompt",
    "LLM Result (raw)",
    "N Predicted Steps",
    "MAE Penalised",        # missing steps → 0.0
    "MAE Matched",          # only on predicted steps
    "Step Coverage %",      # % of 48 steps predicted
    "Extraction OK",
    "Model", "Prompting", "Elapsed (s)",
]

CSV_ENERGY_EXTREME = [
    "Key", "Region", "Season", "Day",
    "Prompt",
    "LLM Result (raw)",
    "Pred Has Extreme",
    "Pred Peak MW",
    "GT Has Extreme",
    "GT Peak MW",
    "GT N Extreme Slots",
    "Lenient",
    "Consistent",
    "Peak MAE",
    "Extraction OK",
    "Model", "Prompting", "Elapsed (s)",
]


# ─────────────────────────────────────────────────────────────────────────────
# Helper: resolve per-model config
# ─────────────────────────────────────────────────────────────────────────────

def _model_config(model: str, concurrency=None, timeout=None) -> dict:
    d = MODEL_DEFAULTS.get(model, {})
    return {
        "concurrency": concurrency or d.get("concurrency", _DEFAULT_CONCURRENCY),
        "timeout":     timeout     or d.get("timeout",     _DEFAULT_TIMEOUT),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Persistence Baseline — Energy Forecast
# ─────────────────────────────────────────────────────────────────────────────

class EnergyPersistenceBaseline:
    """
    Persistence baseline for energy forecast.
    Copies last 48 history values as the 48-step forecast.
    100% step coverage, no extraction step.
    """

    def __init__(self, data: List[dict], log_prefix):
        self.data       = data
        self.log_prefix = log_prefix
        self.metrics    = EnergyForecastMetrics()

    def run(self, logging: bool = True) -> dict:
        logger = None
        if logging:
            ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
            path   = f"results_log/energy_forecast_persistence_{ts}.csv"
            os.makedirs("results_log", exist_ok=True)
            logger = csv.writer(open(path, "w", newline="", encoding="utf-8-sig"))
            logger.writerow(CSV_ENERGY_FORECAST)

        for datum in self.data:
            history = datum.get("history", [])
            target  = datum.get("target", [])
            ctx     = datum.get("context", {})
            pred    = persistence_forecast(history, ENERGY_HORIZON)
            scores  = self.metrics.update(pred, target, extraction_ok=True)

            if logger:
                logger.writerow([
                    datum.get("key", ""),
                    ctx.get("region", ""),
                    ctx.get("season", ""),
                    ctx.get("day_of_week", ""),
                    datum.get("prompt", "")[:200],
                    ",".join(f"{v:.2f}" for v in pred),
                    scores["n_pred_steps"],
                    f"{scores['mae_penalised']:.4f}",
                    f"{scores['mae_matched']:.4f}" if scores["mae_matched"] else "N/A",
                    f"{scores['step_coverage']:.1f}",
                    True,
                    "persistence", "persistence", 0.0,
                ])

        final = self.metrics.get()
        self.log_prefix.info("\n=== Energy Persistence Baseline ===")
        for k, v in final.items():
            self.log_prefix.info(f"  {k:25s}: {v}")
        return final


# ─────────────────────────────────────────────────────────────────────────────
# Threshold Rule Baseline — Energy Extreme
# ─────────────────────────────────────────────────────────────────────────────

class EnergyThresholdBaseline:
    """
    Rule-based extreme baseline: if recent max load > 95% of threshold → Yes.
    Uses only information available in the prompt (history + threshold).
    """

    def __init__(self, data: List[dict], log_prefix,
                 trigger_fraction: float = 0.95):
        self.data             = data
        self.log_prefix       = log_prefix
        self.trigger_fraction = trigger_fraction
        self.metrics          = EnergyExtremeMetrics()

    def run(self, logging: bool = True) -> dict:
        logger = None
        if logging:
            ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = f"results_log/energy_extreme_threshold_{ts}.csv"
            os.makedirs("results_log", exist_ok=True)
            logger = csv.writer(open(path, "w", newline="", encoding="utf-8-sig"))
            logger.writerow(CSV_ENERGY_EXTREME)

        for datum in self.data:
            history   = datum.get("history", [])
            threshold = get_threshold(datum) or 999999.0
            gt_has, gt_peak = get_label_extreme(datum)
            gt_slots  = datum.get("ground_truth", {}).get("n_extreme_slots", 0)
            ctx       = datum.get("context", {})

            pred_has, pred_peak = threshold_rule_baseline(
                history, threshold, trigger_fraction=self.trigger_fraction)

            scores = self.metrics.update(
                pred_has=pred_has, pred_peak_mw=pred_peak,
                gt_has=gt_has,    gt_peak_mw=gt_peak,
            )

            if logger:
                logger.writerow([
                    datum.get("key", ""),
                    ctx.get("region", ""),
                    ctx.get("season", ""),
                    ctx.get("day_of_week", ""),
                    datum.get("prompt", "")[:200],
                    f"RULE: {'YES' if pred_has else 'NO'}",
                    pred_has, pred_peak or 0.0,
                    gt_has, gt_peak, gt_slots,
                    scores["lenient"], scores["consistent"],
                    scores["peak_mae"] or "N/A",
                    True,
                    "threshold_rule", "threshold_rule", 0.0,
                ])

        final = self.metrics.cget()
        self.log_prefix.info("\n=== Energy Threshold Rule Baseline ===")
        for k, v in final.items():
            self.log_prefix.info(f"  {k:25s}: {v}")
        return final


# ─────────────────────────────────────────────────────────────────────────────
# LLM Experiment — Energy Forecast
# ─────────────────────────────────────────────────────────────────────────────

class EnergyForecastExperiment:
    """
    Runs LLM-based energy load forecasting experiment.
    Produces same CSV format and metrics structure as WeatherExperiment.
    """

    def __init__(
        self,
        data:            List[dict],
        log_prefix,
        model:           str,
        prompting_mode:  str,        # "zeroshot" or "fewshot"
        fewshot_prompt:  str = "",   # initial_prompt string
        concurrency:     int = None,
        timeout:         int = None,
        persistence_mae: float = None,
    ):
        self.data            = data
        self.log_prefix      = log_prefix
        self.model           = model
        self.prompting_mode  = prompting_mode
        self.fewshot_prompt  = fewshot_prompt
        self.persistence_mae = persistence_mae
        self.metrics         = EnergyForecastMetrics()
        self.results         = []

        cfg              = _model_config(model, concurrency, timeout)
        self.concurrency = cfg["concurrency"]
        self.timeout     = cfg["timeout"]
        self.semaphore   = asyncio.Semaphore(self.concurrency)

    def run(self, logging: bool = True) -> dict:
        logger = None
        if logging:
            ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = f"results_log/energy_forecast_{self.model}_{self.prompting_mode}_{ts}.csv"
            os.makedirs("results_log", exist_ok=True)
            f      = open(path, "w", newline="", encoding="utf-8-sig")
            logger = csv.writer(f)
            logger.writerow(CSV_ENERGY_FORECAST)

        self.log_prefix.info(
            f"[EnergyForecast] model={self.model} mode={self.prompting_mode} "
            f"n={len(self.data)} concurrency={self.concurrency} timeout={self.timeout}s")

        async def process(idx, datum):
            async with self.semaphore:
                try:
                    prompt = datum.get("prompt", "")
                    target = datum.get("target", [])
                    ctx    = datum.get("context", {})

                    start = time.perf_counter()
                    llm_response = await asyncio.wait_for(
                        talk_to_llm(prompt, fewshot=self.fewshot_prompt,
                                    model=self.model),
                        timeout=self.timeout,
                    )
                    elapsed = time.perf_counter() - start

                    pred, extraction_ok = extract_forecast_numbers(llm_response)
                    scores = self.metrics.update(pred, target, extraction_ok)

                    self.results.append({
                        "index": idx, "key": datum.get("key", ""),
                        "response": llm_response, "pred": pred,
                        "target": target, **scores,
                    })

                    if logger:
                        logger.writerow([
                            datum.get("key", ""),
                            ctx.get("region", ""), ctx.get("season", ""),
                            ctx.get("day_of_week", ""),
                            prompt[:200], llm_response[:300],
                            scores["n_pred_steps"],
                            f"{scores['mae_penalised']:.4f}",
                            f"{scores['mae_matched']:.4f}" if scores["mae_matched"] else "N/A",
                            f"{scores['step_coverage']:.1f}",
                            extraction_ok,
                            self.model, self.prompting_mode, f"{elapsed:.2f}",
                        ])

                except asyncio.TimeoutError:
                    self.log_prefix.info(
                        f"[TIMEOUT] idx={idx} model={self.model} after {self.timeout}s")
                except Exception as e:
                    self.log_prefix.info(f"[ERROR] idx={idx}: {e}")

        async def run_all():
            tasks = [process(i, d) for i, d in enumerate(self.data)]
            await tqdm_asyncio.gather(*tasks, desc="Energy Forecast LLM")

        asyncio.run(run_all())

        # Warn on skipped samples
        if len(self.results) < len(self.data):
            self.log_prefix.info(
                f"[WARNING] {len(self.results)}/{len(self.data)} processed")

        final = self.metrics.get(persistence_mae=self.persistence_mae)

        if logger:
            logger.writerow([
                "Final Avg.", "", "", "", "", "",
                "", f"{final['mae_penalised']:.4f}",
                f"{final['mae_matched']:.4f}" if final["mae_matched"] else "N/A",
                f"{final['step_coverage']:.1f}",
                f"{final['extraction_rate']:.1f}%",
                self.model, self.prompting_mode, "",
            ])
            f.close()

        self.log_prefix.info(f"\n[EnergyForecast] {self.model} | {self.prompting_mode}")
        for k, v in final.items():
            self.log_prefix.info(f"  {k:25s}: {v}")
        return final


# ─────────────────────────────────────────────────────────────────────────────
# LLM Experiment — Energy Extreme
# ─────────────────────────────────────────────────────────────────────────────

class EnergyExtremeExperiment:
    """
    Runs LLM-based extreme energy classification experiment.
    Produces same CSV format and metrics structure as WeatherExtremeExperiment.
    """

    def __init__(
        self,
        data:           List[dict],
        log_prefix,
        model:          str,
        prompting_mode: str,
        fewshot_prompt: str = "",
        concurrency:    int = None,
        timeout:        int = None,
    ):
        self.data           = data
        self.log_prefix     = log_prefix
        self.model          = model
        self.prompting_mode = prompting_mode
        self.fewshot_prompt = fewshot_prompt
        self.metrics        = EnergyExtremeMetrics()
        self.results        = []

        cfg              = _model_config(model, concurrency, timeout)
        self.concurrency = cfg["concurrency"]
        self.timeout     = cfg["timeout"]
        self.semaphore   = asyncio.Semaphore(self.concurrency)

    def run(self, logging: bool = True) -> dict:
        logger = None
        if logging:
            ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = f"results_log/energy_extreme_{self.model}_{self.prompting_mode}_{ts}.csv"
            os.makedirs("results_log", exist_ok=True)
            f      = open(path, "w", newline="", encoding="utf-8-sig")
            logger = csv.writer(f)
            logger.writerow(CSV_ENERGY_EXTREME)

        self.log_prefix.info(
            f"[EnergyExtreme] model={self.model} mode={self.prompting_mode} "
            f"n={len(self.data)} concurrency={self.concurrency} timeout={self.timeout}s")

        async def process(idx, datum):
            async with self.semaphore:
                try:
                    prompt    = datum.get("prompt", "")
                    gt_has, gt_peak = get_label_extreme(datum)
                    gt_slots  = datum.get("ground_truth", {}).get("n_extreme_slots", 0)
                    ctx       = datum.get("context", {})

                    start = time.perf_counter()
                    llm_response = await asyncio.wait_for(
                        talk_to_llm(prompt, fewshot=self.fewshot_prompt,
                                    model=self.model),
                        timeout=self.timeout,
                    )
                    elapsed = time.perf_counter() - start

                    pred_has, pred_peak, extraction_ok = extract_extreme_response(llm_response)
                    scores = self.metrics.update(
                        pred_has=pred_has,  pred_peak_mw=pred_peak,
                        gt_has=gt_has,      gt_peak_mw=gt_peak,
                        extraction_ok=extraction_ok,
                    )

                    self.results.append({
                        "index": idx, "key": datum.get("key", ""),
                        "pred_has": pred_has, "pred_peak": pred_peak,
                        "gt_has": gt_has, "gt_peak": gt_peak,
                        **scores,
                    })

                    if logger:
                        logger.writerow([
                            datum.get("key", ""),
                            ctx.get("region", ""), ctx.get("season", ""),
                            ctx.get("day_of_week", ""),
                            prompt[:200], llm_response[:300],
                            pred_has, pred_peak or "",
                            gt_has, gt_peak, gt_slots,
                            scores["lenient"], scores["consistent"],
                            scores["peak_mae"] or "N/A",
                            extraction_ok,
                            self.model, self.prompting_mode, f"{elapsed:.2f}",
                        ])

                except asyncio.TimeoutError:
                    self.log_prefix.info(
                        f"[TIMEOUT] idx={idx} model={self.model} after {self.timeout}s")
                except Exception as e:
                    self.log_prefix.info(f"[ERROR] idx={idx}: {e}")

        async def run_all():
            tasks = [process(i, d) for i, d in enumerate(self.data)]
            await tqdm_asyncio.gather(*tasks, desc="Energy Extreme LLM")

        asyncio.run(run_all())

        if len(self.results) < len(self.data):
            self.log_prefix.info(
                f"[WARNING] {len(self.results)}/{len(self.data)} processed")

        final = self.metrics.get()

        if logger:
            logger.writerow([
                "Final Avg.", "", "", "", "", "",
                f"TP:{final['tp']}", f"FP:{final['fp']}",
                f"FN:{final['fn']}", f"TN:{final['tn']}", "",
                f"{final['lenient']:.4f}", f"{final['consistency']:.4f}",
                f"{final['peak_mae']:.4f}" if final["peak_mae"] else "N/A",
                f"{final['extraction_rate']:.1f}%",
                self.model, self.prompting_mode, "",
            ])
            f.close()

        self.log_prefix.info(f"\n[EnergyExtreme] {self.model} | {self.prompting_mode}")
        for k, v in final.items():
            self.log_prefix.info(f"  {k:25s}: {v}")
        return final