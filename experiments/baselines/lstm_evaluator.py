# experiments/baselines/lstm_evaluator.py
"""
Task-aware LSTM evaluator.

Supports:
    task="weather"          → feeds into Metrics.update_weather()
    task="energy_forecast"  → feeds into EnergyForecastMetrics.update()

Usage:
    evaluator = LSTMEvaluator(
        checkpoint_path="results_log/lstm_weather_best.pt",
        train_data="data/weather/train.jsonl",
        test_data=test_data_list,
        task="weather",
        log_prefix=log,
        persistence_scores=persistence_scores,
    )
    scores = evaluator.run(logging=True)
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from experiments.baselines.features import (
    build_dataset, fit_scalers, lstm_collate_fn,
    sorted_obs_timestamps, parse_horizon_from_question,
    DEFAULT_WEATHER_FEATURES, DEFAULT_ENERGY_FEATURES,
    ZScoreScaler,
)
from experiments.baselines.lstm_model import LSTMBaselineOptionB, LSTMModelConfig
from experiments.metrics import Metrics
from experiments.energy_metrics import EnergyForecastMetrics
from experiments.utils import Logger, CSV_WEATHER
from experiments.runner import compute_skill_scores


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _generate_future_timestamps(last_ts: str, horizon: int) -> List[str]:
    fmt     = "%Y-%m-%d %H:%M:%S"
    last_dt = datetime.strptime(last_ts, fmt)
    return [(last_dt + timedelta(hours=h)).strftime(fmt)
            for h in range(1, horizon + 1)]


def _weather_pred_dict(
    y_hat_row: np.ndarray,    # [H_max, D] — inverse-transformed
    y_times:   List[str],
    H:         int,
    feature_order: List[str],
) -> Dict[str, Dict[str, Any]]:
    """Convert LSTM output tensor to ground_truth-compatible dict."""
    pred = {}
    for h_idx, ts in enumerate(y_times[:H]):
        pred[ts] = {f: float(y_hat_row[h_idx, fi])
                    for fi, f in enumerate(feature_order)}
    return pred


# ─────────────────────────────────────────────────────────────────────────────
# LSTMEvaluator
# ─────────────────────────────────────────────────────────────────────────────

class LSTMEvaluator:
    """
    Loads a trained LSTM checkpoint and evaluates on test data.
    Supports task="weather" and task="energy_forecast".
    Results feed into the same Metrics pipeline as LLM experiments.
    """

    SUPPORTED_TASKS = ("weather", "energy_forecast")

    def __init__(
        self,
        checkpoint_path:    str,
        train_data,               # path or list — used to refit scalers
        test_data:          List[dict],
        task:               str,
        log_prefix,
        seq_len:            int  = None,
        H_max:              int  = None,
        device:             str  = None,
        feature_order:      Optional[List[str]] = None,
        persistence_scores: dict = None,
    ):
        if task not in self.SUPPORTED_TASKS:
            raise ValueError(
                f"task='{task}' not supported by LSTMEvaluator. "
                f"Supported: {self.SUPPORTED_TASKS}. "
                f"For classification tasks use rule-based baselines."
            )

        self.checkpoint_path    = checkpoint_path
        self.train_data         = train_data
        self.test_data          = test_data
        self.task               = task
        self.log_prefix         = log_prefix
        self.persistence_scores = persistence_scores
        self.results            = []

        # Resolve defaults per task
        _defaults = {"weather": (6, 10), "energy_forecast": (48, 48)}
        default_seq, default_hmax = _defaults[task]
        self.seq_len       = seq_len or default_seq
        self.H_max         = H_max   or default_hmax
        self.feature_order = (feature_order or
                              (DEFAULT_WEATHER_FEATURES if task == "weather"
                               else DEFAULT_ENERGY_FEATURES))

        # Device
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
            print(f"[LSTMEvaluator] GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = "cpu"
            print("[LSTMEvaluator] CPU")

        # Metrics accumulator — task-specific
        self.metrics = (Metrics() if task == "weather"
                        else EnergyForecastMetrics())

    # ─────────────────────────────────────────────────────────────────────────

    def _load_model_and_scalers(self):
        ckpt  = torch.load(self.checkpoint_path, map_location=self.device)
        cfg   = LSTMModelConfig(**ckpt["model_config"])
        model = LSTMBaselineOptionB(cfg).to(self.device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        x_scaler, y_scaler = fit_scalers(
            task=self.task, path_or_data=self.train_data,
            seq_len=self.seq_len, H_max=self.H_max,
        )
        return model, x_scaler, y_scaler

    # ─────────────────────────────────────────────────────────────────────────

    def run(self, logging: bool = True) -> dict:
        self.log_prefix.info(
            f"[LSTMEvaluator] task={self.task} "
            f"ckpt={self.checkpoint_path} n={len(self.test_data)}")

        model, x_scaler, y_scaler = self._load_model_and_scalers()
        logger = Logger("weather", f"lstm_{self.task}", CSV_WEATHER) if logging else None

        # Write test data to tmp file for Dataset
        tmp = f"_tmp_lstm_{self.task}_eval.jsonl"
        with open(tmp, "w", encoding="utf-8") as f:
            for d in self.test_data:
                f.write(json.dumps(d) + "\n")

        ds = build_dataset(
            task=self.task, path_or_data=tmp,
            seq_len=self.seq_len, H_max=self.H_max,
            x_scaler=x_scaler, y_scaler=y_scaler,
            scale_x=True, scale_y=True,
        )

        batch_size = 16 if self.device == "cuda" else 1
        dl = DataLoader(
            ds, batch_size=batch_size, shuffle=False,
            collate_fn=lstm_collate_fn,
            pin_memory=(self.device == "cuda"),
        )

        sample_idx = 0
        with torch.no_grad():
            for batch in dl:
                batch_dev    = {k: v.to(self.device) for k, v in batch.items()}
                y_hat_scaled = model(batch_dev)       # [B, H_max, D]

                for b in range(y_hat_scaled.shape[0]):
                    datum  = self.test_data[sample_idx]
                    y_hat  = y_hat_scaled[b].cpu().numpy()          # [H_max, D]
                    y_orig = y_scaler.inverse_transform(y_hat)       # original units

                    scores = self._score_sample(datum, y_orig, logger, sample_idx)
                    self.results.append({"index": sample_idx, **scores})
                    sample_idx += 1

        if os.path.exists(tmp):
            os.remove(tmp)

        final = self._get_final_scores()

        if logger:
            logger.end()

        self.log_prefix.info(f"\n=== LSTM Baseline ({self.task}) ===")
        for k, v in final.items():
            if v is not None:
                self.log_prefix.info(
                    f"  {k:30s}: {v:.4f}" if isinstance(v, float) else
                    f"  {k:30s}: {v}")
        return final

    # ─────────────────────────────────────────────────────────────────────────
    # Task-specific scoring
    # ─────────────────────────────────────────────────────────────────────────

    def _score_sample(self, datum: dict, y_orig: np.ndarray,
                      logger, idx: int) -> dict:
        if self.task == "weather":
            return self._score_weather(datum, y_orig, logger, idx)
        elif self.task == "energy_forecast":
            return self._score_energy(datum, y_orig, logger, idx)

    def _score_weather(self, datum, y_orig, logger, idx):
        obs_json     = datum["obs_json"]
        ground_truth = datum["ground_truth"]

        question = obs_json.get("question", "")
        horizon  = parse_horizon_from_question(question)
        obs_ts   = sorted_obs_timestamps(obs_json)
        y_times  = _generate_future_timestamps(obs_ts[-1], horizon)

        pred_dict = _weather_pred_dict(y_orig, y_times, horizon, self.feature_order)

        scores = self.metrics.update_weather(
            json_llm_response=pred_dict,
            sentence_llm_response="",
            ground_truth=ground_truth,
            result_sentence="",
            mode_json=True,
        )

        if logger:
            logger.log([
                idx, "lstm_weather", ground_truth,
                json.dumps(pred_dict), json.dumps(pred_dict),
                scores["levenshtein_distance"],
                scores["norm_levenshtein_distance"],
                scores["lcs"], scores["norm_lcs"],
                scores["l1loss"],
                scores.get("l1loss_matched"),
                scores.get("timestamp_coverage", 100.0),
                scores["coverage"],
                scores["bertscore"], scores["cosinesim"],
                scores["bleu"], scores["rouge_l"],
                scores["strict"], scores["lenient"],
                "lstm_weather", None, None, "lstm_weather", 0.0,
            ])
        return scores

    def _score_energy(self, datum, y_orig, logger, idx):
        target = datum.get("target", [])
        pred   = [float(y_orig[i, 0]) for i in range(min(self.H_max, len(target)))]

        scores = self.metrics.update(pred, target, extraction_ok=True)

        if logger:
            logger.log([
                idx,
                datum.get("context", {}).get("region", ""),
                datum.get("context", {}).get("season", ""),
                datum.get("context", {}).get("day_of_week", ""),
                datum.get("prompt", "")[:200],
                ",".join(f"{v:.2f}" for v in pred),
                scores["n_pred_steps"],
                f"{scores['mae_penalised']:.4f}",
                f"{scores['mae_matched']:.4f}" if scores["mae_matched"] else "N/A",
                f"{scores['step_coverage']:.1f}",
                True,
                "lstm_energy_forecast", "lstm_baseline", 0.0,
            ])
        return scores

    def _get_final_scores(self) -> dict:
        if self.task == "weather":
            final = self.metrics.get(subject="weather")
            if self.persistence_scores:
                skill = compute_skill_scores(final, self.persistence_scores)
                final.update(skill)
        else:
            pers_mae = (self.persistence_scores.get("mae_penalised")
                        if self.persistence_scores else None)
            final = self.metrics.get(persistence_mae=pers_mae)
        return final