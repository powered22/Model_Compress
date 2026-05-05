# experiments/energy_metrics.py
"""
EnergyMetrics class for energy forecasting and extreme energy tasks.

Mirrors the Metrics class interface so results feed into the same
summary CSV pipeline used by weather experiments.

Energy Forecast metrics:
  - mae_penalised     (missing steps → 0.0, same as weather l1loss)
  - mae_matched       (only on predicted steps, same as weather l1loss_matched)
  - step_coverage     (% of 48 steps present in prediction)
  - skill_penalised   (vs persistence baseline)
  - skill_matched     (vs persistence baseline, matched only)

Energy Extreme metrics (mirrors WeatherExtremeExperiment):
  - accuracy, precision, recall, f1
  - peak_mae          (MAE of peak MW estimate, only on True Positives)
  - consistency       (No → peak=None, Yes → peak>0)
  - tp, fp, fn, tn
"""
from __future__ import annotations

from typing import Dict, Optional, List, Tuple
from experiments.energy_utils import compute_forecast_mae, compute_skill_score


class EnergyForecastMetrics:
    """
    Accumulates metrics for energy load forecasting task.
    Compatible with the summary CSV format used by WeatherExperiment.
    """

    def __init__(self):
        self._n                   = 0
        self._mae_pen_sum         = 0.0
        self._mae_matched_sum     = 0.0
        self._mae_matched_count   = 0     # samples with ≥1 predicted step
        self._step_coverage_sum   = 0.0
        self._extraction_ok_count = 0

    def update(
        self,
        pred: List[float],
        target: List[float],
        extraction_ok: bool,
    ) -> dict:
        """
        Score one sample. Returns per-sample scores dict.

        Args:
            pred:          extracted forecast values (may be empty)
            target:        ground truth 48 values
            extraction_ok: whether extraction succeeded
        """
        self._n += 1

        mae_pen, mae_matched, step_cov = compute_forecast_mae(pred, target)

        self._mae_pen_sum       += mae_pen
        self._step_coverage_sum += step_cov

        if mae_matched is not None:
            self._mae_matched_sum   += mae_matched
            self._mae_matched_count += 1

        if extraction_ok:
            self._extraction_ok_count += 1

        return {
            "mae_penalised":    mae_pen,
            "mae_matched":      mae_matched,
            "step_coverage":    step_cov,
            "extraction_ok":    extraction_ok,
            "n_pred_steps":     len(pred),
        }

    def get(self, persistence_mae: float = None) -> dict:
        """Return averaged metrics. Optionally compute skill scores."""
        if self._n == 0:
            return self._empty()

        mae_pen     = self._mae_pen_sum     / self._n
        step_cov    = self._step_coverage_sum / self._n
        extraction  = (self._extraction_ok_count / self._n) * 100.0

        mae_matched = (self._mae_matched_sum / self._mae_matched_count
                       if self._mae_matched_count > 0 else None)

        result = {
            "mae_penalised":      mae_pen,
            "mae_matched":        mae_matched,
            "step_coverage":      step_cov,
            "extraction_rate":    extraction,
            "n_samples":          self._n,
        }

        if persistence_mae and persistence_mae > 0:
            result["skill_penalised"] = compute_skill_score(mae_pen, persistence_mae)
            if mae_matched is not None:
                result["skill_matched"] = compute_skill_score(mae_matched, persistence_mae)
            else:
                result["skill_matched"] = None

        return result

    @staticmethod
    def _empty() -> dict:
        return {
            "mae_penalised": 0.0, "mae_matched": None,
            "step_coverage": 0.0, "extraction_rate": 0.0, "n_samples": 0,
        }


class EnergyExtremeMetrics:
    """
    Accumulates metrics for extreme energy classification task.
    Mirrors update_weather_extreme() / get() interface from Metrics class.
    """

    def __init__(self):
        self._n                  = 0
        self._tp = self._fp = self._fn = self._tn = 0
        self._lenient_correct    = 0
        self._consistent_correct = 0

        # Peak MW estimation (only meaningful on True Positives)
        self._peak_mae_sum       = 0.0
        self._peak_mae_count     = 0
        self._extraction_ok      = 0

    def update(
        self,
        pred_has:      bool,
        pred_peak_mw:  Optional[float],
        gt_has:        bool,
        gt_peak_mw:    float,
        extraction_ok: bool = True,
    ) -> dict:
        """
        Score one sample.

        Args:
            pred_has:      predicted has_extreme
            pred_peak_mw:  predicted peak MW (None if pred_has=False)
            gt_has:        ground truth has_extreme
            gt_peak_mw:    ground truth peak_magnitude_mw
            extraction_ok: whether LLM response was parseable
        """
        self._n += 1

        if extraction_ok:
            self._extraction_ok += 1

        # Confusion matrix
        if pred_has and gt_has:       self._tp += 1
        elif pred_has and not gt_has: self._fp += 1
        elif not pred_has and gt_has: self._fn += 1
        else:                         self._tn += 1

        # Lenient: binary label correct
        lenient = (pred_has == gt_has)
        if lenient:
            self._lenient_correct += 1

        # Consistency: No → peak should be None/0, Yes → peak should be >0
        consistent = (
            (not pred_has and (pred_peak_mw is None or pred_peak_mw == 0.0)) or
            (pred_has and pred_peak_mw is not None and pred_peak_mw > 0.0)
        )
        if consistent:
            self._consistent_correct += 1

        # Peak MAE — only on true positives where both have peak values
        peak_mae = None
        if gt_has and pred_has and gt_peak_mw > 0 and pred_peak_mw is not None:
            peak_mae = abs(pred_peak_mw - gt_peak_mw)
            self._peak_mae_sum   += peak_mae
            self._peak_mae_count += 1

        return {
            "lenient":      1.0 if lenient    else 0.0,
            "consistent":   1.0 if consistent else 0.0,
            "peak_mae":     peak_mae,
            "extraction_ok": extraction_ok,
        }

    def cget(self) -> dict:
        """Return averaged classification metrics."""
        n = self._n
        if n == 0:
            return self._empty()

        tp, fp, fn, tn = self._tp, self._fp, self._fn, self._tn
        acc  = (tp + tn) / n
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0

        peak_mae = (self._peak_mae_sum / self._peak_mae_count
                    if self._peak_mae_count > 0 else None)

        return {
            "accuracy":      acc,
            "precision":     prec,
            "recall":        rec,
            "f1":            f1,
            "peak_mae":      peak_mae,
            "lenient":       self._lenient_correct    / n,
            "consistency":   self._consistent_correct / n,
            "extraction_rate": (self._extraction_ok / n) * 100.0,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "n_samples": n,
        }

    @staticmethod
    def _empty() -> dict:
        return {
            "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
            "peak_mae": None, "lenient": 0.0, "consistency": 0.0,
            "extraction_rate": 0.0, "tp": 0, "fp": 0, "fn": 0, "tn": 0,
            "n_samples": 0,
        }