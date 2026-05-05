# experiments/baselines/features.py
"""
Dataset classes for all supported tasks:
  - weather          : multi-field time series, JSONL format
  - energy_forecast  : univariate load series, JSON array format

Note: weather_extreme and energy_extreme are classification tasks.
      LSTM is not a natural fit for classification here — use the
      rule-based baselines in energy_runner.py and the majority class
      baseline for those tasks instead.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
from torch.utils.data import Dataset


# ─────────────────────────────────────────────────────────────────────────────
# Feature definitions per task
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_WEATHER_FEATURES: List[str] = [
    "east_west_wind_speed_10m",
    "north_south_wind_speed_10m",
    "dewpoint_temperature_2m",
    "air_temperature_2m",
    "mean_sea_level_pressure",
    "surface_pressure",
    "total_precipitation",
]

# Energy forecast is univariate — the single feature is the load value itself
DEFAULT_ENERGY_FEATURES: List[str] = ["load_mw"]


# ─────────────────────────────────────────────────────────────────────────────
# Logging helpers (used by train_lstm.py)
# ─────────────────────────────────────────────────────────────────────────────

def append_log(path: str, epoch: int, split: str, loss_type: str, loss: float):
    with open(path, "a") as f:
        f.write(f"{epoch},{split},{loss_type},{loss:.6f}\n")


def append_info(path: str, info: str):
    with open(path, "a") as f:
        f.write(f"# {info}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Z-score scaler (unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────

class ZScoreScaler:
    """Global z-score per feature dimension. Fit ONLY on train split."""

    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.mean: Optional[np.ndarray] = None
        self.std:  Optional[np.ndarray] = None

    def fit(self, X_2d: np.ndarray) -> "ZScoreScaler":
        if X_2d.ndim != 2:
            raise ValueError(f"fit expects [N, D], got shape={X_2d.shape}")
        self.mean = X_2d.mean(axis=0)
        self.std  = np.maximum(X_2d.std(axis=0), self.eps)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean is None:
            raise RuntimeError("Scaler not fitted.")
        return (X - self.mean) / self.std

    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        if self.mean is None:
            raise RuntimeError("Scaler not fitted.")
        return (X_scaled * self.std) + self.mean


# ─────────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load JSONL — one JSON object per line."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_json_array(path: str) -> List[Dict[str, Any]]:
    """Load JSON array file."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array in {path}")
    return data


def load_data(path: str) -> List[Dict[str, Any]]:
    """
    Robustly loads four file formats:
      1. JSON array:              [ {...}, {...} ]
      2. True JSONL:              {...}\\n{...}\\n
      3. Concatenated JSON objs:  pretty-printed multi-line objects
      4. Single JSON object:      { ... }
    """
    with open(path, encoding="utf-8") as f:
        content = f.read().strip()

    if not content:
        raise ValueError(f"File is empty: {path}")

    first = content[0]

    if first == "[":
        return json.loads(content)

    if first == "{":
        decoder = json.JSONDecoder()
        results = []
        idx     = 0
        n       = len(content)
        while idx < n:
            while idx < n and content[idx] in " \t\n\r":
                idx += 1
            if idx >= n:
                break
            try:
                obj, end = decoder.raw_decode(content, idx)
                results.append(obj)
                idx = end
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"JSON parse error in {path} near char {idx}: {e}"
                ) from e
        return results

    raise ValueError(f"Unrecognised format in {path}. First char: {first!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Horizon parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_horizon_from_question(question: str) -> int:
    """Parse forecast horizon in hours from weather question string."""
    m = re.search(r"\bnext\s+(\d+)\s+hours?\b", question.lower())
    if not m:
        raise ValueError(f"Cannot parse horizon from: {question}")
    return int(m.group(1))


def sorted_obs_timestamps(obs_json: Dict) -> List[str]:
    return sorted([k for k in obs_json.keys() if k != "question"])


# ─────────────────────────────────────────────────────────────────────────────
# Padding utility (shared)
# ─────────────────────────────────────────────────────────────────────────────

def pad_2d(
    X: np.ndarray, T_max: int, pad_value: float = 0.0
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Pad X [T, D] → [T_max, D]. Returns (X_pad, mask, T_use)."""
    T, D   = X.shape
    T_use  = min(T, T_max)
    X_pad  = np.full((T_max, D), pad_value, dtype=np.float32)
    X_pad[:T_use] = X[:T_use]
    mask   = np.zeros((T_max,), dtype=np.float32)
    mask[:T_use] = 1.0
    return X_pad, mask, T_use


# ─────────────────────────────────────────────────────────────────────────────
# LSTMSample dataclass (shared across tasks)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LSTMSample:
    x:      torch.Tensor   # [seq_len, D]
    x_mask: torch.Tensor   # [seq_len]
    x_len:  int

    y:      torch.Tensor   # [H_max, D]
    y_mask: torch.Tensor   # [H_max]
    H:      int            # actual horizon steps

    task:    str           # "weather" or "energy_forecast"
    meta:    dict          # task-specific metadata for evaluator
    x_times: List[str]     # input timestamps (weather) or step indices (energy)
    y_times: List[str]     # output timestamps (weather) or step indices (energy)


def lstm_collate_fn(batch: List[LSTMSample]) -> Dict[str, torch.Tensor]:
    return {
        "x":      torch.stack([b.x      for b in batch]),
        "x_mask": torch.stack([b.x_mask for b in batch]),
        "x_len":  torch.tensor([b.x_len for b in batch], dtype=torch.long),
        "y":      torch.stack([b.y      for b in batch]),
        "y_mask": torch.stack([b.y_mask for b in batch]),
        "H":      torch.tensor([b.H     for b in batch], dtype=torch.long),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Dataset: Weather Forecasting
# Input format:  JSONL with obs_json (dict of timestamp → field dict)
# Output format: ground_truth (dict of timestamp → field dict)
# ─────────────────────────────────────────────────────────────────────────────

class WeatherLSTMDataset(Dataset):
    """
    LSTM dataset for short-term weather forecasting.

    Input datum structure:
    {
      "obs_json": {
        "2012-12-22 06:00:00": {"east_west_wind_speed_10m": -0.35, ...},
        ...
        "question": "Please describe weather observation in the next 3 hours..."
      },
      "ground_truth": {
        "2012-12-22 13:00:00": {"east_west_wind_speed_10m": 0.37, ...},
        ...
      }
    }
    """

    def __init__(
        self,
        path_or_data,                         # file path OR list of dicts
        seq_len:       int = 6,
        H_max:         int = 10,
        feature_order: Optional[List[str]] = None,
        window_mode:   str = "last",
        x_scaler:      Optional[ZScoreScaler] = None,
        y_scaler:      Optional[ZScoreScaler] = None,
        scale_x:       bool = True,
        scale_y:       bool = True,
    ):
        self.rows = (load_data(path_or_data)
                     if isinstance(path_or_data, str) else path_or_data)
        self.seq_len       = seq_len
        self.H_max         = H_max
        self.feature_order = feature_order or DEFAULT_WEATHER_FEATURES
        self.window_mode   = window_mode
        self.x_scaler      = x_scaler
        self.y_scaler      = y_scaler
        self.scale_x       = scale_x
        self.scale_y       = scale_y

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> LSTMSample:
        r        = self.rows[idx]
        obs_json = r["obs_json"]
        gt       = r["ground_truth"]

        question = obs_json.get("question", "")
        H        = parse_horizon_from_question(question)

        x_times_all = sorted_obs_timestamps(obs_json)
        x_times = (x_times_all[-self.seq_len:] if self.window_mode == "last"
                   else x_times_all[:self.seq_len])

        y_times_all = sorted(gt.keys())
        y_times     = y_times_all[:min(H, len(y_times_all))]
        H_eff       = min(len(y_times), self.H_max)

        # Build matrices
        X = np.array([[float(obs_json[t].get(f, 0.0)) for f in self.feature_order]
                      for t in x_times], dtype=np.float32)
        Y = np.array([[float(gt[t].get(f, 0.0)) for f in self.feature_order]
                      for t in y_times[:H_eff]], dtype=np.float32)

        X_pad, x_mask, x_len = pad_2d(X, self.seq_len)
        Y_pad, y_mask, _     = pad_2d(Y, self.H_max)

        if self.scale_x and self.x_scaler is not None:
            X_pad = self.x_scaler.transform(X_pad)
        if self.scale_y and self.y_scaler is not None:
            Y_pad = self.y_scaler.transform(Y_pad)

        area = obs_json[x_times_all[0]].get("area", "")

        return LSTMSample(
            x=torch.from_numpy(X_pad).float(),
            x_mask=torch.from_numpy(x_mask).float(),
            x_len=int(x_len),
            y=torch.from_numpy(Y_pad).float(),
            y_mask=torch.from_numpy(y_mask).float(),
            H=H_eff,
            task="weather",
            meta={"area": area, "ground_truth": gt},
            x_times=x_times,
            y_times=y_times[:self.H_max],
        )


# ─────────────────────────────────────────────────────────────────────────────
# Dataset: Energy Load Forecasting
# Input format:  JSON array with "history" (list of 48 floats)
# Output format: "target" (list of 48 floats)
# ─────────────────────────────────────────────────────────────────────────────

class EnergyForecastLSTMDataset(Dataset):
    """
    LSTM dataset for energy load forecasting.

    Input datum structure:
    {
      "key": "energy_DE_LU_2023-01-02_00:30",
      "task": "energy_forecast",
      "history": [36605.08, 35834.05, ...],   # 48 values
      "target":  [44067.5,  43005.92, ...],   # 48 values
      "context": {"season": "Winter", "day_of_week": "Tuesday", ...}
    }

    Both history and target are treated as univariate [T, 1] tensors.
    seq_len = number of history steps to use as input (default 48).
    H_max   = number of target steps (default 48).
    """

    def __init__(
        self,
        path_or_data,
        seq_len:   int = 48,
        H_max:     int = 48,
        x_scaler:  Optional[ZScoreScaler] = None,
        y_scaler:  Optional[ZScoreScaler] = None,
        scale_x:   bool = True,
        scale_y:   bool = True,
    ):
        self.rows      = (load_data(path_or_data)
                          if isinstance(path_or_data, str) else path_or_data)
        self.seq_len   = seq_len
        self.H_max     = H_max
        self.x_scaler  = x_scaler
        self.y_scaler  = y_scaler
        self.scale_x   = scale_x
        self.scale_y   = scale_y

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> LSTMSample:
        r       = self.rows[idx]
        history = r.get("history", [])
        target  = r.get("target",  [])
        ctx     = r.get("context", {})

        # Use last seq_len values of history
        hist_use = history[-self.seq_len:] if len(history) >= self.seq_len else history
        tgt_use  = target[:self.H_max]

        H_eff = len(tgt_use)

        # Shape to [T, 1] — univariate
        X = np.array([[v] for v in hist_use], dtype=np.float32)
        Y = np.array([[v] for v in tgt_use],  dtype=np.float32)

        X_pad, x_mask, x_len = pad_2d(X, self.seq_len)
        Y_pad, y_mask, _     = pad_2d(Y, self.H_max)

        if self.scale_x and self.x_scaler is not None:
            X_pad = self.x_scaler.transform(X_pad)
        if self.scale_y and self.y_scaler is not None:
            Y_pad = self.y_scaler.transform(Y_pad)

        # String step labels for evaluator (no real timestamps in energy)
        x_times = [f"h-{self.seq_len - i}" for i in range(len(hist_use))]
        y_times = [f"t+{i+1}"              for i in range(H_eff)]

        return LSTMSample(
            x=torch.from_numpy(X_pad).float(),
            x_mask=torch.from_numpy(x_mask).float(),
            x_len=int(x_len),
            y=torch.from_numpy(Y_pad).float(),
            y_mask=torch.from_numpy(y_mask).float(),
            H=H_eff,
            task="energy_forecast",
            meta={
                "key":    r.get("key", ""),
                "target": target,
                "context": ctx,
            },
            x_times=x_times,
            y_times=y_times,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Scaler fitting utilities
# ─────────────────────────────────────────────────────────────────────────────

def fit_weather_scalers(
    path_or_data,
    seq_len:       int = 6,
    H_max:         int = 10,
    feature_order: Optional[List[str]] = None,
    window_mode:   str = "last",
) -> Tuple[ZScoreScaler, ZScoreScaler]:
    """Fit x/y scalers from weather train data."""
    feat = feature_order or DEFAULT_WEATHER_FEATURES
    rows = load_data(path_or_data) if isinstance(path_or_data, str) else path_or_data

    X_all, Y_all = [], []
    for r in rows:
        obs_json = r["obs_json"]
        gt       = r["ground_truth"]
        question = obs_json.get("question", "")
        H        = parse_horizon_from_question(question)

        x_times_all = sorted_obs_timestamps(obs_json)
        x_times = (x_times_all[-seq_len:] if window_mode == "last"
                   else x_times_all[:seq_len])
        y_times = sorted(gt.keys())[:min(H, H_max)]

        X_all.append(np.array([[float(obs_json[t].get(f, 0.0)) for f in feat]
                                for t in x_times], dtype=np.float32))
        Y_all.append(np.array([[float(gt[t].get(f, 0.0)) for f in feat]
                                for t in y_times], dtype=np.float32))

    X_2d = np.concatenate(X_all) if X_all else np.zeros((0, len(feat)), dtype=np.float32)
    Y_2d = np.concatenate(Y_all) if Y_all else np.zeros((0, len(feat)), dtype=np.float32)
    return ZScoreScaler().fit(X_2d), ZScoreScaler().fit(Y_2d)


def fit_energy_scalers(
    path_or_data,
    seq_len: int = 48,
    H_max:   int = 48,
) -> Tuple[ZScoreScaler, ZScoreScaler]:
    """Fit x/y scalers from energy forecast train data."""
    rows = load_data(path_or_data) if isinstance(path_or_data, str) else path_or_data

    X_all, Y_all = [], []
    for r in rows:
        hist = r.get("history", [])[-seq_len:]
        tgt  = r.get("target",  [])[:H_max]
        X_all.append(np.array([[v] for v in hist], dtype=np.float32))
        Y_all.append(np.array([[v] for v in tgt],  dtype=np.float32))

    X_2d = np.concatenate(X_all) if X_all else np.zeros((0, 1), dtype=np.float32)
    Y_2d = np.concatenate(Y_all) if Y_all else np.zeros((0, 1), dtype=np.float32)
    return ZScoreScaler().fit(X_2d), ZScoreScaler().fit(Y_2d)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: build dataset by task name
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset(
    task:          str,
    path_or_data,
    seq_len:       int,
    H_max:         int,
    x_scaler:      Optional[ZScoreScaler] = None,
    y_scaler:      Optional[ZScoreScaler] = None,
    scale_x:       bool = True,
    scale_y:       bool = True,
    feature_order: Optional[List[str]] = None,
    window_mode:   str = "last",
) -> Dataset:
    """
    Factory function — returns the correct Dataset class for the given task.

    Supported tasks:
        "weather"         → WeatherLSTMDataset
        "energy_forecast" → EnergyForecastLSTMDataset
    """
    if task == "weather":
        return WeatherLSTMDataset(
            path_or_data, seq_len=seq_len, H_max=H_max,
            feature_order=feature_order, window_mode=window_mode,
            x_scaler=x_scaler, y_scaler=y_scaler,
            scale_x=scale_x, scale_y=scale_y,
        )
    elif task == "energy_forecast":
        return EnergyForecastLSTMDataset(
            path_or_data, seq_len=seq_len, H_max=H_max,
            x_scaler=x_scaler, y_scaler=y_scaler,
            scale_x=scale_x, scale_y=scale_y,
        )
    else:
        raise ValueError(
            f"Unknown task '{task}'. Supported: 'weather', 'energy_forecast'.\n"
            f"Note: 'weather_extreme' and 'energy_extreme' are classification tasks "
            f"— use rule-based baselines instead of LSTM."
        )


def fit_scalers(
    task:          str,
    path_or_data,
    seq_len:       int,
    H_max:         int,
    feature_order: Optional[List[str]] = None,
    window_mode:   str = "last",
) -> Tuple[ZScoreScaler, ZScoreScaler]:
    """Fit scalers for the given task from train data."""
    if task == "weather":
        return fit_weather_scalers(path_or_data, seq_len, H_max,
                                   feature_order, window_mode)
    elif task == "energy_forecast":
        return fit_energy_scalers(path_or_data, seq_len, H_max)
    else:
        raise ValueError(f"Unknown task '{task}'.")