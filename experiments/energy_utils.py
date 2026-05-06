# experiments/energy_utils.py
"""
Utilities for energy forecasting and extreme energy tasks.

Energy forecast:
  - Input:  48 float values (last 24h at 30-min intervals)
  - Output: 48 float values (next 24h)
  - Metric: MAE penalised + MAE matched + skill score

Energy extreme:
  - Input:  48 float values + explicit threshold
  - Output: YES/NO + peak MW estimate
  - Metric: F1 + precision + recall + peak MAE
"""
from __future__ import annotations

import json
import re
import os
from collections import defaultdict, Counter
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

ENERGY_HORIZON = 48          # fixed output steps (next 24h at 30-min intervals)
ENERGY_HISTORY = 48          # fixed input steps (last 24h)
STEP_MINUTES   = 30          # interval between steps


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_energy_json(path: str) -> List[dict]:
    """Load energy dataset from JSON array file."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array, got {type(data)}")
    return data


def load_energy_jsonl(path: str) -> List[dict]:
    """Load energy dataset from JSONL file (one object per line)."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_energy_data(path: str) -> List[dict]:
    """Auto-detect JSON array vs JSONL format."""
    with open(path, encoding="utf-8") as f:
        first_char = f.read(1)
    if first_char == "[":
        return load_energy_json(path)
    return load_energy_jsonl(path)


# ─────────────────────────────────────────────────────────────────────────────
# Context helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_season(datum: dict) -> str:
    return datum.get("context", {}).get("season", "unknown")


def get_region(datum: dict) -> str:
    return datum.get("context", {}).get("region", "unknown")


def get_label_extreme(datum: dict) -> Tuple[bool, float]:
    """Return (has_extreme, peak_magnitude_mw) from ground_truth."""
    gt = datum.get("ground_truth", {})
    return bool(gt.get("has_extreme", False)), float(gt.get("peak_magnitude_mw", 0.0))


def get_threshold(datum: dict) -> Optional[float]:
    """Extract threshold from prompt text."""
    prompt = datum.get("prompt", "")
    m = re.search(r"threshold[:\s]+([0-9,.]+)\s*MW", prompt, re.IGNORECASE)
    if m:
        return float(m.group(1).replace(",", ""))
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Output extraction from LLM response — Energy Forecast
# ─────────────────────────────────────────────────────────────────────────────

def extract_forecast_numbers(llm_response: str) -> Tuple[List[float], bool]:
    """
    Extract 48 forecast values from LLM response.

    Tries in order:
      1. Comma-separated numbers (as instructed in prompt)
      2. Numbers separated by newlines
      3. Any sequence of numbers found in text

    Returns:
        (values, extraction_ok)
        values: list of floats (empty if extraction failed)
        extraction_ok: True if exactly 48 values were extracted
    """
    if not llm_response or not llm_response.strip():
        return [], False

    # Strip code fences if present
    text = llm_response.strip()
    if text.startswith("```"):
        fence_end = text.rfind("```")
        text = text[3:fence_end].strip() if fence_end > 0 else text[3:].strip()

    # Strategy 1: find a comma-separated block of numbers
    # Look for the longest run of comma-separated numbers
    comma_pattern = re.compile(
        r"(-?\d+(?:\.\d+)?)"           # first number
        r"(?:\s*,\s*-?\d+(?:\.\d+)?)+", # followed by more comma-separated
        re.MULTILINE
    )
    matches = comma_pattern.findall(text)
    # Extract all numbers from the best comma-separated block
    best_block = ""
    for m in re.finditer(r"(-?\d+(?:\.\d+)?)(?:\s*,\s*-?\d+(?:\.\d+)?)+", text):
        if len(m.group(0)) > len(best_block):
            best_block = m.group(0)

    if best_block:
        nums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", best_block)]
        if len(nums) >= 40:   # allow some tolerance
            return nums[:ENERGY_HORIZON], len(nums) >= ENERGY_HORIZON

    # Strategy 2: newline-separated numbers
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    num_lines = []
    for line in lines:
        m = re.match(r"^-?\d+(?:\.\d+)?$", line)
        if m:
            num_lines.append(float(line))
    if len(num_lines) >= 40:
        return num_lines[:ENERGY_HORIZON], len(num_lines) >= ENERGY_HORIZON

    # Strategy 3: any numbers in text (last resort)
    all_nums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", text)]
    # Filter to plausible energy load range (5000–200000 MW)
    plausible = [x for x in all_nums if 5000 <= x <= 200000]
    if len(plausible) >= 40:
        return plausible[:ENERGY_HORIZON], len(plausible) >= ENERGY_HORIZON

    return [], False


# ─────────────────────────────────────────────────────────────────────────────
# Output extraction — Energy Extreme
# ─────────────────────────────────────────────────────────────────────────────

def extract_extreme_response(llm_response: str) -> Tuple[bool, Optional[float], bool]:
    """
    Extract (has_extreme, peak_mw_estimate, extraction_ok) from LLM response.

    Expected format: "YES\nPeak load estimate: 67500 MW"
    or:              "NO"
    """
    if not llm_response:
        return False, None, False

    text = llm_response.strip().lower()

    # Determine Yes/No
    has_extreme = None
    if text.startswith("yes"):
        has_extreme = True
    elif text.startswith("no"):
        has_extreme = False
    else:
        m = re.search(r"\b(yes|no)\b", text)
        has_extreme = (m.group(1) == "yes") if m else False

    # Extract peak MW estimate (only meaningful if Yes)
    peak_mw = None
    if has_extreme:
        # Patterns: "67500 MW", "peak: 67500", "estimate: 67,500"
        patterns = [
            r"peak[^\d]*([0-9,]+(?:\.\d+)?)\s*(?:mw|megawatt)?",
            r"([0-9,]+(?:\.\d+)?)\s*mw",
            r"estimate[^\d]*([0-9,]+(?:\.\d+)?)",
        ]
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                try:
                    peak_mw = float(m.group(1).replace(",", ""))
                    if 5000 <= peak_mw <= 200000:   # plausibility check
                        break
                    else:
                        peak_mw = None
                except ValueError:
                    continue

    extraction_ok = has_extreme is not None
    return has_extreme or False, peak_mw, extraction_ok


# ─────────────────────────────────────────────────────────────────────────────
# MAE computation — Energy Forecast
# ─────────────────────────────────────────────────────────────────────────────

def compute_forecast_mae(
    pred: List[float],
    target: List[float],
) -> Tuple[float, float, float]:
    """
    Compute penalised MAE, matched MAE, and step coverage.

    Penalised MAE: missing predictions → 0.0 (same logic as weather)
    Matched MAE:   only on steps where prediction exists
    Step coverage: % of 48 steps successfully predicted

    Returns: (mae_penalised, mae_matched, step_coverage_pct)
    """
    n_target = len(target)
    if n_target == 0:
        return 0.0, 0.0, 0.0

    n_pred    = len(pred)
    n_matched = min(n_pred, n_target)

    # Penalised MAE — pad pred with 0.0 if shorter than target
    pred_padded = list(pred) + [0.0] * max(0, n_target - n_pred)
    mae_pen = sum(abs(p - t) for p, t in zip(pred_padded, target)) / n_target

    # Matched MAE — only on available steps
    if n_matched > 0:
        mae_matched = sum(
            abs(pred[i] - target[i]) for i in range(n_matched)
        ) / n_matched
    else:
        mae_matched = None

    step_coverage = (n_matched / n_target) * 100.0

    return mae_pen, mae_matched, step_coverage


def compute_skill_score(llm_mae: float, baseline_mae: float) -> Optional[float]:
    """skill = 1 - (llm_mae / baseline_mae). None if baseline_mae = 0."""
    if baseline_mae and baseline_mae > 0:
        return round(1.0 - (llm_mae / baseline_mae), 4)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Baselines — Energy Forecast
# ─────────────────────────────────────────────────────────────────────────────

def persistence_forecast(history: List[float], horizon: int = ENERGY_HORIZON) -> List[float]:
    """
    Copy last `horizon` values of history forward as the forecast.
    If history has fewer than horizon values, repeat the last value.
    """
    if not history:
        return [0.0] * horizon
    if len(history) >= horizon:
        return list(history[-horizon:])
    # pad by repeating last value
    return list(history) + [history[-1]] * (horizon - len(history))


def seasonal_naive_forecast(
    full_history: List[float],
    horizon: int = ENERGY_HORIZON,
    steps_per_week: int = 48 * 7,   # 336 steps = 7 days × 48 steps/day
) -> Optional[List[float]]:
    """
    Seasonal naive: predict next 48 steps = same period one week ago.
    Requires at least steps_per_week + horizon values in full_history.
    Returns None if insufficient history.
    """
    needed = steps_per_week + horizon
    if len(full_history) < needed:
        return None
    start = len(full_history) - steps_per_week
    return list(full_history[start: start + horizon])


# ─────────────────────────────────────────────────────────────────────────────
# Baselines — Energy Extreme
# ─────────────────────────────────────────────────────────────────────────────

def majority_class_extreme(data: List[dict]) -> Tuple[bool, str]:
    """
    Returns the majority class (has_extreme value) across dataset.
    Used to configure the majority class baseline predictor.
    """
    counts = Counter(get_label_extreme(d)[0] for d in data)
    majority = counts.most_common(1)[0][0]
    return majority


def threshold_rule_baseline(
    history: List[float],
    threshold: float,
    lookback_slots: int = 12,     # last 6 hours = 12 × 30-min slots
    trigger_fraction: float = 0.95,
) -> Tuple[bool, Optional[float]]:
    """
    Rule-based extreme baseline:
      If max of recent history > trigger_fraction × threshold → predict Yes
      Peak estimate = max of recent history (if Yes)

    This is meaningful because:
      - It uses exactly the same information given to the LLM (history + threshold)
      - It requires no learning or language understanding
      - If LLM cannot beat this, it is not reasoning beyond mechanical threshold use
    """
    recent      = history[-lookback_slots:] if len(history) >= lookback_slots else history
    recent_max  = max(recent) if recent else 0.0
    pred_has    = recent_max > (threshold * trigger_fraction)
    peak_est    = recent_max if pred_has else None
    return pred_has, peak_est


# ─────────────────────────────────────────────────────────────────────────────
# Few-shot split — Energy Forecast
# ─────────────────────────────────────────────────────────────────────────────

def fixed_fewshot_energy_split(
    data: List[dict],
    seed: int = 42,
) -> Tuple[List[dict], List[dict]]:
    """
    Select 1 few-shot example per season (Winter/Spring/Summer/Autumn).
    Stratified by season so fewshot represents seasonal distribution.
    Seed controls which example is picked within each season group.
    Returns (few_shot_examples, test_data).
    """
    from experiments.multiseed_eval import stratified_multiseed_split
    return stratified_multiseed_split(
        data,
        get_stratum_fn=get_season,
        n_per_stratum=1,
        seed=seed,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Few-shot split — Energy Extreme
# ─────────────────────────────────────────────────────────────────────────────

def fixed_fewshot_energy_extreme_split(
    data: List[dict],
    seed: int = 42,
) -> Tuple[List[dict], List[dict]]:
    """
    Select 1 few-shot example per class (True/False).
    Stratified by class so fewshot represents class distribution.
    Seed controls which example is picked within each class group.
    Returns (few_shot_examples, test_data).
    """
    from experiments.multiseed_eval import stratified_multiseed_split
    return stratified_multiseed_split(
        data,
        get_stratum_fn=lambda d: get_label_extreme(d)[0],
        n_per_stratum=1,
        seed=seed,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Prompt formatters
# ─────────────────────────────────────────────────────────────────────────────

def format_energy_fewshot_prompt(examples: List[dict]) -> str:
    """Format energy forecast few-shot examples into initial_prompt string."""
    lines = [
        "You are an electricity load forecasting assistant.",
        "Here are some examples of how to answer:\n",
    ]
    for i, datum in enumerate(examples, 1):
        ctx     = datum.get("context", {})
        history = datum.get("history", [])
        target  = datum.get("target", [])
        lines.append(f"--- Example {i} ({ctx.get('season','')}, {ctx.get('day_of_week','')}) ---")
        lines.append(f"Historical load (48 values):\n{', '.join(str(v) for v in history)}")
        lines.append(f"Answer (48 values):\n{', '.join(str(v) for v in target)}\n")

    lines += [
        "--- Now answer the following ---",
        "Output exactly 48 comma-separated numbers. No explanation.",
        "",
    ]
    return "\n".join(lines)


def format_energy_extreme_fewshot_prompt(examples: List[dict]) -> str:
    """Format energy extreme few-shot examples into initial_prompt string."""
    lines = [
        "You are an electricity grid analyst.",
        "Here are some examples of how to answer:\n",
    ]
    for i, datum in enumerate(examples, 1):
        has, peak = get_label_extreme(datum)
        ctx = datum.get("context", {})
        answer = "YES" if has else "NO"
        peak_line = f"\nPeak load estimate: {peak:.2f} MW" if has and peak else ""
        lines.append(f"--- Example {i} ({ctx.get('season','')}, {ctx.get('day_of_week','')}) ---")
        lines.append(datum.get("prompt", "").strip())
        lines.append(f"Answer: {answer}{peak_line}\n")

    lines += [
        "--- Now answer the following ---",
        "Answer YES or NO. If YES, also provide peak load estimate in MW.",
        "",
    ]
    return "\n".join(lines)


def format_ragfs_energy_prompt(examples: List[dict]) -> str:
    """Build a dynamic few-shot prompt from RAG-retrieved energy forecast examples.

    Called at inference time by EnergyForecastExperiment when rag_retriever is set.
    Examples are already sorted by similarity (most similar first) by the retriever.
    """
    lines = [
        "You are an electricity load forecasting assistant.",
        "The following examples were retrieved because they are most similar "
        "to the current query:\n",
    ]
    for i, datum in enumerate(examples, 1):
        ctx     = datum.get("context", {})
        history = datum.get("history", [])
        target  = datum.get("target", [])
        lines.append(f"--- Retrieved Example {i} "
                     f"({ctx.get('season','')}, {ctx.get('day_of_week','')}) ---")
        lines.append(f"Historical load (48 values):\n{', '.join(str(v) for v in history)}")
        lines.append(f"Answer (48 values):\n{', '.join(str(v) for v in target)}\n")

    lines += [
        "--- Now answer the following ---",
        "Output exactly 48 comma-separated numbers. No explanation.",
        "",
    ]
    return "\n".join(lines)


def format_ragfs_energy_extreme_prompt(examples: List[dict]) -> str:
    """Build a dynamic few-shot prompt from RAG-retrieved energy extreme examples.

    Called at inference time by EnergyExtremeExperiment when rag_retriever is set.
    Examples are already sorted by similarity (most similar first) by the retriever.
    """
    lines = [
        "You are an electricity grid analyst.",
        "The following examples were retrieved because they are most similar "
        "to the current query:\n",
    ]
    for i, datum in enumerate(examples, 1):
        has, peak = get_label_extreme(datum)
        ctx = datum.get("context", {})
        answer = "YES" if has else "NO"
        peak_line = f"\nPeak load estimate: {peak:.2f} MW" if has and peak else ""
        lines.append(f"--- Retrieved Example {i} "
                     f"({ctx.get('season','')}, {ctx.get('day_of_week','')}) ---")
        lines.append(datum.get("prompt", "").strip())
        lines.append(f"Answer: {answer}{peak_line}\n")

    lines += [
        "--- Now answer the following ---",
        "Answer YES or NO. If YES, also provide peak load estimate in MW.",
        "",
    ]
    return "\n".join(lines)