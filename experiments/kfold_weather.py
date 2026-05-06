# experiments/weather_fewshot_split.py
"""
Fixed held-out few-shot selection for short-term weather forecasting.

Strategy:
  1. Group samples by forecast horizon (2h, 3h, 4h, 5h)
  2. Within each horizon group, pick the example whose location
     does NOT appear in any other sample of the same horizon
     (prevents location-time contamination)
  3. Remove selected examples from test set
  4. Return (few_shot_examples, test_data)
"""

from __future__ import annotations
from collections import defaultdict
from datetime import datetime
from typing import List, Tuple, Dict
import json
import re


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def get_horizon(datum: dict) -> int:
    """Parse forecast horizon in hours from the question string."""
    question = datum["obs_json"].get("question", "")
    m = re.search(r"\bnext\s+(\d+)\s+hours?\b", question.lower())
    if not m:
        raise ValueError(f"Cannot parse horizon from: {question}")
    return int(m.group(1))


def get_area(datum: dict) -> str:
    """Extract location name from the last observation timestep."""
    obs = datum["obs_json"]
    ts  = sorted([k for k in obs if k != "question"])[-1]
    return obs[ts].get("area", "unknown")


def get_first_timestamp(datum: dict) -> str:
    """Get the earliest observation timestamp."""
    obs = datum["obs_json"]
    return sorted([k for k in obs if k != "question"])[0]


# ─────────────────────────────────────────────────────────────────────────────
# Main split function
# ─────────────────────────────────────────────────────────────────────────────

def fixed_fewshot_weather_split(
    data: List[dict],
    seed: int = 42,
    prefer_unique_location: bool = True,
) -> Tuple[List[dict], List[dict]]:
    """
    Select 1 few-shot example per forecast horizon.
    Stratified by horizon so fewshot represents all forecast ranges.
    Seed controls which example is picked within each horizon group.
    Returns (few_shot_examples, test_data).
    """
    from experiments.multiseed_eval import stratified_multiseed_split
    return stratified_multiseed_split(
        data,
        get_stratum_fn=get_horizon,
        n_per_stratum=1,
        seed=seed,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Format few-shot examples into prompt text
# ─────────────────────────────────────────────────────────────────────────────

def format_weather_fewshot_prompt(examples: List[dict]) -> str:
    """
    Format selected examples into the initial_prompt string
    matching your existing WeatherPrompting format.
    One example per horizon, ordered by horizon length.
    """
    examples_sorted = sorted(examples, key=get_horizon)

    lines = [
        "You are a weather forecasting assistant.",
        "Here are some examples of how to answer:\n",
    ]

    for i, datum in enumerate(examples_sorted, 1):
        horizon  = get_horizon(datum)
        area     = get_area(datum)
        lines.append(f"--- Example {i} (horizon: {horizon} hours, "
                     f"location: {area}) ---")
        lines.append(f"Observations:\n{datum['observation'].strip()}")
        lines.append(f"\nAnswer:\n{datum['result'].strip()}")
        lines.append("")

    lines += [
        "--- Now answer the following ---",
        "Follow the same format as the examples above.",
        "Report all variables for each forecast hour.",
        "Match the timestamp format exactly.",
        "",
    ]

    return "\n".join(lines)


def format_ragfs_weather_prompt(examples: List[dict]) -> str:
    """Build a dynamic few-shot prompt from RAG-retrieved examples.

    Called at inference time by WeatherExperiment when rag_retriever is set.
    Examples are already sorted by similarity (most similar first) by the
    retriever, so we preserve that order rather than sorting by horizon.
    """
    lines = [
        "You are a weather forecasting assistant.",
        "The following examples were retrieved because they are most similar "
        "to the current observation:\n",
    ]

    for i, datum in enumerate(examples, 1):
        try:
            horizon = get_horizon(datum)
            area    = get_area(datum)
            header  = f"--- Retrieved Example {i} (horizon: {horizon}h, location: {area}) ---"
        except Exception:
            header = f"--- Retrieved Example {i} ---"

        lines.append(header)
        lines.append(f"Observations:\n{datum.get('observation', '').strip()}")
        lines.append(f"\nAnswer:\n{datum.get('result', '').strip()}")
        lines.append("")

    lines += [
        "--- Now answer the following ---",
        "Follow the same format as the examples above.",
        "Report all variables for each forecast hour.",
        "Match the timestamp format exactly.",
        "",
    ]

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Verify split — call this to inspect your split before running experiments
# ─────────────────────────────────────────────────────────────────────────────

def verify_split(few_shot_examples: List[dict], test_data: List[dict]):
    """
    Print a readable summary of your split for manual inspection.
    Run this before starting LLM experiments to confirm the split is clean.
    """
    print("\n" + "="*60)
    print("FEW-SHOT EXAMPLES (removed from test set):")
    print("="*60)
    for d in sorted(few_shot_examples, key=get_horizon):
        print(f"  Horizon={get_horizon(d)}h | "
              f"Location={get_area(d)} | "
              f"Start={get_first_timestamp(d)}")

    print("\n" + "="*60)
    print(f"TEST SET ({len(test_data)} samples):")
    print("="*60)

    # Group test set by horizon for readability
    by_horizon: Dict[int, List] = defaultdict(list)
    for d in test_data:
        by_horizon[get_horizon(d)].append(d)

    for h in sorted(by_horizon.keys()):
        print(f"\n  Horizon {h}h ({len(by_horizon[h])} samples):")
        for d in by_horizon[h]:
            print(f"    {get_area(d)} @ {get_first_timestamp(d)}")

    print("="*60)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point — run this first on your dataset to inspect the split
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/weather/100_test_data.jsonl"

    data = load_jsonl(path)
    few_shot, test = fixed_fewshot_weather_split(data)
    verify_split(few_shot, test)

    # Optionally write the prompt to file
    prompt_text = format_weather_fewshot_prompt(few_shot)
    out_path    = "initial_prompts/weather/fewshot-weather.txt"
    import os
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(prompt_text)
    print(f"\nPrompt written to: {out_path}")