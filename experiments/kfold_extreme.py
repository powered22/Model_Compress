# experiments/kfold_extreme.py
"""
5-Fold Cross-Validation wrapper for WeatherExtremeExperiment.
Produces stable F1/precision/recall metrics even with only 100 samples.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional
from experiments.runner import WeatherExtremeExperiment
from experiments.utils import get_logging


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> List[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def get_label(datum: dict) -> Tuple[bool, str]:
    gt_obj = datum["ground_truth"]
    _, gt  = next(iter(gt_obj.items()))
    return bool(gt["has_extreme_weather"]), str(gt["event_type"])


def print_class_distribution(data: List[dict], label: str = "Dataset"):
    """Print class distribution — run this first on your 100-sample dataset."""
    counts: Counter = Counter()
    for d in data:
        has, etype = get_label(d)
        counts[(has, etype)] += 1

    total = len(data)
    pos   = sum(v for (has, _), v in counts.items() if has)
    neg   = total - pos

    print(f"\n{'='*50}")
    print(f"[{label}] Total: {total} samples")
    print(f"  Positive (extreme=True):  {pos} ({pos/total:.1%})")
    print(f"  Negative (extreme=False): {neg} ({neg/total:.1%})")
    print(f"\n  Per-class breakdown:")
    for (has, etype), count in sorted(counts.items()):
        print(f"    has={has}, type={etype:20s}: {count:3d} ({count/total:.1%})")
    print('='*50)
    return pos, neg


# ─────────────────────────────────────────────────────────────────────────────
# Stratified K-Fold splitter
# ─────────────────────────────────────────────────────────────────────────────

def stratified_kfold(
    data: List[dict],
    k: int = 5,
    seed: int = 42,
) -> List[Tuple[List[dict], List[dict]]]:
    """
    Stratified K-Fold split preserving class distribution across folds.

    Handles rare classes (fewer samples than k) by placing rare-class
    samples only in training folds rather than crashing or producing
    empty test folds. This is necessary for your dataset where
    Flash Flood, Flood, and Lightning each have only 1 sample.
    """
    random.seed(seed)

    # Group data by (has_extreme, event_type) label
    groups: Dict[tuple, List[dict]] = defaultdict(list)
    for datum in data:
        label = get_label(datum)
        groups[label].append(datum)

    # Separate rare classes (fewer samples than k) from normal classes
    normal_groups = {key: items for key, items in groups.items()
                     if len(items) >= k}
    rare_groups   = {key: items for key, items in groups.items()
                     if len(items) < k}

    if rare_groups:
        rare_keys   = list(rare_groups.keys())
        rare_counts = {k_: len(v) for k_, v in rare_groups.items()}
        print(f"\n[Stratified {k}-Fold] Rare classes (kept in train only): "
              f"{rare_keys} counts={rare_counts}")

    # Shuffle within each normal group
    for key in normal_groups:
        random.shuffle(normal_groups[key])

    # Split each normal group into k roughly equal folds
    group_folds: Dict[tuple, List[List[dict]]] = {}
    for key, items in normal_groups.items():
        fold_size = len(items) / k
        group_folds[key] = [
            items[round(i * fold_size): round((i+1) * fold_size)]
            for i in range(k)
        ]

    # Collect all rare samples — always go to train, never to test
    rare_samples = [item for items in rare_groups.values() for item in items]

    # Assemble final folds
    folds = []
    for fold_idx in range(k):
        test  = []
        train = list(rare_samples)   # rare samples always in train

        for key in normal_groups:
            test  += group_folds[key][fold_idx]
            train += [item
                      for i, fold in enumerate(group_folds[key])
                      if i != fold_idx
                      for item in fold]
        folds.append((train, test))

    # Print fold summary
    print(f"\n[Stratified {k}-Fold Split]")
    for i, (train, test) in enumerate(folds):
        pos_test = sum(1 for d in test if get_label(d)[0])
        print(f"  Fold {i+1}: train={len(train)}, test={len(test)} "
              f"(test pos={pos_test}, neg={len(test)-pos_test})")

    return folds


# ─────────────────────────────────────────────────────────────────────────────
# Few-shot example selector (stratified, from train pool)
# ─────────────────────────────────────────────────────────────────────────────

def select_fewshot_examples(
    pool: List[dict],
    n_shots: int = 4,
    seed: int = 42,
) -> List[dict]:
    """
    Select stratified few-shot examples from the train pool.
    Picks examples round-robin across (has, event_type) groups
    to maximise diversity and avoid label bias.
    """
    random.seed(seed)

    groups: Dict[tuple, List[dict]] = defaultdict(list)
    for datum in pool:
        label = get_label(datum)
        groups[label].append(datum)

    # Shuffle within groups for variety across folds
    for key in groups:
        random.shuffle(groups[key])

    selected = []
    keys     = list(groups.keys())
    used     = defaultdict(int)
    idx      = 0

    while len(selected) < n_shots:
        key = keys[idx % len(keys)]
        if used[key] < len(groups[key]):
            selected.append(groups[key][used[key]])
            used[key] += 1
        idx += 1
        if all(used[k] >= len(groups[k]) for k in keys):
            break

    return selected


# ─────────────────────────────────────────────────────────────────────────────
# Format few-shot examples into prompt text matching your existing .txt format
# ─────────────────────────────────────────────────────────────────────────────

def format_fewshot_prompt(
    examples: List[dict],
    n_examples_shown: int = None,
) -> str:
    """
    Formats selected examples into the same style as fewshot-weather-extreme.txt.
    Output is used as the `fewshot` argument in talk_to_llm().
    """
    n = n_examples_shown or len(examples)
    lines = ["You are a severe weather analyst.\n"]

    for i, datum in enumerate(examples[:n], 1):
        has, etype = get_label(datum)
        answer     = "Yes," if has else "No,"
        event_line = etype if has else "NA"

        lines.append(f"Example {i}:")
        lines.append(f"Observations: {datum['observation'].strip()}")
        lines.append(f"Question: {datum['question'].strip()}")
        lines.append(f"Answer: {answer}")
        lines.append(f"EventType: {event_line}")
        lines.append("")

    lines += [
        "Output format:",
        "Line 1: Yes or No",
        "Line 2: EventType: Hail or Thunderstorm Wind or Flash Flood or "
        "Tornado or Lightning or Flood or Funnel Cloud or NA",
        "",
        "Provide a direct and concise answer.",
        "Do not repeat the instructions or restate the output format.",
        "Do not list multiple extreme weather events.",
        "Follow the required output format exactly.",
        "",
    ]

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate scores across folds
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_fold_scores(all_fold_scores: List[dict]) -> dict:
    """
    Average scores across k folds and compute standard deviation.
    Reports mean ± std for every metric.
    """
    import statistics

    keys = [k for k in all_fold_scores[0] if isinstance(all_fold_scores[0][k], (int, float))]
    aggregated = {}

    for key in keys:
        values = [s[key] for s in all_fold_scores]
        mean   = sum(values) / len(values)
        std    = statistics.stdev(values) if len(values) > 1 else 0.0
        aggregated[key]          = round(mean, 4)
        aggregated[f"{key}_std"] = round(std, 4)

    return aggregated


# ─────────────────────────────────────────────────────────────────────────────
# Main 5-Fold CV runner
# ─────────────────────────────────────────────────────────────────────────────

def run_kfold_extreme(
    jsonl_path: str,
    model: str,
    prompting_mode: str,        # "fewshot" or "zeroshot"
    format_dbase: str,          # "sentence" or "jsonl"
    concurrency: int = 5,
    k: int = 5,
    n_shots: int = 4,
    seed: int = 42,
    log_dir: str = "results_log",
) -> dict:
    """
    Full 5-fold cross-validation pipeline for WeatherExtremeExperiment.

    For zero-shot:  runs once on all data (no split needed)
    For few-shot:   runs k folds, each with different train/test split
                    and fresh few-shot examples selected from train pool

    Returns aggregated metrics (mean ± std across folds).
    """
    log = get_logging(
        name=f"kfold_{model}_{prompting_mode}",
        log_dir=log_dir,
        log_prefix=f"kfold_{model}_{prompting_mode}",
    )

    all_data = load_jsonl(jsonl_path)
    pos, neg = print_class_distribution(all_data, label="Full Dataset")

    # ── Zero-shot: single run on all data ────────────────────────────────────
    if prompting_mode == "zeroshot":
        log.info("[Zero-shot] Running on full dataset — no split needed.")

        # Write a zero-shot prompt file (no examples, just instructions)
        zeroshot_prompt = (
            "You are a severe weather analyst.\n\n"
            "Output format:\n"
            "Line 1: Yes or No\n"
            "Line 2: EventType: Hail or Thunderstorm Wind or Flash Flood or "
            "Tornado or Lightning or Flood or Funnel Cloud or NA\n\n"
            "Provide a direct and concise answer.\n"
            "Do not repeat the instructions or restate the output format.\n"
            "Do not list multiple extreme weather events.\n"
            "Follow the required output format exactly.\n"
        )

        # Temporarily write to file so WeatherExtremeExperiment can read it
        _write_prompt_file(zeroshot_prompt, model, "zeroshot")

        exp = WeatherExtremeExperiment(
            data_list=all_data,
            log_prefix=log,
            prompting_mode="zeroshot",
            model=model,
            concurrency=concurrency,
            task="weather-extreme",
            format_dbase=format_dbase,
        )
        exp.run(logging=True)
        final_scores = exp.metrics.get(subject="weather-extreme")
        log.info("\n=== Zero-shot Final Scores ===")
        for k_, v in final_scores.items():
            log.info(f"  {k_}: {v}")
        return final_scores

    # ── Few-shot: k-fold cross-validation ────────────────────────────────────
    log.info(f"[Few-shot {k}-Fold CV] model={model}, n_shots={n_shots}")
    folds = stratified_kfold(all_data, k=k, seed=seed)
    all_fold_scores = []

    for fold_idx, (train_pool, test_data) in enumerate(folds, 1):
        log.info(f"\n{'─'*40}")
        log.info(f"Fold {fold_idx}/{k}: train={len(train_pool)}, test={len(test_data)}")

        # Select fresh few-shot examples from this fold's train pool
        examples       = select_fewshot_examples(train_pool, n_shots=n_shots, seed=seed+fold_idx)
        fewshot_prompt = format_fewshot_prompt(examples)

        # Write prompt to file for this fold
        prompt_path = _write_prompt_file(fewshot_prompt, model, f"fewshot_fold{fold_idx}")
        log.info(f"  Few-shot examples: {[get_label(e) for e in examples]}")
        log.info(f"  Prompt written to: {prompt_path}")

        exp = WeatherExtremeExperiment(
            data_list=test_data,
            log_prefix=log,
            prompting_mode=f"fewshot_fold{fold_idx}",
            model=model,
            concurrency=concurrency,  # None → resolved from MODEL_DEFAULTS
            timeout=None,             # None → resolved from MODEL_DEFAULTS
            task="weather-extreme",
            format_dbase=format_dbase,
        )
        exp.run(logging=True)
        fold_scores = exp.metrics.get(subject="weather-extreme")

        # Warn if this fold processed fewer samples than expected
        n_expected  = len(test_data)
        n_processed = len(exp.results)
        if n_processed < n_expected:
            log.info(f"  [WARNING] Fold {fold_idx}: only {n_processed}/"
                     f"{n_expected} samples processed — timeouts likely")

        log.info(f"  Fold {fold_idx} scores: "
                 f"F1={fold_scores.get('f1', 0):.4f}, "
                 f"Acc={fold_scores.get('accuracy', 0):.4f}, "
                 f"Prec={fold_scores.get('precision', 0):.4f}, "
                 f"Rec={fold_scores.get('recall', 0):.4f}")

        all_fold_scores.append(fold_scores)

    # ── Aggregate across folds ────────────────────────────────────────────────
    aggregated = aggregate_fold_scores(all_fold_scores)

    log.info(f"\n{'='*50}")
    log.info(f"[{k}-Fold CV Final] model={model}, prompting={prompting_mode}")
    for metric in ["accuracy", "precision", "recall", "f1",
                   "type_accuracy", "strict", "lenient"]:
        mean = aggregated.get(metric, 0)
        std  = aggregated.get(f"{metric}_std", 0)
        log.info(f"  {metric:20s}: {mean:.4f} ± {std:.4f}")

    return aggregated


# ─────────────────────────────────────────────────────────────────────────────
# Helper: write prompt to .txt file (WeatherPrompting reads from file)
# ─────────────────────────────────────────────────────────────────────────────

def _write_prompt_file(prompt_text: str, model: str, suffix: str) -> str:
    """
    Writes the prompt to initial_prompts/weather-extreme/{suffix}-weather-extreme.txt
    so WeatherPrompting.get_initial_prompt() can read it normally.
    """
    import os
    prompt_dir = os.path.join(
        os.path.dirname(__file__), "..", "initial_prompts", "weather-extreme"
    )
    os.makedirs(prompt_dir, exist_ok=True)
    path = os.path.join(prompt_dir, f"{suffix}-weather-extreme.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(prompt_text)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl",    required=True, help="Path to your 100-sample jsonl")
    ap.add_argument("--model",    default="llama3")
    ap.add_argument("--mode",     choices=["zeroshot", "fewshot"], default="fewshot")
    ap.add_argument("--format",   choices=["sentence", "jsonl"],   default="sentence")
    ap.add_argument("--k",        type=int, default=5)
    ap.add_argument("--n-shots",  type=int, default=4)
    ap.add_argument("--log-dir",  default="results_log")
    args = ap.parse_args()

    results = run_kfold_extreme(
        jsonl_path=args.jsonl,
        model=args.model,
        prompting_mode=args.mode,
        format_dbase=args.format,
        k=args.k,
        n_shots=args.n_shots,
        log_dir=args.log_dir,
    )

    print("\n=== Final Aggregated Results ===")
    for metric in ["accuracy", "precision", "recall", "f1", "type_accuracy", "strict", "lenient"]:
        mean = results.get(metric, 0)
        std  = results.get(f"{metric}_std", 0)
        print(f"  {metric:20s}: {mean:.4f} ± {std:.4f}")