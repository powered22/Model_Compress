"""
experiments/multiseed_eval.py
Unified stratified multi-seed split and aggregation for all tasks.

Every task uses stratified_multiseed_split with a task-specific
get_stratum_fn so that fewshot examples represent the data distribution.
Repeating with different seeds gives mean ± std for stable metric estimates.

Strata per task:
  energy forecast  → season (Winter / Spring / Summer / Autumn)
  energy extreme   → class  (True / False)
  extreme weather  → (has_extreme, event_type)
  weather forecast → forecast horizon (hours)
"""

import math
import random
from collections import defaultdict
from typing import Any, Callable, Dict, List, Tuple


def stratified_multiseed_split(
    data: List[dict],
    get_stratum_fn: Callable[[dict], Any],
    n_per_stratum: int,
    seed: int,
) -> Tuple[List[dict], List[dict]]:
    """
    Select n_per_stratum examples per stratum as the fewshot pool.
    Remaining samples form the test set.

    Strata with <= n_per_stratum total samples are skipped so every
    stratum always contributes at least one sample to the test set.

    Args:
        data:            full dataset
        get_stratum_fn:  returns a hashable stratum label for each datum
        n_per_stratum:   fewshot examples to draw per stratum
        seed:            controls within-stratum shuffling

    Returns:
        (fewshot_examples, test_data)
    """
    random.seed(seed)

    groups: Dict[Any, List[dict]] = defaultdict(list)
    for datum in data:
        groups[get_stratum_fn(datum)].append(datum)

    fewshot: List[dict] = []
    used_ids: set = set()

    for stratum, items in sorted(groups.items(), key=lambda x: str(x[0])):
        if len(items) <= n_per_stratum:
            continue
        shuffled = items[:]
        random.shuffle(shuffled)
        selected = shuffled[:n_per_stratum]
        fewshot.extend(selected)
        used_ids.update(id(s) for s in selected)

    test_data = [d for d in data if id(d) not in used_ids]

    print(f"[MultiSeed split | seed={seed}]  "
          f"fewshot={len(fewshot)}  test={len(test_data)}  "
          f"strata={len(groups)}")
    return fewshot, test_data


def aggregate_seeds(scores_per_seed: List[dict]) -> dict:
    """
    Aggregate per-seed score dicts into a single dict with mean and std.

    Numeric metrics:     key=mean value, key_std=std value
    Non-numeric metrics: key=last seed's value (e.g. counts, strings)

    Example input:  [{"f1": 0.80, "n": 92}, {"f1": 0.75, "n": 91}]
    Example output: {"f1": 0.775, "f1_std": 0.025, "n": 91}
    """
    if not scores_per_seed:
        return {}
    if len(scores_per_seed) == 1:
        return dict(scores_per_seed[0])

    all_keys: set = set()
    for s in scores_per_seed:
        all_keys.update(s.keys())

    result: dict = {}
    for key in sorted(all_keys):
        values = [
            s[key] for s in scores_per_seed
            if key in s
            and isinstance(s[key], (int, float))
            and s[key] is not None
        ]
        if len(values) == len(scores_per_seed):
            mean = sum(values) / len(values)
            variance = sum((v - mean) ** 2 for v in values) / len(values)
            result[key] = mean
            result[f"{key}_std"] = math.sqrt(variance)
        else:
            result[key] = scores_per_seed[-1].get(key)

    return result
