# split_datasets.py
"""
Splits all four datasets into train and val files.

Usage:
    python -m split_datasets

Outputs (written to same folder as input):
    data/weather/train.jsonl          val.jsonl
    data/weather_extreme/train.jsonl  val.jsonl
    data/energy/train.json            val.json
    data/energy_extreme/train.json    val.json

Split strategies:
    weather, energy_forecast  → temporal split (preserve time ordering)
    weather_extreme, energy   → stratified split (preserve class balance)
"""
from __future__ import annotations

import json
import os
import random
from collections import defaultdict
from typing import List, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Configuration — edit paths and ratios here
# ─────────────────────────────────────────────────────────────────────────────

DATASETS = [
    {
        "name":"weather",
        "path":"data/weather/100_test_data.jsonl",   # ← your 100-sample file
        "strategy": "temporal",
        "val_ratio": 0.2,
        # Works for both old and new format — obs_json always has timestamp keys
        "sort_key": lambda d: sorted(
            [k for k in d["obs_json"].keys() if k != "question"]
        )[0],
    },
    {
        "name":     "weather_extreme",
        "path":     "data/weather_extreme/159_dataset.jsonl",  # ← your 159-sample file
        "strategy": "stratified",
        "val_ratio": 0.2,
        # Works for both old (JSONL) and new (JSON object) format
        # ground_truth structure is identical in both
        "label_fn": lambda d: (
            bool(list(d["ground_truth"].values())[0]["has_extreme_weather"]),
            str(list(d["ground_truth"].values())[0]["event_type"]),
        ),
    },
    {
        "name":     "energy_forecast",
        "path":     "data/energy_task/energy_forecast_new.json",
        "strategy": "temporal",
        "val_ratio": 0.2,
        "sort_key": lambda d: d.get("context", {}).get("start_timestamp", ""),
    },
    {
        "name":     "energy_extreme",
        "path":     "data/energy_extreme/energy_extreme_new.json",
        "strategy": "stratified",
        "val_ratio": 0.2,
        "label_fn": lambda d: bool(d["ground_truth"]["has_extreme"]),
    },
]

SEED = 42


# ─────────────────────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────────────────────

def load(path: str) -> List[dict]:
    """
    Robustly loads four file formats:
      1. JSON array:              [ {...}, {...} ]
      2. True JSONL:              {...}\n{...}\n     (one object per line)
      3. Concatenated JSON objs:  {...}\n\n{...}\n   (pretty-printed, multi-line)
      4. Single JSON object:      { ... }            (wrapped in list)
    """
    with open(path, encoding="utf-8") as f:
        content = f.read().strip()

    if not content:
        raise ValueError(f"File is empty: {path}")

    first = content[0]

    # ── Format 1: JSON array ─────────────────────────────────────────────────
    if first == "[":
        return json.loads(content)

    # ── Formats 2, 3, 4: starts with { ───────────────────────────────────────
    if first == "{":
        decoder = json.JSONDecoder()
        results = []
        idx     = 0
        n       = len(content)

        while idx < n:
            # Skip whitespace between objects
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
                    f"JSON parse error in {path} near char {idx}: {e}\n"
                    f"Context: {content[max(0,idx-50):idx+50]!r}"
                ) from e

        return results

    raise ValueError(
        f"Unrecognised file format in {path}. "
        f"First character: {first!r}. Expected '[' or '{{'."
    )


def save(data: List[dict], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ext = os.path.splitext(path)[1]
    with open(path, "w", encoding="utf-8") as f:
        if ext == ".json":
            json.dump(data, f, ensure_ascii=False, indent=2)
        else:   # .jsonl
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  Saved {len(data):4d} samples → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Split strategies
# ─────────────────────────────────────────────────────────────────────────────

def temporal_split(
    data:      List[dict],
    sort_key,
    val_ratio: float = 0.2,
) -> Tuple[List[dict], List[dict]]:
    """
    Sort by timestamp and split chronologically.
    Earliest samples → train, latest samples → val.
    No shuffling — preserves temporal ordering.
    """
    sorted_data = sorted(data, key=sort_key)
    n_val       = max(1, int(len(sorted_data) * val_ratio))
    train       = sorted_data[:-n_val]
    val         = sorted_data[-n_val:]
    return train, val


def stratified_split(
    data:      List[dict],
    label_fn,
    val_ratio: float = 0.2,
    seed:      int   = 42,
) -> Tuple[List[dict], List[dict]]:
    """
    Group by class label, then split each group proportionally.
    Guarantees each class appears in both train and val.
    Shuffles within each group before splitting.
    """
    random.seed(seed)

    groups: dict = defaultdict(list)
    for datum in data:
        groups[label_fn(datum)].append(datum)

    train, val = [], []
    for label, items in sorted(groups.items(), key=lambda x: str(x[0])):
        random.shuffle(items)
        n_val_group = max(1, int(len(items) * val_ratio))

        # Edge case: if group has only 1 sample, put it in train
        if len(items) == 1:
            train += items
            print(f"    Class {label}: 1 sample → train only (too few to split)")
            continue

        val   += items[:n_val_group]
        train += items[n_val_group:]
        print(f"    Class {label}: {len(items)} total "
              f"→ train={len(items)-n_val_group}, val={n_val_group}")

    # Shuffle final train and val so classes are interleaved
    random.shuffle(train)
    random.shuffle(val)
    return train, val


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Dataset Splitting")
    print("=" * 60)

    for cfg in DATASETS:
        name     = cfg["name"]
        path     = cfg["path"]
        strategy = cfg["strategy"]
        val_r    = cfg["val_ratio"]

        print(f"\n[{name}]  strategy={strategy}  val_ratio={val_r}")
        print(f"  Loading: {path}")

        if not os.path.exists(path):
            print(f"  [SKIP] File not found: {path}")
            continue

        data = load(path)
        print(f"  Total samples: {len(data)}")

        # Determine output extension
        ext = ".json" if path.endswith(".json") else ".jsonl"

        # Determine output directory
        out_dir  = os.path.dirname(path)
        train_path = os.path.join(out_dir, f"train{ext}")
        val_path   = os.path.join(out_dir, f"val{ext}")

        # Run split
        if strategy == "temporal":
            train, val = temporal_split(data, cfg["sort_key"], val_r)

            # Print time range info
            sorted_data = sorted(data, key=cfg["sort_key"])
            n_val       = len(val)
            print(f"  Train period: {cfg['sort_key'](sorted_data[0])[:10]} "
                  f"→ {cfg['sort_key'](sorted_data[-(n_val+1)])[:10]}")
            print(f"  Val period:   {cfg['sort_key'](sorted_data[-n_val])[:10]} "
                  f"→ {cfg['sort_key'](sorted_data[-1])[:10]}")

        elif strategy == "stratified":
            train, val = stratified_split(data, cfg["label_fn"], val_r, SEED)

        print(f"  Train: {len(train)}  Val: {len(val)}")
        save(train, train_path)
        save(val,   val_path)

    print("\n" + "=" * 60)
    print("Done. Update DATASETS paths above if your files are elsewhere.")
    print("=" * 60)


if __name__ == "__main__":
    main()