# experiments/baselines/extract_lstm_dataset.py
"""
Inspect and optionally export an LSTM dataset to .npz.
Supports both weather and energy_forecast tasks.

Usage:
    # Inspect weather dataset
    python -m experiments.baselines.extract_lstm_dataset \
        --task weather \
        --data data/weather/10_test_data.jsonl \
        --seq-len 6 --hmax 10

    # Inspect energy dataset
    python -m experiments.baselines.extract_lstm_dataset \
        --task energy_forecast \
        --data data/energy/sample_energy_forecast.json \
        --seq-len 48 --hmax 48

    # Export to .npz (add --train-data to fit scalers first)
    python -m experiments.baselines.extract_lstm_dataset \
        --task weather \
        --data data/weather/test.jsonl \
        --train-data data/weather/train.jsonl \
        --out data/weather/test_lstm.npz
"""
from __future__ import annotations

import argparse
import numpy as np

from experiments.baselines.features import (
    build_dataset, fit_scalers,
    lstm_collate_fn,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task",       required=True,
                    choices=["weather", "energy_forecast"],
                    help="Task name")
    ap.add_argument("--data",       required=True,
                    help="Path to dataset (JSONL or JSON array)")
    ap.add_argument("--train-data", default="",
                    help="Path to train data for fitting scalers (optional)")
    ap.add_argument("--seq-len",    type=int, default=None,
                    help="Input sequence length")
    ap.add_argument("--hmax",       type=int, default=None,
                    help="Max output horizon")
    ap.add_argument("--out",        default="",
                    help="Optional output .npz path")
    args = ap.parse_args()

    # Resolve task defaults
    _defaults = {"weather": (6, 10), "energy_forecast": (48, 48)}
    default_seq, default_hmax = _defaults[args.task]
    seq_len = args.seq_len or default_seq
    H_max   = args.hmax    or default_hmax

    # Fit scalers from train data if provided
    x_scaler = y_scaler = None
    scale = False
    if args.train_data:
        print(f"Fitting scalers from {args.train_data}...")
        x_scaler, y_scaler = fit_scalers(
            task=args.task, path_or_data=args.train_data,
            seq_len=seq_len, H_max=H_max,
        )
        scale = True
        print(f"  x_scaler shape: {x_scaler.mean.shape}")
        print(f"  y_scaler shape: {y_scaler.mean.shape}")

    # Build dataset
    ds = build_dataset(
        task=args.task, path_or_data=args.data,
        seq_len=seq_len, H_max=H_max,
        x_scaler=x_scaler, y_scaler=y_scaler,
        scale_x=scale, scale_y=scale,
    )

    print(f"\nTask:          {args.task}")
    print(f"Total samples: {len(ds)}")

    # Print sample 0
    s = ds[0]
    print(f"\n=== SAMPLE 0 ===")
    print(f"  task:    {s.task}")
    print(f"  x shape: {tuple(s.x.shape)}  x_len={s.x_len}")
    print(f"  y shape: {tuple(s.y.shape)}  H={s.H}")
    print(f"  x_mask sum: {float(s.x_mask.sum())}")
    print(f"  y_mask sum: {float(s.y_mask.sum())}")
    print(f"  x_times: {s.x_times[:3]}...")
    print(f"  y_times: {s.y_times[:3]}...")
    print(f"  first x row: {s.x[0].tolist()}")
    print(f"  first y row: {s.y[0].tolist()}")
    print(f"  meta keys:   {list(s.meta.keys())}")

    # Optionally export to .npz
    if args.out:
        X     = np.stack([ds[i].x.numpy()      for i in range(len(ds))])
        Xmask = np.stack([ds[i].x_mask.numpy() for i in range(len(ds))])
        Xlen  = np.array([ds[i].x_len          for i in range(len(ds))], dtype=np.int64)
        Y     = np.stack([ds[i].y.numpy()      for i in range(len(ds))])
        Ymask = np.stack([ds[i].y_mask.numpy() for i in range(len(ds))])
        H     = np.array([ds[i].H              for i in range(len(ds))], dtype=np.int64)

        np.savez_compressed(args.out,
                            X=X, Xmask=Xmask, Xlen=Xlen,
                            Y=Y, Ymask=Ymask, H=H)
        print(f"\nSaved: {args.out}")
        print(f"  X:     {X.shape}")
        print(f"  Y:     {Y.shape}")


if __name__ == "__main__":
    main()