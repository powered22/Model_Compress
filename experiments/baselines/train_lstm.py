# experiments/baselines/train_lstm.py
"""
Task-aware LSTM training script.

Supports:
    --task weather          : multi-field weather forecasting
    --task energy_forecast  : univariate energy load forecasting

Usage examples:
    # Weather
    python -m experiments.baselines.train_lstm \
        --task weather \
        --train-data data/weather/train.jsonl \
        --val-data   data/weather/val.jsonl \
        --seq-len 6 --hmax 10 --hidden-dim 128 --epochs 20

    # Energy forecast
    python -m experiments.baselines.train_lstm \
        --task energy_forecast \
        --train-data data/energy/train.json \
        --val-data   data/energy/val.json \
        --seq-len 48 --hmax 48 --hidden-dim 256 --epochs 30
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime

import torch
from torch.utils.data import DataLoader

from experiments.baselines.features import (
    append_info, append_log,
    build_dataset, fit_scalers,
    DEFAULT_WEATHER_FEATURES, DEFAULT_ENERGY_FEATURES,
    lstm_collate_fn,
)
from experiments.baselines.lstm_model import (
    LSTMModelConfig, LSTMBaselineOptionB,
    masked_mae_loss, masked_mse_loss, train_step,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _input_dim(task: str, feature_order) -> int:
    if task == "weather":
        return len(feature_order or DEFAULT_WEATHER_FEATURES)
    elif task == "energy_forecast":
        return 1   # univariate
    raise ValueError(f"Unknown task: {task}")


def _default_seq_hmax(task: str):
    """Default seq_len and H_max per task."""
    if task == "weather":
        return 6, 10
    elif task == "energy_forecast":
        return 48, 48
    raise ValueError(f"Unknown task: {task}")


def init_log(args) -> str:
    os.makedirs("results_log", exist_ok=True)
    fname = (f"lstm_{args.task}_h{args.hmax}_seq{args.seq_len}"
             f"_hidden{args.hidden_dim}_layers{args.num_layers}"
             f"_{args.loss}.log")
    path = os.path.join("results_log", fname)
    with open(path, "w") as f:
        f.write(f"# LSTM training — task={args.task}\n")
        f.write(f"# time: {datetime.now()}\n")
        f.write(f"# args: {vars(args)}\n")
        f.write("epoch,split,loss_type,loss\n")
    return path


def save_checkpoint(model, cfg, path: str, epoch: int, val_loss: float):
    torch.save({
        "epoch":            epoch,
        "val_loss":         val_loss,
        "model_config":     cfg.__dict__,
        "model_state_dict": model.state_dict(),
    }, path)
    print(f"[Checkpoint] Saved → {path}  (val_loss={val_loss:.6f})")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()

    # Task selection — the key new argument
    ap.add_argument("--task", required=True,
                    choices=["weather", "energy_forecast"],
                    help="Task to train on")

    # Data
    ap.add_argument("--train-data", required=True,
                    help="Path to train data (JSONL for weather, JSON for energy)")
    ap.add_argument("--val-data",   default="",
                    help="Path to val data (optional but recommended)")

    # Sequence / horizon — defaults depend on task
    ap.add_argument("--seq-len",    type=int, default=None,
                    help="Input sequence length (default: 6 for weather, 48 for energy)")
    ap.add_argument("--hmax",       type=int, default=None,
                    help="Max output horizon (default: 10 for weather, 48 for energy)")

    # Model architecture
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--num-layers", type=int, default=2)

    # Training
    ap.add_argument("--batch-size", type=int, default=None,
                    help="Batch size (default: 32 CPU / 256 GPU)")
    ap.add_argument("--epochs",     type=int, default=20)
    ap.add_argument("--lr",         type=float, default=1e-3)
    ap.add_argument("--loss",       choices=["mae", "mse"], default="mae")
    ap.add_argument("--save-dir",   default="results_log")

    args = ap.parse_args()

    # Fill defaults that depend on task
    default_seq, default_hmax = _default_seq_hmax(args.task)
    if args.seq_len is None: args.seq_len = default_seq
    if args.hmax    is None: args.hmax    = default_hmax

    # Device
    _on_gpu = torch.cuda.is_available()
    if _on_gpu:
        device   = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem  = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[CUDA] {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        device = "cpu"
        print("[CPU] No GPU detected")

    if args.batch_size is None:
        args.batch_size = 256 if _on_gpu else 32

    log_path = init_log(args)
    os.environ["LOG_PATH"] = log_path
    append_info(log_path, f"task={args.task} device={device}")

    # ── Scalers ──────────────────────────────────────────────────────────────
    print(f"[{args.task}] Fitting scalers from {args.train_data}...")
    x_scaler, y_scaler = fit_scalers(
        task=args.task, path_or_data=args.train_data,
        seq_len=args.seq_len, H_max=args.hmax,
    )
    append_info(log_path, f"x_scaler mean shape: {x_scaler.mean.shape}")
    append_info(log_path, f"y_scaler mean shape: {y_scaler.mean.shape}")

    # ── Datasets ─────────────────────────────────────────────────────────────
    train_ds = build_dataset(
        task=args.task, path_or_data=args.train_data,
        seq_len=args.seq_len, H_max=args.hmax,
        x_scaler=x_scaler, y_scaler=y_scaler,
        scale_x=True, scale_y=True,
    )
    append_info(log_path, f"Train samples: {len(train_ds)}")
    print(f"[{args.task}] Train: {len(train_ds)} samples")

    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=lstm_collate_fn,
        pin_memory=_on_gpu,
        num_workers=4 if _on_gpu else 0,
    )

    val_dl = None
    if args.val_data:
        val_ds = build_dataset(
            task=args.task, path_or_data=args.val_data,
            seq_len=args.seq_len, H_max=args.hmax,
            x_scaler=x_scaler, y_scaler=y_scaler,
            scale_x=True, scale_y=True,
        )
        val_dl = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            collate_fn=lstm_collate_fn,
            pin_memory=_on_gpu,
            num_workers=4 if _on_gpu else 0,
        )
        print(f"[{args.task}] Val: {len(val_ds)} samples")

    # ── Model ─────────────────────────────────────────────────────────────────
    D   = _input_dim(args.task, None)
    cfg = LSTMModelConfig(
        input_dim=D, output_dim=D,
        hidden_dim=args.hidden_dim, num_layers=args.num_layers,
        h_max=args.hmax, head="mlp",
    )
    model = LSTMBaselineOptionB(cfg).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr)

    os.makedirs(args.save_dir, exist_ok=True)
    best_path = os.path.join(
        args.save_dir,
        f"lstm_{args.task}_best"
        f"_h{args.hmax}_seq{args.seq_len}"
        f"_hidden{args.hidden_dim}_layers{args.num_layers}.pt"
    )
    best_val_loss = float("inf")

    # ── Training loop ─────────────────────────────────────────────────────────
    for ep in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in train_dl:
            batch  = {k: v.to(device) for k, v in batch.items()}
            loss, _ = train_step(batch, model, opt, loss_type=args.loss)
            losses.append(float(loss))

        train_loss = sum(losses) / max(1, len(losses))
        print(f"[ep {ep:3d}] train_{args.loss}={train_loss:.6f}")
        append_log(log_path, ep, "train", args.loss, train_loss)

        if val_dl is not None:
            model.eval()
            vlosses = []
            with torch.no_grad():
                for batch in val_dl:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    y_hat = model(batch)
                    vloss = (masked_mae_loss(y_hat, batch["y"], batch["y_mask"])
                             if args.loss == "mae" else
                             masked_mse_loss(y_hat, batch["y"], batch["y_mask"]))
                    vlosses.append(float(vloss))

            val_loss = sum(vlosses) / max(1, len(vlosses))
            print(f"[ep {ep:3d}] val_{args.loss}_scaled={val_loss:.6f}")
            append_log(log_path, ep, "val", args.loss, val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, cfg, best_path, ep, val_loss)

    # Save final checkpoint
    final_path = best_path.replace("_best_", "_final_")
    save_checkpoint(model, cfg, final_path, args.epochs,
                    best_val_loss if val_dl else float("nan"))
    print(f"\nDone. Best: {best_path}")


if __name__ == "__main__":
    main()