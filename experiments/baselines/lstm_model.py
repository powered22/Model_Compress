# experiments/baselines/lstm_model.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence


@dataclass
class LSTMModelConfig:
    input_dim: int            # D
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.1      # applied between LSTM layers (if num_layers > 1)
    bidirectional: bool = False

    h_max: int = 10           # fixed output horizon (Option B)
    output_dim: Optional[int] = None  # default: same as input_dim (predict same features)

    head: str = "mlp"         # "mlp" or "linear"
    mlp_dim: int = 256        # used if head == "mlp"


class LSTMBaselineOptionB(nn.Module):
    """
    Baseline LSTM for Option B:
      - Input:  x [B, seq_len, D], x_len [B]
      - Output: y_hat [B, H_max, Dy]  (Dy defaults to D)

    Design:
      - Encoder LSTM consumes the history sequence.
      - We take the last-layer final hidden state h_n (or concat if bidirectional),
        then use a projection head to produce H_max * Dy outputs.
      - Reshape to [B, H_max, Dy].

    Notes:
      - This is a simple baseline (not full seq2seq decoding).
      - Works well as a first LSTM baseline and matches your dataset format.
    """

    def __init__(self, cfg: LSTMModelConfig):
        super().__init__()
        self.cfg = cfg
        self.input_dim = cfg.input_dim
        self.output_dim = cfg.output_dim if cfg.output_dim is not None else cfg.input_dim
        self.h_max = cfg.h_max

        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            dropout=(cfg.dropout if cfg.num_layers > 1 else 0.0),
            bidirectional=cfg.bidirectional,
            batch_first=True,
        )

        dir_mult = 2 if cfg.bidirectional else 1
        enc_dim = cfg.hidden_dim * dir_mult
        out_flat = self.h_max * self.output_dim

        if cfg.head == "linear":
            self.head = nn.Linear(enc_dim, out_flat)
        elif cfg.head == "mlp":
            self.head = nn.Sequential(
                nn.Linear(enc_dim, cfg.mlp_dim),
                nn.ReLU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.mlp_dim, out_flat),
            )
        else:
            raise ValueError(f"Unknown head={cfg.head}. Use 'linear' or 'mlp'.")

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        batch keys expected (from lstm_collate_fn):
          - x: [B, seq_len, D]
          - x_len: [B] (<= seq_len)
        """
        x = batch["x"]
        x_len = batch["x_len"]

        # pack padded sequence so LSTM ignores padded timesteps
        # IMPORTANT: enforce_sorted=False because your batch is not necessarily sorted by length
        packed = pack_padded_sequence(x, x_len.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)

        # h_n: [num_layers * num_directions, B, hidden_dim]
        # take last layer
        if self.cfg.bidirectional:
            # last layer has two directions at the end: [-2] forward, [-1] backward
            h_last = torch.cat([h_n[-2], h_n[-1]], dim=-1)  # [B, hidden_dim*2]
        else:
            h_last = h_n[-1]  # [B, hidden_dim]

        y_flat = self.head(h_last)  # [B, H_max * Dy]
        y_hat = y_flat.view(x.size(0), self.h_max, self.output_dim)  # [B, H_max, Dy]
        return y_hat


def masked_mae_loss(
    y_hat: torch.Tensor,
    y_true: torch.Tensor,
    y_mask: torch.Tensor,
    reduction: str = "mean",
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    y_hat:  [B, H_max, D]
    y_true: [B, H_max, D]
    y_mask: [B, H_max]  (1.0 valid step, 0.0 padded/ignored)

    Returns MAE computed only over valid horizon steps.
    """
    # expand mask to match feature dim
    mask = y_mask.unsqueeze(-1)  # [B, H, 1]
    abs_err = (y_hat - y_true).abs() * mask

    if reduction == "none":
        return abs_err

    denom = mask.sum() * y_true.size(-1)  # (#valid steps) * D
    denom = torch.clamp(denom, min=eps)

    return abs_err.sum() / denom


def masked_mse_loss(
    y_hat: torch.Tensor,
    y_true: torch.Tensor,
    y_mask: torch.Tensor,
    reduction: str = "mean",
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Same idea as masked_mae_loss but for MSE.
    """
    mask = y_mask.unsqueeze(-1)
    sq_err = (y_hat - y_true).pow(2) * mask

    if reduction == "none":
        return sq_err

    denom = mask.sum() * y_true.size(-1)
    denom = torch.clamp(denom, min=eps)

    return sq_err.sum() / denom


@torch.no_grad()
def predict(batch: Dict[str, torch.Tensor], model: nn.Module) -> torch.Tensor:
    """
    Convenience function for inference. Returns y_hat [B, H_max, D].
    """
    model.eval()
    return model(batch)


def train_step(
    batch: Dict[str, torch.Tensor],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_type: str = "mae",  # "mae" or "mse"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    One optimization step.
    Expects:
      batch["x"], batch["x_len"], batch["y"], batch["y_mask"]
    """
    model.train()
    optimizer.zero_grad(set_to_none=True)

    y_hat = model(batch)              # [B, H_max, D]
    y = batch["y"]                    # [B, H_max, D]
    y_mask = batch["y_mask"]          # [B, H_max]

    if loss_type == "mae":
        loss = masked_mae_loss(y_hat, y, y_mask)
    elif loss_type == "mse":
        loss = masked_mse_loss(y_hat, y, y_mask)
    else:
        raise ValueError("loss_type must be 'mae' or 'mse'")

    loss.backward()
    optimizer.step()

    return loss.detach(), y_hat.detach()

