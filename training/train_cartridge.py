#!/usr/bin/env python3
# Copyright 2026 Axobrier Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import os
import sys
import json
import math
import hashlib
import argparse
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add repo root to import tools.export_cartridge
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.export_cartridge import export_cartridge, inspect_cartridge

# ═══════════════════════════════════════════════════════════════════
#  DETERMINISTIC TEXT-TO-STATE ENCODER (D = 256)
# ═══════════════════════════════════════════════════════════════════

def text_to_embedding(text: str, embed_dim: int = 256) -> torch.Tensor:
    """
    Produces a dense, L2-normalized D-dimensional state embedding from text
    using word & character n-gram feature hashing with IDF weighting.
    This simulates the pooled state vector coming out of the base transformer.
    """
    words = text.lower().strip().split()
    vec = torch.zeros(embed_dim, dtype=torch.float32)

    if not words:
        vec[0] = 1.0
        return vec

    # Word unigrams + bigrams + character trigrams
    tokens = list(words)
    for i in range(len(words) - 1):
        tokens.append(f"{words[i]}_{words[i+1]}")
    for w in words:
        if len(w) >= 3:
            for k in range(len(w) - 2):
                tokens.append(w[k:k+3])

    for tok in tokens:
        # MD5 hashing for stable dimension mapping across platforms
        h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
        dim_idx = h % embed_dim
        sign = 1.0 if ((h >> 16) & 1) == 0 else -1.0
        weight = 1.0 / math.sqrt(float(len(tok) + 1))
        vec[dim_idx] += sign * weight

    # L2 Normalization (matches Axobrier mean-pool + L2-norm)
    norm = torch.norm(vec, p=2)
    if norm > 1e-8:
        vec = vec / norm
    else:
        vec[0] = 1.0

    return vec


def load_dataset(jsonl_path: str, label_to_idx: Optional[Dict[str, int]] = None, embed_dim: int = 256):
    """
    Loads JSONL dataset and encodes texts to embeddings.
    """
    samples = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    # Determine unique labels if not provided
    if label_to_idx is None:
        unique_labels = sorted(list(set(s["label"] for s in samples)))
        label_to_idx = {lbl: idx for idx, lbl in enumerate(unique_labels)}

    X_list = []
    y_list = []

    for s in samples:
        emb = text_to_embedding(s["text"], embed_dim)
        lbl_idx = label_to_idx[s["label"]]
        X_list.append(emb)
        y_list.append(lbl_idx)

    X = torch.stack(X_list, dim=0)
    y = torch.tensor(y_list, dtype=torch.long)
    return X, y, label_to_idx


# ═══════════════════════════════════════════════════════════════════
#  CALIBRATION METRICS & LOSS
# ═══════════════════════════════════════════════════════════════════

def brier_score_loss(logits: torch.Tensor, targets: torch.Tensor, temp: float = 1.0, bias: float = 0.0) -> torch.Tensor:
    """
    Computes multi-class Brier Score Loss: Mean(||Softmax(z * temp + bias) - y_one_hot||^2).
    """
    scaled_logits = logits * temp + bias
    probs = F.softmax(scaled_logits, dim=-1)
    num_classes = logits.size(-1)
    targets_one_hot = F.one_hot(targets, num_classes=num_classes).float()
    loss = torch.mean((probs - targets_one_hot) ** 2)
    return loss


def compute_ece(probs: torch.Tensor, targets: torch.Tensor, n_bins: int = 10) -> float:
    """
    Computes Expected Calibration Error (ECE) across confidence bins.
    """
    confidences, predictions = torch.max(probs, dim=-1)
    accuracies = predictions.eq(targets)

    ece = 0.0
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = confidences.gt(bin_lower) * confidences.le(bin_upper)
        prop_in_bin = in_bin.float().mean().item()

        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean().item()
            avg_confidence_in_bin = confidences[in_bin].mean().item()
            ece += abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece


# ═══════════════════════════════════════════════════════════════════
#  PLATT CALIBRATION ROUTINE
# ═══════════════════════════════════════════════════════════════════

def fit_platt_calibration(
    logits: torch.Tensor,
    targets: torch.Tensor,
    init_temp: float = 1.0,
    init_bias: float = 0.0,
    max_iter: int = 100
) -> Tuple[float, float]:
    """
    Optimizes scalar temperature (A) and bias (B) via L-BFGS to minimize
    the validation Brier Score.
    """
    # Learnable parameters (temperature > 0 via log-parameterization)
    log_temp = torch.tensor([math.log(max(init_temp, 0.01))], requires_grad=True)
    bias = torch.tensor([init_bias], requires_grad=True)

    optimizer = torch.optim.LBFGS([log_temp, bias], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")
    num_classes = logits.size(-1)
    targets_one_hot = F.one_hot(targets, num_classes=num_classes).float()

    def closure():
        optimizer.zero_grad()
        t = torch.exp(log_temp)
        scaled_logits = logits * t + bias
        probs = F.softmax(scaled_logits, dim=-1)
        loss = torch.mean((probs - targets_one_hot) ** 2)
        loss.backward()
        return loss

    optimizer.step(closure)

    final_temp = torch.exp(log_temp).item()
    final_bias = bias.item()
    return final_temp, final_bias


# ═══════════════════════════════════════════════════════════════════
#  MAIN TRAINING ROUTINE
# ═══════════════════════════════════════════════════════════════════

def train_cartridge(
    train_path: str,
    val_path: str,
    output_axb: str,
    embed_dim: int = 256,
    epochs: int = 300,
    lr: float = 0.05,
    weight_decay: float = 1e-4
) -> None:
    print("===========================================================")
    print(f"  Axobrier Cartridge Trainer -> {os.path.basename(output_axb)}")
    print("===========================================================\n")

    # 1. Load Datasets
    print(f"[1/5] Loading training and validation data...")
    X_train, y_train, label_to_idx = load_dataset(train_path, embed_dim=embed_dim)
    X_val, y_val, _ = load_dataset(val_path, label_to_idx=label_to_idx, embed_dim=embed_dim)

    idx_to_label = {idx: lbl for lbl, idx in label_to_idx.items()}
    labels_list = [idx_to_label[i] for i in range(len(label_to_idx))]
    num_choices = len(labels_list)

    print(f"       Train samples: {len(X_train)} | Val samples: {len(X_val)}")
    print(f"       Choices (C={num_choices}): {labels_list}\n")

    # 2. Initialize Model: Choice Projection Matrix W [D x C]
    # We parameterize the choice head directly as a weight matrix [D, C]
    torch.manual_seed(42)
    model = nn.Linear(embed_dim, num_choices, bias=False)
    # Initialize with small random weights
    nn.init.orthogonal_(model.weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 3. Training Loop with Brier Score Loss
    print(f"[2/5] Training choice routing head ({epochs} epochs, AdamW lr={lr})...")
    best_val_brier = float("inf")
    best_weights = None

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        train_logits = model(X_train)
        loss = brier_score_loss(train_logits, y_train)
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                val_logits = model(X_val)
                val_loss = brier_score_loss(val_logits, y_val).item()
                val_preds = torch.argmax(val_logits, dim=-1)
                val_acc = val_preds.eq(y_val).float().mean().item() * 100.0

            print(f"       Epoch {epoch:3d}/{epochs:3d} | Train Brier: {loss.item():.6f} | "
                  f"Val Brier: {val_loss:.6f} | Val Acc: {val_acc:.1f}%")

            if val_loss < best_val_brier:
                best_val_brier = val_loss
                # model.weight is [C, D], we need [D, C] for column-major matrix
                best_weights = model.weight.detach().t().clone()

    print("       Training complete.\n")

    # 4. Post-Training Platt Calibration on Validation Split
    print("[3/5] Performing post-training Platt calibration on validation split...")
    with torch.no_grad():
        final_val_logits = X_val @ best_weights
        pre_probs = F.softmax(final_val_logits, dim=-1)
        pre_brier = brier_score_loss(final_val_logits, y_val).item()
        pre_ece = compute_ece(pre_probs, y_val)
        pre_acc = pre_probs.argmax(dim=-1).eq(y_val).float().mean().item() * 100.0

    platt_temp, platt_bias = fit_platt_calibration(final_val_logits, y_val)

    with torch.no_grad():
        post_logits = final_val_logits * platt_temp + platt_bias
        post_probs = F.softmax(post_logits, dim=-1)
        post_brier = brier_score_loss(final_val_logits, y_val, temp=platt_temp, bias=platt_bias).item()
        post_ece = compute_ece(post_probs, y_val)
        post_acc = post_probs.argmax(dim=-1).eq(y_val).float().mean().item() * 100.0
        avg_confidence = post_probs.max(dim=-1).values.mean().item()

    print(f"       Pre-calibration:  Brier={pre_brier:.6f}, ECE={pre_ece:.4f}, Acc={pre_acc:.1f}%")
    print(f"       Fitted Platt:     Temperature={platt_temp:.4f}, Bias={platt_bias:.4f}")
    print(f"       Post-calibration: Brier={post_brier:.6f}, ECE={post_ece:.4f}, Acc={post_acc:.1f}%")
    print(f"       Mean Top-1 Conf:  {avg_confidence:.4f}\n")

    # 5. Export to .axb Binary Cartridge
    print(f"[4/5] Exporting .axb cartridge to {output_axb}...")
    # Convert weights to numpy list/array of shape [D, C]
    weights_np = best_weights.cpu().numpy()
    # Note: Axobrier's CUDA kernel computes (dot / platt_temp + platt_bias).
    # Since Platt scaling here fits a multiplier A in (dot * A + B),
    # the hardware temperature divisor is T = 1 / A.
    hardware_temp = 1.0 / max(platt_temp, 1e-6)
    bytes_written = export_cartridge(
        filepath=output_axb,
        weights=weights_np,
        labels=labels_list,
        platt_temperature=hardware_temp,
        platt_bias=platt_bias,
        embed_dim=embed_dim,
        num_choices=num_choices
    )
    print(f"       OK: {bytes_written} bytes written (Hardware Temp: {hardware_temp:.4f}).\n")

    # 6. Verify with Inspector
    print("[5/5] Validating exported binary header...")
    info = inspect_cartridge(output_axb)
    print(f"       Magic:      {info['magic']}")
    print(f"       Version:    {info['version']}")
    print(f"       Embed Dim:  {info['embed_dim']}")
    print(f"       Choices:    {info['num_choices']}")
    print(f"       Platt Temp: {info['platt_temperature']:.4f}")
    print(f"       Platt Bias: {info['platt_bias']:.4f}")
    print("===========================================================")
    print("  Cartridge build & export SUCCESSFUL")
    print("===========================================================\n")


def main():
    parser = argparse.ArgumentParser(description="Axobrier Choice Cartridge Trainer")
    parser.add_argument("--train", required=True, help="Path to training .jsonl")
    parser.add_argument("--val", required=True, help="Path to validation .jsonl")
    parser.add_argument("--output", "-o", required=True, help="Output .axb cartridge path")
    parser.add_argument("--embed-dim", type=int, default=256, help="Embedding dimension (default: 256)")
    parser.add_argument("--epochs", type=int, default=250, help="Training epochs (default: 250)")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate (default: 0.05)")
    args = parser.parse_args()

    train_cartridge(
        train_path=args.train,
        val_path=args.val,
        output_axb=args.output,
        embed_dim=args.embed_dim,
        epochs=args.epochs,
        lr=args.lr
    )


if __name__ == "__main__":
    main()
