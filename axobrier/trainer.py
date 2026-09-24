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
import re
from typing import List, Dict, Tuple, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from axobrier.export import export_cartridge, inspect_cartridge

AXO_HASH_SEED_PRIME = 0x696c6f63636f7262


def extract_diagnostic_signal(text: str) -> str:
    """Isolate root-cause error diagnostics and high-signal frames from noisy stack traces."""
    raw_lines = [l.strip() for l in text.strip().splitlines()]
    lines = [l for l in raw_lines if l and not re.match(r"^[-=*#_~]{3,}$", l)]
    if len(lines) <= 4:
        return " ".join(lines) if lines else text

    signal_lines = []
    keywords = ("error", "exception", "fatal", "failed", "cannot", "could not", "assert", "not found", "missing", "denied", "panic")
    for l in lines:
        lower = l.lower()
        if any(k in lower for k in keywords):
            signal_lines.append(l)

    if signal_lines:
        if len(signal_lines) <= 3:
            return " ".join(signal_lines)
        return f"{signal_lines[0]} {signal_lines[-2]} {signal_lines[-1]}"

    return f"{lines[0]} {lines[-2]} {lines[-1]}"


def text_to_embedding(text: str, embed_dim: int = 256, distill: bool = True) -> torch.Tensor:
    """Project text string into an L2-normalized dense embedding vector."""
    if distill:
        text = extract_diagnostic_signal(text)
    words = text.lower().strip().split()
    vec = torch.zeros(embed_dim, dtype=torch.float32)

    if not words:
        vec[0] = 1.0
        return vec

    tokens = list(words)
    for i in range(len(words) - 1):
        tokens.append(f"{words[i]}_{words[i+1]}")
    for w in words:
        if len(w) >= 3:
            for k in range(len(w) - 2):
                tokens.append(w[k:k+3])

    for tok in tokens:
        h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
        dim_idx = h % embed_dim
        sign = 1.0 if ((h >> 16) & 1) == 0 else -1.0
        weight = 1.0 / math.sqrt(float(len(tok) + 1))
        vec[dim_idx] += sign * weight

    norm = torch.norm(vec, p=2)
    if norm > 1e-8:
        vec = vec / norm
    else:
        vec[0] = 1.0

    return vec


def load_dataset(
    jsonl_path: str,
    label_to_idx: Optional[Dict[str, int]] = None,
    embed_dim: int = 256
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, int]]:
    """Load JSONL dataset and encode samples into embedding tensors."""
    samples = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    if not samples:
        raise ValueError(f"Dataset at {jsonl_path} is empty.")

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


def brier_score_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    temp: float = 1.0,
    bias: float = 0.0
) -> torch.Tensor:
    """Mean squared error between softmax probabilities and one-hot targets."""
    scaled_logits = logits * temp + bias
    probs = F.softmax(scaled_logits, dim=-1)
    num_classes = logits.size(-1)
    targets_one_hot = F.one_hot(targets, num_classes=num_classes).float()
    return torch.mean((probs - targets_one_hot) ** 2)


def compute_ece(probs: torch.Tensor, targets: torch.Tensor, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error across confidence bins."""
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


def fit_platt_calibration(
    logits: torch.Tensor,
    targets: torch.Tensor,
    init_temp: float = 1.0,
    init_bias: float = 0.0,
    max_iter: int = 100
) -> Tuple[float, float]:
    """Fit temperature scaling parameter A and bias B via L-BFGS optimization."""
    log_temp = torch.tensor([math.log(max(init_temp, 0.01))], requires_grad=True)
    bias = torch.tensor([init_bias], requires_grad=True)

    optimizer = torch.optim.LBFGS(
        [log_temp, bias], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe"
    )
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
    return torch.exp(log_temp).item(), bias.item()


def train_cartridge(
    train_path: str,
    val_path: Optional[str] = None,
    output_axb: str = "cartridge.axb",
    embed_dim: int = 256,
    epochs: int = 250,
    lr: float = 0.05,
    weight_decay: float = 1e-4,
    verbose: bool = True
) -> Dict[str, Any]:
    """Train decision head with Brier loss and export calibrated .axb cartridge."""
    X_train, y_train, label_to_idx = load_dataset(train_path, embed_dim=embed_dim)

    if val_path and os.path.exists(val_path):
        X_val, y_val, _ = load_dataset(val_path, label_to_idx=label_to_idx, embed_dim=embed_dim)
    else:
        n_total = len(X_train)
        n_val = max(1, int(n_total * 0.2))
        perm = torch.randperm(n_total)
        val_indices = perm[:n_val]
        train_indices = perm[n_val:]

        X_val = X_train[val_indices]
        y_val = y_train[val_indices]
        X_train = X_train[train_indices]
        y_train = y_train[train_indices]

    idx_to_label = {idx: lbl for lbl, idx in label_to_idx.items()}
    labels_list = [idx_to_label[i] for i in range(len(label_to_idx))]
    num_choices = len(labels_list)

    torch.manual_seed(42)
    model = nn.Linear(embed_dim, num_choices, bias=False)
    nn.init.orthogonal_(model.weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

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

            if val_loss < best_val_brier:
                best_val_brier = val_loss
                best_weights = model.weight.detach().t().clone()

    if best_weights is None:
        best_weights = model.weight.detach().t().clone()

    with torch.no_grad():
        final_val_logits = X_val @ best_weights
        pre_brier = brier_score_loss(final_val_logits, y_val).item()
        pre_probs = F.softmax(final_val_logits, dim=-1)
        pre_ece = compute_ece(pre_probs, y_val)
        pre_acc = pre_probs.argmax(dim=-1).eq(y_val).float().mean().item() * 100.0

    platt_temp_mult, platt_bias = fit_platt_calibration(final_val_logits, y_val)

    with torch.no_grad():
        post_logits = final_val_logits * platt_temp_mult + platt_bias
        post_probs = F.softmax(post_logits, dim=-1)
        post_brier = brier_score_loss(final_val_logits, y_val, temp=platt_temp_mult, bias=platt_bias).item()
        post_ece = compute_ece(post_probs, y_val)
        post_acc = post_probs.argmax(dim=-1).eq(y_val).float().mean().item() * 100.0
        avg_confidence = post_probs.max(dim=-1).values.mean().item()

    hardware_temp = 1.0 / max(platt_temp_mult, 1e-6)

    weights_np = best_weights.cpu().numpy()
    bytes_written = export_cartridge(
        filepath=output_axb,
        weights=weights_np,
        labels=labels_list,
        platt_temperature=hardware_temp,
        platt_bias=platt_bias,
        embed_dim=embed_dim,
        num_choices=num_choices
    )

    return {
        "status": "success",
        "output_path": os.path.abspath(output_axb),
        "num_choices": num_choices,
        "labels": labels_list,
        "bytes_written": bytes_written,
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "val_accuracy": post_acc,
        "val_brier": post_brier,
        "val_ece": post_ece,
        "mean_confidence": avg_confidence,
        "hardware_platt_temperature": hardware_temp,
        "hardware_platt_bias": platt_bias
    }
