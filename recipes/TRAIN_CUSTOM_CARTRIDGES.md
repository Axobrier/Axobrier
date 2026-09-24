# Recipe: Training & Calibrating Custom Domain Cartridges (.axb)

Learn how to distill any closed-set classification or decision-routing task into a standalone, microsecond-latency `.axb` cartridge using the Axobrier CLI.

---

## Overview

An Axobrier cartridge (`.axb`) consists of:
1. **28-byte packed header** (magic `0x00425841`, version, dimensions, Platt calibration parameters).
2. **Device-mapped FP16 choice projection matrix** ($C \times D$).
3. **Null-delimited UTF-8 label string table**.

Training takes **under 3 minutes** on a standard GPU or CPU and produces a deterministic file that can be hot-swapped into VRAM with zero runtime memory allocations.

---

## Step 1: Format Your Training Data (`.jsonl`)

Axobrier consumes standard line-delimited JSON (`.jsonl`). Each row must have:
- `text`: The input query, diagnostic string, or user prompt.
- `label`: The target categorical choice.

Example `data/my_domain_train.jsonl`:
```json
{"text": "Cannot find module '@tanstack/react-query'", "label": "MISSING_DEPENDENCY"}
{"text": "TypeError: 'NoneType' object is not callable", "label": "TYPE_MISMATCH"}
{"text": "Uncaught SyntaxError: Unexpected token '}'", "label": "SYNTAX_ERROR"}
{"text": "AWS_SECRET_ACCESS_KEY is not defined in environment", "label": "ENV_VAR_MISSING"}
{"text": "AssertionError: expected status 200, got 500", "label": "TEST_ASSERTION_FAILURE"}
```

> **Tip:** You can use Claude or GPT-4o to synthetically generate 500 to 1,000 diverse training examples for your specific domain in minutes.

---

## Step 2: Train & Calibrate via CLI

Run the single-command training pipeline:

```bash
python -m axobrier.cli train \
  --data data/my_domain_train.jsonl \
  --val-data data/my_domain_val.jsonl \
  --output models/my_router.axb \
  --embed-dim 256 \
  --epochs 25 \
  --batch-size 32 \
  --lr 0.005
```

### What Happens During Training?
1. **Embedding Projection:** Inputs are embedded into fixed $D=256$ vector space.
2. **Brier Loss Minimization:** The choice head is trained using Brier score loss to penalize overconfident, uncalibrated probability predictions.
3. **Platt Temperature Scaling:** After training, validation logits are fitted against a temperature parameter $T$ and bias $b$ to ensure output probabilities reflect true empirical error bounds.
4. **Binary Packing:** Weights are converted to FP16 and packed into the `.axb` binary format along with label strings.

---

## Step 3: Inspect & Deploy Your Cartridge

### 1. Inspect Header & Metadata
Verify the compiled cartridge using the `axobrier inspect` command:

```bash
axobrier inspect models/my_router.axb
```

Output:
```json
{
  "status": "success",
  "cartridge": "my_router.axb",
  "magic": "0x00425841",
  "version": 1,
  "embed_dim": 256,
  "num_choices": 5,
  "platt_temperature": 0.5124,
  "labels": [
    "ENV_VAR_MISSING",
    "MISSING_DEPENDENCY",
    "SYNTAX_ERROR",
    "TEST_ASSERTION_FAILURE",
    "TYPE_MISMATCH"
  ],
  "total_size_bytes": 2844
}
```

### 2. Test Execution on GPU
Test live queries directly from the command line:

```bash
axobrier route \
  --cartridge models/my_router.axb \
  --query "Cannot find module 'dotenv'"
```

Output:
```json
{
  "status": "success",
  "choice_index": 1,
  "label": "MISSING_DEPENDENCY",
  "confidence": 0.9996,
  "raw_logit": 8.12,
  "entropy": 0.0021,
  "latency_us": 40.85
}
```

### 3. Deploy in Python
Load the cartridge into your application with zero-copy memory mapping:

```python
from axobrier import AxoEngine

with AxoEngine() as engine:
    cart = engine.load_cartridge("models/my_router.axb")
    result = engine.route(cart, "Cannot find module 'dotenv'")
    print(f"Action: {result['label']} ({result['confidence']*100:.1f}%) in {result['latency_us']} µs")
```
