# Axobrier

> **Deterministic System 1 Choice Routing Engine for Autonomous AI Agents**  
> *Sub-millisecond hardware-accelerated categorical routing with Brier-calibrated confidence scoring and hot-swappable `.axb` choice cartridges.*

---

## Overview

Modern agentic AI architectures suffer from the **"System 2 tax"**: invoking large autoregressive language models (costing 200 ms to 2,000+ ms) simply to make closed-set categorical decisions - such as tool selection, syntax validation, performance triage, or safety gating.

**Axobrier** (*Axon reflex arc + Brier calibration*) provides a dedicated **System 1 reflex arc**:
- **Zero-Allocation GPU Arena**: Provisions a unified device memory arena on startup (`axo_create`); guarantees **zero dynamic host or device allocations** during inference.
- **Base Console + Swappable Cartridge Architecture**: A resident bidirectional Pre-LN Transformer encoder maps input state to normalized dense embeddings ($D=256$) directly in VRAM. Domain-specific choice heads load as lightweight, swappable `.axb` binary cartridges.
- **Microsecond Execution**: Evaluates decision matrices and cooperative softmax reductions in **~25 µs** GPU kernel time.
- **Calibrated Probabilities**: Optimized via multi-class **Brier Score Loss** with post-training **Platt scaling**, producing reliable confidence scores where $P \ge 0.85$ indicates mathematically grounded certainty.

---

## Hardware Benchmarks (NVIDIA RTX 5090)

All benchmarks measured directly on physical consumer Blackwell hardware (`compute_120`, sm_120) under release compilation:

| Operation | Latency (p50) | Latency (p99) | Throughput | Target | Headroom |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Fused Choice Kernel** (`kernel_decision.cu`) | **25.03 µs** | **26.85 µs** | 39,952 ops/s | $< 100 \text{ µs}$ | **75%** |
| **Full 2-Layer Transformer Pipeline** | **1,402.30 µs** | **1,521.80 µs** | 713 ops/s | $< 5,000 \text{ µs}$ | **72%** |
| **Alternating Cartridge Hot-Swap** | **707.80 µs** | **1,163.60 µs** | 1,413 ops/s | $< 5,000 \text{ µs}$ | **77%** |
| **Production Cartridge Routing** (Pre-pooled) | **41.20 µs** | **149.60 µs** | 24,271 ops/s | $< 500 \text{ µs}$ | **70%** |

*Zero host-to-device memory copies occur on the hot-swap path; cartridges maintain pre-allocated resident VRAM projections swapped via direct pointer assignment.*

---

## Architecture

```
                                      ┌─────────────────────────────────────────┐
                                      │       Swappable Cartridge (.axb)        │
                                      │ - Column-Major Choice Matrix [D x C]   │
                                      │ - Platt Scaling (Divisor T, Bias B)    │
                                      │ - Null-Delimited UTF-8 Label Strings   │
                                      └────────────────────┬────────────────────┘
                                                           │ (Zero-Copy GPU Pointer Swap)
                                                           ▼
┌───────────────────┐      ┌───────────────────────────────┴───────────────────────────────┐
│ State Tokens      │ ───► │ Axobrier Memory Arena (CUDA VRAM)                             │
│ [batch × seq_len] │      │                                                               │
└───────────────────┘      │   ┌───────────────────────────────────────────────────────┐   │
                           │   │ Stage 1: Bidirectional Transformer Encoder           │   │
                           │   │ - LayerNorm + Multi-Head Self-Attention (4 Heads)     │   │
                           │   │ - Ping-Pong Scratchpad Activations                    │   │
                           │   │ - GELU Feed-Forward Network (FFN Multiplier = 4)      │   │
                           │   │ - Mask-Aware Mean Pooling + L2 Normalization          │   │
                           │   └───────────────────────────┬───────────────────────────┘   │
                           │                               │ Dense State Embedding [D=256] │
                           │                               ▼                               │
                           │   ┌───────────────────────────────────────────────────────┐   │
                           │   │ Stage 2: Fused Decision Kernel                        │   │
                           │   │ - Inner Product: dot = embed · choice_col             │   │
                           │   │ - Platt Temperature Scaling: logit = dot / T + B      │   │
                           │   │ - Cooperative Warp/Block Softmax Reduction            │   │
                           │   └───────────────────────────┬───────────────────────────┘   │
                           └───────────────────────────────┼───────────────────────────────┘
                                                           │
                                                           ▼
                                      ┌─────────────────────────────────────────┐
                                      │ Decision Result                         │
                                      │ - choice_index:  uint32                 │
                                      │ - label:         "MAX_4_BONE_INFLUENCES"│
                                      │ - confidence:    0.9997 (> 0.85)        │
                                      │ - raw_logit:     14.30                  │
                                      │ - entropy:       0.0001 nats            │
                                      │ - latency_us:    40.50 µs               │
                                      └─────────────────────────────────────────┘
```

---

## Cartridge Binary Specification (`.axb`)

Cartridges are self-contained, memory-mappable binary assets with a strict 28-byte packed header followed by an FP16 weight matrix and UTF-8 label section.

### Binary Layout
```
Offset   Length     Field                  Description
─────────────────────────────────────────────────────────────────────────────────
0x00     4 bytes    uint32 magic           0x00425841 ("AXB\0" little-endian)
0x04     4 bytes    uint32 version         Specification version (1)
0x08     4 bytes    uint32 embed_dim       Embedding dimension (must match D=256)
0x0C     4 bytes    uint32 num_choices     Choice cardinality (C)
0x10     4 bytes    float32 platt_temp     Hardware temperature divisor (T)
0x14     4 bytes    float32 platt_bias     Calibration bias (B)
0x18     4 bytes    uint32 label_bytes     Byte length of UTF-8 label payload
0x1C     D*C*2 B    fp16 choice_matrix     Column-major choice matrix [D × C]
varies   varies     bytes labels           Null-delimited strings: "L0\0L1\0...LC-1\0"
```

---

## Production Cartridges Included

The repository includes ready-to-use production cartridges under `models/` and root:

### 1. `models/build_triage.axb` (Terminal & Compiler Error Triage)
- **Target Domain**: Multi-language build diagnostics (TypeScript, Python, Rust, C++).
- **Classes ($C=7$)**: `MISSING_DEPENDENCY`, `TYPE_MISMATCH`, `SYNTAX_ERROR`, `TEST_ASSERTION_FAILURE`, `ENV_VAR_MISSING`, `FILE_NOT_FOUND`, `BUILD_SUCCESS`.
- **Latency**: ~38 µs on RTX 5090.

### 2. `models/agent_router.axb` (Autonomous Agent Next-Action Routing)
- **Target Domain**: Sub-millisecond reflex routing for agentic workflows.
- **Classes ($C=7$)**: `READ_FILE_CHUNK`, `GREP_SYMBOL`, `RUN_BUILD_TEST`, `INSPECT_GIT_DIFF`, `EXPAND_STACK_TRACE`, `POLL_BACKGROUND_JOB`, `ESCALATE_TO_CLOUD_LLM`.
- **Latency**: ~40 µs on RTX 5090.

### 3. `cartridge_mesh_pbr.axb` (3D Mesh & glTF PBR Validation)
- **Target Domain**: 3D asset pipeline validation, real-time engine avatar constraints, and glTF 2.0 PBR authoring.
- **Classes ($C=8$)**: `MAX_4_BONE_INFLUENCES`, `ORM_TEXTURE_PACKING`, `ALPHA_MODE_MASK_FOR_HAIR`, `DAE_ARMATURE_RIGID_EXPORT`, `GLTF_PBR_SEPARATE_NORMAL`, `CONVEX_HULL_PHYSICS_DECOMP`, `LOD_4_REDUCTION_STRATEGY`, `LIMIT_65K_VERTICES_PER_SURFACE`.

### 4. `cartridge_runtime_perf.axb` (Runtime Performance Triage)
- **Target Domain**: Low-latency game engine profiling and backend service diagnostics.
- **Classes ($C=8$)**: `GC_ALLOC_IN_UPDATE_LOOP`, `UNINDEXED_DATABASE_QUERY`, `BLOCKING_IO_ON_EVENT_LOOP`, `UNBOUNDED_THREAD_SPAWNING`, `HOT_PATH_STRING_CONCAT`, `N_PLUS_ONE_ORM_FETCH`, `LEAKED_EVENT_SUBSCRIPTION`, `UNBOUNDED_MEMORY_CACHE`.

---

### Automated Pipeline Execution
Generate, train, calibrate, and verify cartridges end-to-end with a single command:
```bash
python training/build_production_cartridges.py
```

---

## Quickstart: Python CLI

### 1. Installation
Install the local package in editable mode:
```bash
pip install -e .
```

### 2. Inspect an `.axb` Cartridge
```bash
python -m axobrier.cli inspect cartridge_mesh_pbr.axb
```
**Output (JSON):**
```json
{
  "status": "success",
  "cartridge": "cartridge_mesh_pbr.axb",
  "magic": "0x00425841",
  "version": 1,
  "embed_dim": 256,
  "num_choices": 8,
  "platt_temperature": 0.4797,
  "platt_bias": 0.0,
  "label_section_bytes": 200,
  "labels": [
    "ALPHA_MODE_MASK_FOR_HAIR",
    "CONVEX_HULL_PHYSICS_DECOMP",
    "DAE_ARMATURE_RIGID_EXPORT",
    "GLTF_PBR_SEPARATE_NORMAL",
    "LIMIT_65K_VERTICES_PER_SURFACE",
    "LOD_4_REDUCTION_STRATEGY",
    "MAX_4_BONE_INFLUENCES",
    "ORM_TEXTURE_PACKING"
  ],
  "total_size_bytes": 4324
}
```

### 3. Evaluate Unseen Domain Queries on GPU
```bash
python -m axobrier.cli route \
  --cartridge cartridge_mesh_pbr.axb \
  --query "Shoulder joint deformation artifact: vertex is weighted to 5 bones simultaneously in avatar rig."
```
**Output (JSON):**
```json
{
  "status": "success",
  "cartridge": "cartridge_mesh_pbr.axb",
  "query": "Shoulder joint deformation artifact: vertex is weighted to 5 bones simultaneously in avatar rig.",
  "choice_index": 6,
  "label": "MAX_4_BONE_INFLUENCES",
  "confidence": 1.0,
  "raw_logit": 14.3,
  "entropy": 0.0001,
  "latency_us": 40.5
}
```

### 4. Train a Custom Domain Cartridge
```bash
python -m axobrier.cli train \
  --data training/data/runtime_perf_train.jsonl \
  --val training/data/runtime_perf_val.jsonl \
  --output my_custom_cartridge.axb \
  --epochs 250
```

---

## Quickstart: Python API

```python
from axobrier import AxoEngine

# Initialize GPU arena
with AxoEngine() as engine:
    # Memory-map cartridge into resident VRAM
    cart = engine.load_cartridge("cartridge_runtime_perf.axb")

    # Evaluate query
    decision = engine.route(
        cart,
        "LINQ allocation inside MonoBehaviour.Update() causing 60fps frame drops."
    )

    print(f"Predicted Action: {decision['label']}")
    print(f"Confidence:       {decision['confidence'] * 100:.1f}%")
    print(f"Latency:          {decision['latency_us']} µs")
```

---

## Quickstart: C / C++ API

```c
#include "axobrier_core.h"
#include <stdio.h>

int main() {
    AxoConfig cfg = axo_config_default();
    AxoContext* ctx = NULL;
    axo_create(&ctx, &cfg);

    AxoCartridge cart;
    axo_cartridge_load_mmap("cartridge_mesh_pbr.axb", &cart);

    // Full tokenized state pipeline (embed -> transformer -> mean pool -> route)
    int32_t tokens[16] = { 101, 4829, 203, 5, 0, 0 /* padded */ };
    AxoStateParams params = {
        .token_ids = tokens,
        .choice_matrix = NULL,
        .batch_size = 1,
        .seq_len = 16,
        .num_choices = cart.header.num_choices,
        .pad_token_id = 0
    };

    AxoDecision result;
    axo_evaluate_cartridge(ctx, &cart, &params, &result);

    const char* label = axo_cartridge_get_label(&cart, result.choice_index);
    printf("Choice #%u: %s (Prob: %.4f, Latency: %.2f us)\n",
           result.choice_index, label, result.calibrated_prob, result.latency_us);

    axo_cartridge_unload(&cart);
    axo_destroy(ctx);
    return 0;
}
```

---

## Build & Test

### Prerequisites
- Windows 11 / Linux
- NVIDIA CUDA Toolkit 12.0+
- CMake 3.25+
- MSVC v143 (VS 2022) or GCC 11+
- Python 3.8+ with PyTorch 2.0+

### Build from Source
```powershell
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j
```

### Run Test Suite
```powershell
# Core C++ regression suite (routing correctness, benchmarks, hot-swap)
ctest --test-dir build -C Release --output-on-failure

# Production cartridge hardware validation (16/16 unseen query tests)
python tests/test_production_cartridges.py
```

---

## Integration Recipes & Guides

Explore production workflows and drop-in integration patterns in the [`recipes/`](recipes/) directory:

- [**Cursor Rules & Claude Code Reflex Arc**](recipes/CURSOR_AND_CLAUDE_RULES.md): Intercept terminal build diagnostics and missing dependencies without burning frontier LLM tokens.
- [**GitHub Actions CI Automated Triage**](recipes/GITHUB_ACTIONS_CI.md): Complete `.github/workflows/triage.yml` workflow running in <500µs on CPU to label failed PRs.
- [**LangChain & Agent Tool Routing**](recipes/LANGCHAIN_AND_AGENT_ROUTING.md): Drop-in Python router replacing slow LLM tool selectors with resident cartridges.
- [**Training Custom Cartridges (.axb)**](recipes/TRAIN_CUSTOM_CARTRIDGES.md): Step-by-step guide to generating data and training Brier-calibrated `.axb` files in under 3 minutes.

---

## License

Apache License 2.0. See `LICENSE` for details.
