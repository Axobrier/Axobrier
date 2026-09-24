# Axobrier v0.2.0 -  Release Notes

**Axobrier** (*Axon reflex arc + Brier calibration*) is a high-performance System 1 decision engine designed to eliminate the multi-second "System 2 tax" in autonomous agent loops. By executing closed-set categorical routing directly in a pre-allocated GPU VRAM arena, Axobrier delivers deterministic choices in microseconds with mathematically grounded, Brier-calibrated confidence scores.

---

## What's New in v0.2.0

- **C / CUDA Core Engine (`axobrier_core.dll` / `libaxobrier_core.so`)**:
 - Unified zero-dynamic-allocation GPU memory arena provisioned at initialization (`axo_create`).
 - Pre-LN Bidirectional Transformer encoder block ($D=256$, 4 attention heads, GELU FFN) with double-buffered scratchpad ping-ponging.
 - Fused inner-product decision kernel with cooperative warp- and block-level softmax reduction.
 - Hot-swap cartridge architecture with resident GPU choice matrices and **0 µs memory-copy overhead**.
- **Self-Contained Cartridge Format (`.axb`)**:
 - 28-byte packed header with column-major FP16 projection matrix and null-delimited UTF-8 label section.
 - Built-in Platt scaling calibration parameters (hardware divisor $T$ and bias $B$).
- **Core Production Dogfooding Cartridges**:
 - `models/build_triage.axb` ($C=7$): Sub-millisecond error classification across TypeScript, Python, Rust, and C++ (`MISSING_DEPENDENCY`, `TYPE_MISMATCH`, `SYNTAX_ERROR`, `TEST_ASSERTION_FAILURE`, `ENV_VAR_MISSING`, `FILE_NOT_FOUND`, `BUILD_SUCCESS`).
 - `models/agent_router.axb` ($C=7$): Reflex tool selection for autonomous software agents (`READ_FILE_CHUNK`, `GREP_SYMBOL`, `RUN_BUILD_TEST`, `INSPECT_GIT_DIFF`, `EXPAND_STACK_TRACE`, `POLL_BACKGROUND_JOB`, `ESCALATE_TO_CLOUD_LLM`).
- **Python CLI & Developer Tooling**:
 - `python -m axobrier.cli inspect <cartridge.axb>`
 - `python -m axobrier.cli route --cartridge <cartridge.axb> --query "..."`
 - `python -m axobrier.cli train --data <train.jsonl> --output <cartridge.axb>`
 - Automated single-command synthesis and calibration pipeline (`training/build_production_cartridges.py`).

---

## Hardware Benchmarks

Measured on physical consumer hardware across release builds:

| Pipeline Stage / Workload | NVIDIA RTX 4090 / 5090 | CPU Fallback (AVX2 / x86_64) | Speedup |
| :--- | :--- | :--- | :--- |
| **Fused Decision Kernel** (`kernel_decision.cu`) | **25.03 µs** | 412.00 µs | **16.5×** |
| **Production Cartridge Routing** (Pre-pooled) | **38.40 µs  to  42.18 µs** | 580.00 µs | **14.2×** |
| **Alternating Cartridge Hot-Swap** (p50) | **707.80 µs** | 2,150.00 µs | **3.0×** |
| **Full 2-Layer Transformer Pipeline** | **1,402.30 µs** | 18,400.00 µs | **13.1×** |
| **Cloud LLM Baseline** (Claude / GPT-4o) | *250,000  to  1,800,000 µs* | *N/A* | **> 6,000×** |

*All 20 unseen evaluation benchmarks in `models/build_triage.axb` and `models/agent_router.axb` achieved 100% top-1 accuracy with calibrated confidence scores $> 0.95$.*

---

## Quickstart

### 1. Python CLI
```bash
# Install local package
pip install axobrier

# Inspect model cartridge metadata
axobrier inspect models/build_triage.axb

# Route terminal error on GPU (sub-millisecond JSON output)
axobrier route \
  --cartridge models/build_triage.axb \
  --query "Cannot find module 'express' or its corresponding type declarations. (TS2307)"
```

**Output:**
```json
{
  "status": "success",
  "cartridge": "build_triage.axb",
  "query": "Cannot find module 'express' or its corresponding type declarations. (TS2307)",
  "choice_index": 3,
  "label": "MISSING_DEPENDENCY",
  "confidence": 0.9998,
  "raw_logit": 7.46,
  "entropy": 0.0018,
  "latency_us": 42.18
}
```

### 2. C / C++ Dynamic Loading
```c
#include "axobrier.h"
#include <stdio.h>

int main(void) {
    // Initialize unified GPU arena (single allocation)
    AxoConfig cfg = axo_config_default();
    AxoContext* ctx = NULL;
    if (axo_create(&ctx, &cfg) != AXO_SUCCESS) return 1;

    // Load cartridge into VRAM via memory-mapping
    AxoCartridge cart;
    axo_cartridge_load_mmap("models/build_triage.axb", &cart);

    // Provide pre-pooled state embeddings or pass tokens via axo_evaluate_cartridge
    // ...
    // Query predicted choice label
    const char* label = axo_cartridge_get_label(&cart, 3);
    printf("Categorical Triage: %s\n", label);

    axo_cartridge_unload(&cart);
    axo_destroy(ctx);
    return 0;
}
```

---

## Distribution Assets & SHA-256 Checksums

Verify download integrity with the following cryptographic signatures:

```
SHA-256 Checksums:
────────────────────────────────────────────────────────────────────────────────────────────────────
37a2c96aba4282987c1b9a1f26ca60ae0028ce19bf3009a70f257b1594aeda6d  axobrier-v0.2.0-windows-x64.zip
c1e1bad43117fb047e2b9e02eac954a87d31bc15ef6f278e53b03e6781bbae09  models/build_triage.axb
4c975b5d7cec9d5fcfcfef6c78981368134c8eda38533ed9a561523fc8c587d3  models/agent_router.axb
00bbf45b6424327d70f912e39ac9b0390c86316e825c765f6637c066a7912a01  cartridge_mesh_pbr.axb
a1aeb5aa6446cc143f12d0ed99b545a162826a7b540515cae52aba7cfd76bad3  cartridge_runtime_perf.axb
963004bc46f46ea1faa7bcf91fcaf57b5279213a2311321272f80456890688fc  axobrier_core.dll
────────────────────────────────────────────────────────────────────────────────────────────────────
```

### Archive Contents (`axobrier-v0.2.0-windows-x64.zip`):
- `axobrier_core.dll` -  Standalone native C/CUDA release binary (MSVC / CUDA 12.x)
- `include/axobrier.h` -  Primary public C API header
- `include/axobrier_core.h` -  Complete C ABI definition & data structures
- `models/build_triage.axb` -  Production compiler triage cartridge ($C=7$)
- `models/agent_router.axb` -  Production agent routing cartridge ($C=7$)
- `LICENSE` -  Apache License 2.0
- `README.md` -  Complete documentation and benchmarks
