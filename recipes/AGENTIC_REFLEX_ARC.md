# Integration Recipe: Axobrier Autonomous Agent Reflex Arc

Documentation and integration patterns for the resident reflex hook module (`axobrier_reflex.py`), designed for autonomous developer loops and agentic frameworks.

---

## 1. Architectural Concept

Autonomous coding agents (such as Axobrier, Claude Code, Cursor background agents, and custom ReAct loops) spend significant time in cyclic trial-and-error loops:
1. Agent proposes a code edit or command.
2. The terminal returns an error (e.g. missing dependency, syntax mistake, type mismatch).
3. **Without Axobrier:** The entire traceback is serialized into a 1,500-token prompt, transmitted across the cloud to a frontier LLM (costing $0.01 and taking 2 to 4 seconds), just to decide: *"Run npm install <pkg>"*.
4. **With Axobrier:** The error string is intercepted on metal by `axobrier_reflex.triage_error()`. In **40 microseconds**, Axobrier classifies the category with 99.9% Brier calibration and extracts the exact remediation command.

```
┌───────────────────────────────────────────────────────────┐
│               Autonomous Agent Execution Loop             │
└─────────────┬─────────────────────────────────────────────┘
              │ Command fails with stderr
              ▼
┌───────────────────────────────────────────────────────────┐
│       Axobrier Resident Reflex Arc (axobrier_reflex)│
│      - Resident GPU Memory Arena (<32 MB VRAM)           │
│      - Pre-loaded models/build_triage.axb & agent_router │
└─────────────┬─────────────────────────────────────────────┘
              │ 40 µs evaluation
              ▼
    Confidence >= 0.95?
     ├── YES ──► Execute Suggested Action (0 tokens, $0.00)
     └── NO  ──► Escalate to Frontier LLM (System 2 Reasoning)
```

---

## 2. Using `axobrier_reflex.py`

The module is located at the repository root and can be imported directly into any Python agent loop.

### Resident Loading
On import, `axobrier_reflex` provisions a persistent GPU context and memory-maps both production cartridges:
- `models/build_triage.axb` (`C=7`: `BUILD_SUCCESS`, `ENV_VAR_MISSING`, `FILE_NOT_FOUND`, `MISSING_DEPENDENCY`, `SYNTAX_ERROR`, `TEST_ASSERTION_FAILURE`, `TYPE_MISMATCH`)
- `models/agent_router.axb` (`C=7`: `GREP_SYMBOL`, `READ_FILE_CHUNK`, `RUN_BUILD_TEST`, `INSPECT_GIT_DIFF`, `EXPAND_CONTEXT`, `INVOKE_SUBAGENT`, `GENERATE_IMAGE`)

Zero disk read overhead occurs on subsequent query evaluations.

---

## 3. Core Hook Functions

### 1. `triage_error(error_log: str) -> dict`
Evaluates compiler diagnostics, test tracebacks, and terminal error outputs:

```python
import axobrier_reflex as axo

error = "Cannot find module '@tanstack/react-query' or its corresponding type declarations. (TS2307)"
result = axo.triage_error(error)

print(result)
# {
#   "category": "MISSING_DEPENDENCY",
#   "confidence": 1.0,
#   "latency_us": 40.58,
#   "auto_action": "npm install @tanstack/react-query",
#   "raw_logit": 7.46
# }
```

### 2. `route_tool(instruction: str) -> dict`
Evaluates natural language agent tasks to pick the optimal next tool:

```python
import axobrier_reflex as axo

instruction = "Find where axo_cartridge_load_mmap is defined in include/ using ripgrep"
result = axo.route_tool(instruction)

print(result)
# {
#   "selected_tool": "GREP_SYMBOL",
#   "confidence": 1.0,
#   "latency_us": 40.13,
#   "raw_logit": 5.51
# }
```

---

## 4. Empirical Benchmark Telemetry

Recorded on physical consumer hardware (NVIDIA RTX 4090 / sm_120):

| Invocation | Query | Output | Latency |
| :--- | :--- | :--- | :--- |
| **`triage_error()`** | `Cannot find module 'express' (TS2307)` | `MISSING_DEPENDENCY` &rarr; `npm install express` | **40.58 µs** |
| **`triage_error()`** | `TypeError: unsupported operand type(s)` | `TYPE_MISMATCH` | **40.58 µs** |
| **`triage_error()`** | `error: expected ';' before '}' token` | `SYNTAX_ERROR` &rarr; line 42 check | **41.25 µs** |
| **`route_tool()`** | `Search codebase with ripgrep for symbol` | `GREP_SYMBOL` | **40.13 µs** |
| **`route_tool()`** | `Read lines 1 to 80 of include/axobrier_core.h` | `READ_FILE_CHUNK` | **40.42 µs** |
| **`route_tool()`** | `Inspect unstaged git changes and diff` | `INSPECT_GIT_DIFF` | **39.62 µs** |

---

## 5. Integrating with an Agent Loop

```python
# agent_reflex_loop.py
import subprocess
import axobrier_reflex as axo

def execute_agent_step(action_instruction: str):
    # 1. Sub-millisecond tool selection reflex
    routing = axo.route_tool(action_instruction)
    
    if routing["confidence"] >= 0.90:
        print(f"⚡ Fast-dispatching tool: {routing['selected_tool']} in {routing['latency_us']:.1f} µs")
        # Dispatch local tool directly...
    else:
        print("Ambiguous intent. Invoking cloud planner LLM...")
```
