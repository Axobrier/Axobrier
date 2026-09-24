# Integration Recipe: Cursor Rules & Claude Code Reflex Arc

Use Axobrier to intercept terminal build diagnostics, compiler failures, and tool selections in 40 microseconds - **before** spending Claude or GPT-4o tokens.

---

## 1. Why Hook Axobrier into Agentic IDEs?

When a compiler error or test failure occurs during an agentic coding loop:
- **Standard LLM Flow:** Terminal error output (500 to 2,000 tokens) is sent across the cloud to Claude 3.7 or GPT-4o. The model takes 2 to 4 seconds to read the traceback and tell you to run `npm install express`.
- **Axobrier Reflex Flow:** Axobrier evaluates the stderr log locally on metal in **42 microseconds**. It immediately classifies `MISSING_DEPENDENCY` with 99.9% confidence and proposes `npm install express` before the LLM prompt is even assembled.

---

## 2. Ready-to-Copy `.cursorrules` Configuration

Place the following into `.cursorrules` or `.cursor/rules/axobrier.mdc` in your workspace root:

```markdown
# Axobrier Reflex Rules for Cursor & Terminal Diagnostics

You have access to the local Axobrier reflex engine via `axobrier_reflex.py`.
Whenever a command produces a build error, compiler diagnostic, or failed test:

1. ZERO-TOKEN TRIAGE FIRST:
   Before sending a long explanation or modifying code, run the error through Axobrier triage:
   ```bash
   python -c "import axobrier_reflex as axo; import json; print(json.dumps(axo.triage_error('''<PASTE_STDERR_HERE>''')))"
   ```

2. ACT ON HIGH-CONFIDENCE TRIAGE:
  - If `confidence >= 0.95` and `category == "MISSING_DEPENDENCY"`:
     Immediately execute the recommended `auto_action` (e.g. `npm install <pkg>`, `pip install <pkg>`, `cargo add <crate>`) without writing prose.
  - If `category == "SYNTAX_ERROR"`:
     Jump directly to the flagged line number and inspect missing punctuation or unmatched braces.
  - If `category == "ENV_VAR_MISSING"`:
     Inspect `.env.example` and propose the missing environment key.

3. ESCALATE ONLY ON AMBIGUITY:
   Only engage full multi-step LLM reasoning if Axobrier confidence is < 0.90 or category is ambiguous.
```

---

## 3. Claude Code / Terminal Agent Integration

For autonomous agent loops (e.g., Claude Code, AutoGPT, or custom CLI agents), wrap your command execution tool with this Python wrapper:

```python
# reflex_terminal_runner.py
import subprocess
import axobrier_reflex as axo

def run_command_with_reflex(cmd: str) -> dict:
    """
    Executes a shell command. If it fails, runs sub-millisecond local triage
    to propose instant resolution without cloud LLM round-trips.
    """
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if proc.returncode == 0:
        return {"status": "success", "stdout": proc.stdout}

    # Intercept stderr with Axobrier
    error_text = proc.stderr.strip() or proc.stdout.strip()
    triage = axo.triage_error(error_text)

    # If confidence is high, return the automated next action
    if triage["confidence"] >= 0.95 and triage["category"] == "MISSING_DEPENDENCY":
        return {
            "status": "auto_remediation",
            "category": triage["category"],
            "suggested_action": triage["auto_action"],
            "triage_latency_us": triage["latency_us"],
            "raw_stderr": error_text
        }

    return {
        "status": "error",
        "category": triage["category"],
        "confidence": triage["confidence"],
        "raw_stderr": error_text
    }

if __name__ == "__main__":
    result = run_command_with_reflex("npm run build")
    print(result)
```

---

## 4. Benchmark: Token & Time Savings

| Metric | Cloud LLM Triage | Axobrier Reflex Arc | Savings |
| :--- | :--- | :--- | :--- |
| **Turnaround Latency** | 2,400 ms | **0.042 ms (42 µs)** | **> 50,000× faster** |
| **Token Cost** | ~1,200 tokens ($0.0036) | **0 tokens ($0.00)** | **100% free** |
| **Network Egress** | Error logs sent to cloud | **100% local on metal** | **Air-gapped** |
