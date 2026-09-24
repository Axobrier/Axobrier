# Integration Recipe: Cursor, Codex & Claude Code Build Triage Rules

Drop-in rules and configurations for **Cursor**, **Codex**, **Windsurf**, and **Claude Code** to intercept compiler errors and build logs using Axobrier.

---

## 1. Cursor & Windsurf Rule (`.cursorrules` or `.cursor/rules/axobrier.mdc`)

Create a file named `.cursorrules` (or `.cursor/rules/axobrier.mdc` for Cursor 0.40+) at the root of your workspace:

```markdown
---
description: Axobrier sub-millisecond build triage and tool dispatch rules
globs: *
alwaysApply: true
---

# Local Sub-Millisecond Reflex Rules

You have access to the local Axobrier inference engine via `axobrier_reflex.py` or the CLI (`axobrier`).

## 1. Terminal Error Interception (Zero-Token Triage)
Whenever a command executed in the terminal fails with compiler diagnostics, missing modules, or broken tests:
- DO NOT immediately prompt a large model to analyze the full traceback.
- First, pipe the error through Axobrier locally:
  ```bash
  python -c "import axobrier_reflex as axo; print(axo.triage_error('''<ERROR_TEXT>'''))"
  ```

## 2. Immediate Remediation Actions
- If `category == "MISSING_DEPENDENCY"` and `confidence >= 0.95`:
  Immediately run the `auto_action` command (e.g. `npm install <pkg>`, `pip install <pkg>`, `cargo add <crate>`) without writing prose.
- If `category == "SYNTAX_ERROR"`:
  Jump to the referenced file and line number to inspect missing commas, colons, or mismatched braces.
- If `category == "ENV_VAR_MISSING"`:
  Check `.env.example` and set the required variable.

## 3. Escalation Rules
Only write long explanations or query cloud LLMs if Axobrier returns `confidence < 0.90` or the error indicates an architectural logic bug.
```

---

## 2. Codex & Terminal AI CLI Prompt Injection

When using Codex CLI tools or shell wrappers, inject this reflex guard into your system prompt or alias:

```bash
# In your ~/.bashrc or ~/.zshrc:
axo_eval() {
    python -c "import axobrier_reflex as axo, sys; print(axo.triage_error(sys.stdin.read()))"
}

# Example pipe:
# npm run build 2>&1 | axo_eval
```

---

## 3. Claude Code Rule (`CLAUDE.md`)

Add the following to your project's `CLAUDE.md` in the repository root:

```markdown
## Fast Diagnostics with Axobrier
Before writing multi-paragraph diagnostic reasoning for failed builds, check Axobrier:
1. Run `python -c "import axobrier_reflex as axo; print(axo.triage_error('''$(cat build.log)'''))"`
2. If `MISSING_DEPENDENCY`, execute the package install command directly.
```

---

## 4. Why This Matters

1. **Token Conservation:** Eliminates 1,000 to 3,000 tokens of noisy compiler error output from context windows.
2. **Instant Feedback:** 40 µs response time avoids the 3 to 5 second pause while waiting for cloud models to parse basic syntax or import errors.
3. **Deterministic Cleanliness:** Guarantees standard package manager commands are suggested without hallucinated flags.
