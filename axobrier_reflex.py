# Copyright 2026 Axobrier Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import os
import sys
import re
from typing import Dict, Any, Optional

import axobrier
from axobrier import AxoEngine, AxoCartridge

# ═══════════════════════════════════════════════════════════════════
#  RESIDENT ENGINE & CARTRIDGE INITIALIZATION
# ═══════════════════════════════════════════════════════════════════

_WORKSPACE_ROOT = os.path.dirname(os.path.abspath(__file__))

_BUILD_TRIAGE_PATH = os.path.join(_WORKSPACE_ROOT, "models", "build_triage.axb")
_AGENT_ROUTER_PATH = os.path.join(_WORKSPACE_ROOT, "models", "agent_router.axb")

# Global resident singleton
_ENGINE: Optional[AxoEngine] = None
_CART_TRIAGE: Optional[AxoCartridge] = None
_CART_ROUTER: Optional[AxoCartridge] = None


def _init_reflex():
    """Initializes resident AxoEngine and memory-maps both cartridges into resident memory."""
    global _ENGINE, _CART_TRIAGE, _CART_ROUTER
    if _ENGINE is not None:
        return

    _ENGINE = AxoEngine()

    if not os.path.exists(_BUILD_TRIAGE_PATH):
        raise FileNotFoundError(f"Build triage cartridge not found at: {_BUILD_TRIAGE_PATH}")
    if not os.path.exists(_AGENT_ROUTER_PATH):
        raise FileNotFoundError(f"Agent router cartridge not found at: {_AGENT_ROUTER_PATH}")

    # Memory-map both heads into resident memory
    _CART_TRIAGE = _ENGINE.load_cartridge(_BUILD_TRIAGE_PATH)
    _CART_ROUTER = _ENGINE.load_cartridge(_AGENT_ROUTER_PATH)


# Initialize automatically on module import
_init_reflex()


# ═══════════════════════════════════════════════════════════════════
#  HEURISTIC ACTION RESOLVER (Rule-based recommendation engine)
# ═══════════════════════════════════════════════════════════════════

def _suggest_triage_action(category: str, error_log: str) -> str:
    """Derives an automated remediation or terminal command recommendation."""
    if category == "MISSING_DEPENDENCY":
        # Node / TS
        m_ts = re.search(r"Cannot find module ['\"]([^'\"]+)['\"]", error_log)
        if m_ts:
            pkg = m_ts.group(1).split('/')[0]
            if pkg.startswith('@'):
                pkg = '/'.join(m_ts.group(1).split('/')[:2])
            return f"npm install {pkg}"

        # Python
        m_py = re.search(r"No module named ['\"]([^'\"]+)['\"]", error_log)
        if m_py:
            return f"pip install {m_py.group(1)}"

        # Rust
        m_rs = re.search(r"can't find crate for `([^`]+)`|unresolved import `([^`]+)`", error_log)
        if m_rs:
            crate = m_rs.group(1) or m_rs.group(2)
            return f"cargo add {crate}"

        return "Install missing package via project package manager"

    elif category == "ENV_VAR_MISSING":
        m_env = re.search(r"['\"]?([A-Z0-9_]{3,})['\"]?\s+(?:is not set|missing|required|undefined)", error_log, re.IGNORECASE)
        if m_env:
            return f"export {m_env.group(1)}=<value> (add to .env)"
        return "Configure missing environment variable in .env"

    elif category == "SYNTAX_ERROR":
        m_line = re.search(r"(?:line|:)\s*(\d+)", error_log, re.IGNORECASE)
        if m_line:
            return f"Inspect syntax around line {m_line.group(1)}"
        return "Inspect and fix syntax error"

    elif category == "TYPE_MISMATCH":
        return "Inspect type definition / check variable types and casts"

    elif category == "FILE_NOT_FOUND":
        m_file = re.search(r"(?:No such file or directory|file not found):?\s*['\"]?([^\s'\"]+)['\"]?", error_log, re.IGNORECASE)
        if m_file:
            return f"Verify file exists at path: {m_file.group(1)}"
        return "Verify target path exists"

    elif category == "TEST_ASSERTION_FAILURE":
        return "Inspect test assertion failure and diff actual vs expected output"

    elif category == "BUILD_SUCCESS":
        return "Build succeeded. Proceed to test execution or deployment."

    return "Review compiler diagnostic output"


# ═══════════════════════════════════════════════════════════════════
#  PUBLIC REFLEX HOOK API
# ═══════════════════════════════════════════════════════════════════

def triage_error(error_log: str) -> Dict[str, Any]:
    """
    Evaluates a compiler diagnostic or terminal error log against `build_triage.axb`.

    Parameters:
        error_log (str): Raw error traceback or compiler output.

    Returns:
        dict:
            - category (str): Triage classification (e.g. MISSING_DEPENDENCY, TYPE_MISMATCH, SYNTAX_ERROR)
            - confidence (float): Brier-calibrated probability (0.0 to 1.0)
            - latency_us (float): Hardware execution latency in microseconds
            - auto_action (str): Suggested automated remediation command
            - raw_logit (float): Raw output logit
    """
    if _ENGINE is None or _CART_TRIAGE is None:
        _init_reflex()

    res = _ENGINE.route(_CART_TRIAGE, error_log)
    category = res["label"]
    auto_action = _suggest_triage_action(category, error_log)

    return {
        "category": category,
        "confidence": res["confidence"],
        "latency_us": res["latency_us"],
        "auto_action": auto_action,
        "raw_logit": res["raw_logit"]
    }


def route_tool(instruction: str) -> Dict[str, Any]:
    """
    Evaluates a user instruction or planned sub-task against `agent_router.axb`.

    Parameters:
        instruction (str): Agent prompt or task description.

    Returns:
        dict:
            - selected_tool (str): Recommended action (e.g. GREP_SYMBOL, READ_FILE_CHUNK, RUN_BUILD_TEST)
            - confidence (float): Brier-calibrated probability (0.0 to 1.0)
            - latency_us (float): Hardware execution latency in microseconds
            - raw_logit (float): Raw output logit
    """
    if _ENGINE is None or _CART_ROUTER is None:
        _init_reflex()

    res = _ENGINE.route(_CART_ROUTER, instruction)

    return {
        "selected_tool": res["label"],
        "confidence": res["confidence"],
        "latency_us": res["latency_us"],
        "raw_logit": res["raw_logit"]
    }


# ═══════════════════════════════════════════════════════════════════
#  SELF-TEST & TELEMETRY BENCHMARK
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "=" * 78)
    print("  Axobrier Resident Reflex Hook Telemetry (axobrier_reflex.py)")
    print("=" * 78)

    sample_errors = [
        "Cannot find module 'express' or its corresponding type declarations. (TS2307)",
        "TypeError: unsupported operand type(s) for +: 'int' and 'str' in calculate_sum",
        "error: expected ';' before '}' token at line 42 in src/main.rs"
    ]

    sample_instructions = [
        "Find all occurrences of axo_cartridge_load_mmap across the codebase using ripgrep",
        "Read lines 1 to 80 of include/axobrier_core.h to inspect the struct definitions",
        "Inspect the unstaged git changes and diff in src/kernel_decision.cu"
    ]

    print("\n[1] Compiler & Diagnostic Triage (models/build_triage.axb):")
    print("-" * 78)
    for err in sample_errors:
        out = triage_error(err)
        print(f" Query:       {err[:68]}...")
        print(f" Category:    \033[92m{out['category']}\033[0m (Confidence: {out['confidence']*100:.2f}%)")
        print(f" Action:      \033[96m{out['auto_action']}\033[0m")
        print(f" Latency:     \033[93m{out['latency_us']:.2f} us\033[0m")
        print("-" * 78)

    print("\n[2] Agentic Tool Routing (models/agent_router.axb):")
    print("-" * 78)
    for inst in sample_instructions:
        out = route_tool(inst)
        print(f" Instruction: {inst[:68]}...")
        print(f" Tool:        \033[92m{out['selected_tool']}\033[0m (Confidence: {out['confidence']*100:.2f}%)")
        print(f" Latency:     \033[93m{out['latency_us']:.2f} us\033[0m")
        print("-" * 78)

    print("\nAll resident reflex hooks executed successfully in microsecond range.\n")
