# Integration Recipe: LangChain & Agent Reflex Routing

Replace slow, expensive LLM-based intent classifiers and tool routers with resident Axobrier `.axb` cartridges. Achieve sub-millisecond dispatching with zero token consumption.

---

## 1. The Bottleneck: LLMs as Routers

In standard agentic frameworks (LangChain, LangGraph, LlamaIndex, CrewAI), selecting the next tool often requires an LLM call:
- **LLM Router:** Takes **1,000ms  to  2,500ms** and consumes **500 to 1,200 tokens** per turn just to choose between `read_file`, `grep_symbol`, or `run_tests`.
- **Axobrier Reflex Router:** Evaluates user intent on metal in **40 microseconds** using pre-allocated GPU arenas or AVX2 CPU instructions, consuming **0 tokens**.

---

## 2. Minimal Drop-in Router Snippet

```python
# axobrier_langchain_router.py
from typing import Dict, Any, Callable
import axobrier_reflex as axo

# Define your application tool map
TOOL_REGISTRY: Dict[str, Callable[[str], str]] = {
    "GREP_SYMBOL": lambda q: f"Executing ripgrep search for '{q}'",
    "READ_FILE_CHUNK": lambda q: f"Reading file chunk for query '{q}'",
    "RUN_BUILD_TEST": lambda q: f"Triggering local test runner for '{q}'",
    "INSPECT_GIT_DIFF": lambda q: f"Inspecting unstaged git diffs for '{q}'",
}

def route_and_execute(user_query: str) -> Dict[str, Any]:
    """
    Evaluates user query using resident Axobrier agent_router cartridge.
    If confidence >= 0.90, dispatches immediately with zero LLM API calls.
    Falls back to a frontier LLM only when intent is ambiguous.
    """
    decision = axo.route_tool(user_query)
    tool_name = decision["selected_tool"]
    confidence = decision["confidence"]
    latency_us = decision["latency_us"]

    print(f"⚡ Axobrier Reflex: {tool_name} (Confidence: {confidence*100:.1f}%, Latency: {latency_us:.1f} µs)")

    # 1. High-confidence reflex path (Microsecond execution, $0.00 cost)
    if confidence >= 0.90 and tool_name in TOOL_REGISTRY:
        tool_fn = TOOL_REGISTRY[tool_name]
        result = tool_fn(user_query)
        return {
            "mode": "local_reflex",
            "tool": tool_name,
            "confidence": confidence,
            "latency_us": latency_us,
            "result": result
        }

    # 2. Ambiguity fallback path (Escalate to Claude / GPT-4o only when necessary)
    print("⚠️  Low confidence or ambiguous query. Escalating to frontier LLM...")
    return {
        "mode": "cloud_llm_escalation",
        "tool": "LLM_FALLBACK",
        "confidence": confidence,
        "query": user_query
    }


if __name__ == "__main__":
    queries = [
        "Search codebase with ripgrep for symbol 'axo_cartridge_load_mmap'",
        "Read lines 1 to 50 of include/axobrier_core.h",
        "Run pytest on tests/test_production_cartridges.py",
        "What is the philosophical meaning of determinism?" # Ambiguous / general
    ]

    for q in queries:
        print("\nInput:", q)
        res = route_and_execute(q)
        print("Output:", res)
```

---

## 3. LangChain `RunnableLambda` Integration

If you are using LangChain Expression Language (LCEL), drop Axobrier directly into your runnable chain:

```python
from langchain_core.runnables import RunnableLambda
import axobrier_reflex as axo

def axo_router_node(state: dict) -> str:
    """Classifies user message into next destination tool branch."""
    user_prompt = state["messages"][-1].content
    decision = axo.route_tool(user_prompt)
    
    if decision["confidence"] > 0.90:
        return decision["selected_tool"]
    return "general_llm_conversation"

# Compose into an LCEL pipeline:
axo_router_runnable = RunnableLambda(axo_router_node)
```

---

## 4. Cost & Latency Benchmark

| Workload (100,000 Tool Invocations) | Cloud LLM Router (GPT-4o mini / Claude Haiku) | Axobrier Resident Reflex Router |
| :--- | :--- | :--- |
| **Total Turnaround Time** | 33.3 Hours (1,200 ms / call) | **4 Seconds** (40 µs / call) |
| **API Token Invoices** | ~$150.00  to  $300.00 | **$0.00** |
| **Rate Limit Failures** | Occasional HTTP 429 errors | **0 Failures** (100% Deterministic) |
