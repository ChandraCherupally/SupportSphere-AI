# SupportSphere AI — Developer's Guide

This guide is designed for developers who want to extend, configure, or debug the SupportSphere AI support triage workflow.

---

## 1. How to Add a New Workflow Stage

Adding a new stage (e.g., `summarizer` or `planner`) is designed to be highly extensible and does not require modifying existing classes.

### Step 1: Add Configuration in `src/config.py`
Expose environment variables for the new stage's provider and model, and register it in `LLM_CONFIG`:

```python
# 1. Declare variables
SUMMARIZER_PROVIDER = os.getenv("SUMMARIZER_PROVIDER", "google").lower()
SUMMARIZER_MODEL = os.getenv("SUMMARIZER_MODEL", "gemini-2.5-flash")

# 2. Add validation checks
if SUMMARIZER_PROVIDER not in SUPPORTED_LLM_PROVIDERS:
    raise ValueError(f"Unsupported SUMMARIZER_PROVIDER: {SUMMARIZER_PROVIDER}")

# 3. Add to LLM_CONFIG registry mapping
LLM_CONFIG = {
    ...
    "summarizer": {
        "provider": SUMMARIZER_PROVIDER,
        "model": SUMMARIZER_MODEL,
    }
}
```

### Step 2: Invoke the Stage in a Node
In your LangGraph node implementation, import the shared `llm` client and call `generate` specifying the new stage name:

```python
from src.graph.nodes import llm

def summarize_node(state: SupportState) -> dict:
    prompt = [("system", "Summarize this ticket..."), ("human", state["issue"])]
    
    # Executing client generate resolves configurations and tracks tokens automatically
    result = llm.generate(prompt, stage="summarizer")
    
    return {
        "summary": result.response,
        "summarizer_input_tokens": result.usage.input_tokens,
        "summarizer_output_tokens": result.usage.output_tokens,
    }
```

---

## 2. Coding Guidelines & Standards

### 2.1 Single Responsibility Principle (SRP)
- **Do not mix LLM execution with node business logic**. Nodes must only specify `stage='stage_name'` and call `llm.generate()`.
- **Do not parse usage metadata inside nodes**. All token usage extraction, latency measurements, and provider model queries happen inside `LLMClient.generate()`.

### 2.2 Standard Logging
- **NEVER use raw `print()` statements** in system code.
- Import `logging` and use:
  - `logger.debug()` for trace logs (optional).
  - `logger.info()` for workflow tracking.
  - `logger.warning()` for non-critical warnings.
  - `logger.error()` for exceptions and errors.

### 2.3 State Cleanliness
- Keep `SupportState` minimal and typed.
- Never store client instances, prompt objects, database connections, or provider-specific data structures in the state. Store only primitive types, dataclasses, or lists.

---

## 3. Running & Verifying Changes

### 3.1 Local Script Run
Test that all modules import and execute on sample tickets:
```bash
uv run python main.py
```

### 3.2 Running the UI Dashboard
Start the Streamlit development server locally:
```bash
uv run streamlit run src/app.py
```
This will open the interface exposing the separate **Decision** and **Generation** provider and model selectors.
