# SupportSphere AI — SupportAgent Workflow Specification

This document details the step-by-step processing workflow of the SupportAgent state machine.

---

## 1. Workflow Sequence

The workflow is constructed as a LangGraph state machine. Below is the step-by-step execution path:

```
[START]
   │
   ▼
[Input Check Node] ────────────────► Runs input validation guardrails
   │
   ▼
[Decision Gate Node] ──────────────► Resolves retrieval required or skip
   │
   ├─► (if True) ──► [Retrieve Node] ──► [Retrieval Check Node] (runs guardrails)
   │                       │
   ▼                       ▼
[Generate Node] ───────────────────► Builds prompt and generates reply
   │
   ▼
[Output Check Node] ───────────────► Runs output validation guardrails
   │
   ▼
 [END]
```

---

## 2. Detailed Nodes Specification

### 2.1 Input Check
- **Purpose**: Validate input ticket properties before executing model inference.
- **Rules**: Checks if the issue text is too short, empty, or consists solely of spam/offensive content.
- **Output**: Appends appropriate warnings to the Graph State.

### 2.2 Decision Gate
- **Purpose**: Classify if documentation retrieval is required to answer the ticket.
- **Step 1: Deterministic Classifiers (Pre-LLM)**
  - **Empty/Invalid**: Instantly skips retrieval (confidence = 1.0).
  - **Unsupported Company**: Checks if company is HackerRank, Claude, or Visa. If not, skips retrieval (confidence = 1.0).
  - **Greetings / Small Talk**: Detects simple salutations (e.g. "Hi", "Hello") and thank-you messages. Skips retrieval (confidence = 1.0).
  - **Operational Outage**: Scans for outage keywords (e.g. "site is down", "service unavailable"). Skips retrieval (confidence = 1.0).
- **Step 2: LLM Classifier (Fallback)**
  - If deterministic checks pass, invokes the LLM client using the `decision` stage config (`gemini-2.5-flash-lite` by default) with the routing system instructions.
- **State Updates**:
  - `retrieval_required` (bool)
  - `routing_reason` (str)
  - `confidence` (float)
  - `decision_input_tokens` and `decision_output_tokens` (from `LLMResult.usage`)

### 2.3 Retrieve
- **Purpose**: Query documentation index.
- **Action**: Queries BM25 and Vector index namespaces matching the company name.
- **Reranker**: Executes FlashRank, CrossEncoder, or LLM reranker to select the top-K chunks.

### 2.4 Retrieval Check
- **Purpose**: Validate quality of retrieved chunks.
- **Rules**: Appends warnings if average similarity score is too low or if no relevant context chunks are returned.

### 2.5 Generate
- **Purpose**: Generate structural support reply.
- **Action**: Generates a structured response based on the `SupportResponse` Pydantic model.
- **Config**: Resolves the generation provider/model via `LLMRegistry` (defaults to `gemini-2.5-flash`).
- **State Updates**:
  - `response` (`SupportResponse` object)
  - `generation_input_tokens` and `generation_output_tokens` (from `LLMResult.usage`)

### 2.6 Output Check
- **Purpose**: Post-generation guardrails.
- **Rules**: Scans output reply for forbidden keywords, hallucinations, formatting errors, or empty answers.
