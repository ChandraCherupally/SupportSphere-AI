# 🏗️ System Architecture Guide — SupportSphere AI

Welcome to the **SupportSphere AI System Architecture Guide**. This document explains the high-level design, state machine topology, hybrid retrieval pipeline, safety guardrails, and logging architecture.

---

## 📋 Table of Contents
1. [High-Level System Topology](#1-high-level-system-topology)
2. [LangGraph 5-Node State Machine](#2-langgraph-5-node-state-machine)
3. [Node Breakdown & Functionality](#3-node-breakdown--functionality)
4. [Hybrid Retrieval Pipeline](#4-hybrid-retrieval-pipeline)
5. [NVIDIA NeMo Guardrails & Safety](#5-nvidia-nemo-guardrails--safety)
6. [Logging & Observability](#6-logging--observability)

---

## 1. High-Level System Topology

SupportSphere AI is architected as a **stateless, multi-tenant customer support automation platform**. It handles incoming customer tickets across multiple companies (HackerRank, Claude, Visa) with strict tenant isolation.

```
[ Customer Support Ticket ]
            │
            ▼
 🧠 Node 1: Decision Engine (LLM #1 + NeMo Safety Screening)
   ├── Intent Classification (11 classes)
   ├── Tenant Routing & Guardrails Screening
   └── Strategy Selection (RAG vs Direct Template vs Safe Refusal)
            │
            ├───────────────────────┬────────────────────────┐
            ▼ (Needs RAG)           ▼ (Direct Template)      ▼ (Unsafe/Refused)
 🔍 Node 2: Retrieval Engine   📋 Static Template Response  🛡️ Refusal Template
   ├── Parallel BM25 Keyword Search
   ├── Pinecone Dense Vector Similarity (`gemini-embedding-001`)
   ├── RRF Rank Fusion & Deduplication
   └── Optional Reranker (Cross-Encoder / FlashRank)
            │
            ▼
 🧹 Node 3: Retrieval Validation (Pure Rules — Zero AI Tokens)
   ├── Tenant Boundary Gate (Verifies chunk company matches ticket)
   └── Minimum Chunk Threshold Gate
            │
            ▼
 ✨ Node 4: Generation Engine (LLM #2)
   ├── Grounded Answer Synthesis
   ├── Citation Formatting & Anti-Hallucination Directives
            │
            ▼
 🛡️ Node 5: Response Validation (Pure Rules — Zero AI Tokens)
   ├── Word-Overlap Grounding Gate (Threshold ≥ 0.15)
   └── Escalation Fallback (Triggered if answer is ungrounded)
            │
            ▼
 [ Final Verified Customer Response ]
```

---

## 2. LangGraph 5-Node State Machine

The pipeline is orchestrated as a directed state machine using **LangGraph**. The state is represented as a typed dictionary `SupportState` passed between nodes:

```python
class SupportState(TypedDict):
    ticket: SupportTicket
    decision: Optional[TicketDecision]
    retrieved_contexts: List[str]
    retrieval_validated: bool
    generated_response: Optional[str]
    response_validated: bool
    final_response: str
    execution_status: str
    token_usage: Dict[str, int]
    execution_time_ms: float
```

---

## 3. Node Breakdown & Functionality

### Node 1: Decision Engine (`decision_engine`)
- **Job**: Analyzes raw ticket text to determine safety, intent, category, product area, and routing action.
- **Providers Supported**: OpenAI, Google Gemini, Groq.
- **Safety Screening**: Screened via NVIDIA NeMo Guardrails (Colang 2.0 policy engine) to block prompt injections, jailbreaks, PII leakage, or abusive content.

### Node 2: Retrieval Engine (`retrieval`)
- **Job**: Fetches domain-specific documentation chunks matching the ticket issue.
- **Hybrid Search**: Combines BM25 keyword matching with Pinecone dense vector similarity search.
- **Fusion**: Merges candidate lists using Reciprocal Rank Fusion (RRF).
- **URL Capping**: Enforces max 2 chunks per source URL to ensure diverse context.

### Node 3: Retrieval Validation (`retrieval_validation`)
- **Job**: Purely rule-based gate (0 AI tokens spent).
- **Tenant Isolation**: Ensures retrieved document chunks strictly belong to the company specified in the ticket.

### Node 4: Generation Engine (`generation`)
- **Job**: Uses a second LLM to write a clear, professional answer grounded exclusively in the retrieved documentation.
- **Citation**: Appends exact source URLs for transparency.

### Node 5: Response Validation (`response_validation`)
- **Job**: Purely rule-based gate (0 AI tokens spent).
- **Grounding Gate**: Calculates word overlap between generated answer and retrieved context. If grounding score < 0.15, the response is blocked and routed to fallback escalation.

---

## 4. Hybrid Retrieval Pipeline

```
[ Query Text ] ──┬──> [ BM25 Keyword Search ] ─────────┐
                 │                                     ├──> [ RRF Fusion ] ──> [ Deduplication & Capping ] ──> [ Top-K Context ]
                 └──> [ Dense Vector Search (Pinecone)] ┘
```

1. **BM25 Search**: Matches exact keywords, codes, error messages, and terminology.
2. **Dense Vector Search**: Converts text into 768-dimensional embeddings via `gemini-embedding-001` to capture semantic intent.
3. **Reciprocal Rank Fusion (RRF)**: Scores items using $RRF\_Score(d) = \sum \frac{1}{60 + r(d)}$.
4. **URL Capping**: Prevents a single long documentation page from dominating all context slots.

---

## 5. NVIDIA NeMo Guardrails & Safety

Safety policies are defined using **Colang 2.0** rules in `src/guardrails/config/`:

- **Prompt Injection Defense**: Blocks attempts to override system prompts or bypass company boundaries.
- **PII Masking**: Filters credit card numbers, passwords, and sensitive credentials.
- **Off-Topic Refusal**: Politely declines requests unrelated to supported company products.

---

## 6. Logging & Observability

All system operations write structured logs to `logs/`:

- `logs/application.log`: High-level system initialization and configuration events.
- `logs/pipeline.log`: Node-by-node execution details tagged with `[Node 1: Decision Engine]`, `[Node 2: Retrieval]`, etc.
- `logs/errors.log`: Stack traces and exception logs.

At the end of every ticket execution, an **Audit Summary Block** is outputted:

```text
================================================================================
SUPPORTSPHERE AI — TICKET EXECUTION AUDIT SUMMARY
================================================================================
Company:            HackerRank
Request Type:       product_issue
Product Area:       community
Retrieval Mode:     hybrid (Top-K: 5)
Decision Provider:  google (gemini-2.5-flash-lite)
Generation Model:   google (gemini-3.1-flash-lite)
Execution Status:   SUCCESS
Execution Time:     3,842 ms
Estimated Cost:     $0.000782
================================================================================
```
