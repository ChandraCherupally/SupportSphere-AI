# Antigravity-Implementation-Spec.md

# SupportSphere AI – Architecture Freeze (v1.0)

## Status
**ARCHITECTURE FROZEN**

This document is the single source of truth for implementing the next version of SupportSphere AI.

## Objectives

- Keep the architecture simple (KISS).
- Follow SOLID principles.
- Do not rewrite stable modules.
- Improve workflow, routing and evaluation.
- Preserve the existing retrieval pipeline.

---

## Do NOT Modify

The following modules are considered stable and should only receive bug fixes:

- src/retrieval/bm25_index.py
- src/retrieval/vector_search.py
- src/retrieval/hybrid_search.py
- src/retrieval/rank_fusion.py
- src/retrieval/context_builder.py
- src/retrieval/rerankers/*
- pipeline/ingestion/*
- pipeline/preprocessing/*

---

## Files to Modify

- src/graph/graph.py
- src/graph/nodes.py
- src/graph/state.py
- src/graph/guardrails.py
- src/graph/support_agent.py
- src/ai/prompt_builder.py
- evaluation/*
- src/app.py

Avoid creating unnecessary packages or files.

---

## Target Workflow

START
→ Input Validation
→ Decision Gate
    ├── Retrieval Required
    │      → Retrieve
    │      → Retrieval Validation
    └── Retrieval Not Required
→ Generate
→ Output Validation
→ END

Decision Gate determines:
- retrieval_required
- normalized_issue
- normalized_subject
- routing_reason
- confidence

Use deterministic rules first.
Use the LLM only when rules cannot confidently classify the request.

---

## Decision Rules

Skip retrieval for:
- Greetings
- Thank-you messages
- Obvious out-of-scope questions
- Operational incidents that always escalate
- Empty or invalid requests

Run retrieval for:
- Product questions
- Troubleshooting
- Documentation lookup
- Feature questions
- Account and workflow guidance

---

## SupportState additions

Add only:

- retrieval_required: bool
- normalized_issue: str
- normalized_subject: str
- routing_reason: str
- confidence: float

Do not redesign the state model.

---

## Prompt Builder

Support two modes:

Knowledge Mode:
- Include retrieved context.

Routing Mode:
- Do not include retrieval context.
- Generate directly using routing decision.

---

## Evaluation

Split evaluation into five sections.

1. Classification
- Request Type Accuracy
- Product Area Accuracy
- Status Accuracy
- Precision / Recall / F1

2. Routing
- Retrieval Decision Accuracy
- Escalation Accuracy
- Out-of-Scope Accuracy
- Greeting Accuracy

3. Knowledge (RAGAS)
- Context Precision
- Context Recall
- Faithfulness
- Answer Correctness
- Answer Relevancy

Only evaluate RAGAS when retrieval_required == True.

4. Generation
Aggregate generation quality from RAGAS metrics.

5. System
- Average Latency
- Average Retrieved Chunks
- Retrieval Skip Rate
- Token Usage (if available)

---

## Streamlit

Keep a single src/app.py.

Do not redesign the application.

Improve:
- Routing metrics
- Knowledge metrics
- KPI cards
- Progress reporting
- Configuration visibility

Evaluation controls remain inside the Evaluation tab.

---

## Coding Standards

- Python 3.12
- Type hints
- Google docstrings
- SOLID
- KISS
- DRY
- Logging
- Small reusable functions
- No duplicated logic

---

## Acceptance Criteria

- Existing retrieval pipeline remains functional.
- Top-K configuration propagates correctly.
- Retrieval is skipped when unnecessary.
- Prompt builder supports retrieval/no-retrieval flows.
- RAGAS runs only for knowledge tickets.
- Routing metrics are reported separately.
- Streamlit dashboard reflects the new evaluation structure.
- Existing functionality remains backward compatible.
