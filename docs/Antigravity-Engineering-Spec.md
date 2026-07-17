# Antigravity-Engineering-Spec.md

# SupportSphere AI – Engineering Implementation Blueprint
Version: 1.0 (Architecture Frozen)

> This document is the implementation blueprint for Antigravity CLI.
> Follow this specification exactly. Do not redesign the architecture.

---

# 1. Engineering Principles

- Keep the architecture simple (KISS).
- Follow SOLID.
- Preserve backward compatibility.
- Prefer modifying existing modules over creating new ones.
- Avoid unnecessary abstractions.
- One responsibility per function.
- Type hints everywhere.
- Python 3.12.

---

# 2. Modules That MUST NOT Be Rewritten

These modules are considered production-ready.

- src/retrieval/bm25_index.py
- src/retrieval/vector_search.py
- src/retrieval/hybrid_search.py
- src/retrieval/rank_fusion.py
- src/retrieval/context_builder.py
- src/retrieval/rerankers/*
- pipeline/ingestion/*
- pipeline/preprocessing/*

Bug fixes only.

---

# 3. Modules To Modify

Required

- src/graph/graph.py
- src/graph/nodes.py
- src/graph/state.py
- src/graph/guardrails.py
- src/graph/support_agent.py
- src/ai/prompt_builder.py
- evaluation/*
- src/app.py

Optional

- src/ai/models.py (only if additional response fields are required)

---

# 4. Target Workflow

START

↓

Input Validation

↓

Decision Gate

↓

Need Retrieval?

YES → Retrieve → Retrieval Validation

NO  → Generate

↓

Generate

↓

Output Validation

↓

END

Decision Gate is the only new workflow concept.

---

# 5. Decision Gate Responsibilities

Implement inside graph/nodes.py.

Pseudo Flow

1. Read issue, subject, company.
2. Normalize whitespace.
3. Apply deterministic routing rules.
4. If confidence is low:
   - Use existing LLM to classify.
5. Return:

- retrieval_required
- normalized_issue
- normalized_subject
- routing_reason
- confidence

Rules-first examples:

Greeting → Skip Retrieval

Thank You → Skip Retrieval

Out of Scope → Skip Retrieval

Known Outage → Skip Retrieval

Knowledge Question → Retrieval

---

# 6. SupportState

Extend SupportState only with

retrieval_required: bool

normalized_issue: str

normalized_subject: str

routing_reason: str

confidence: float

Do not redesign the state model.

---

# 7. Graph Changes

Insert Decision Gate after Input Validation.

Use conditional edges.

If retrieval_required:

Input
→ Decision
→ Retrieve
→ Generate

Else

Input
→ Decision
→ Generate

Output validation remains unchanged.

---

# 8. Retriever

Do not redesign Retriever.

Only change its input.

Current

Raw issue

Target

Normalized issue

Everything else remains unchanged.

---

# 9. Prompt Builder

Support two modes.

Knowledge Mode

- Include retrieved documents.

Routing Mode

- No retrieved context.
- Generate response using routing information.

No duplicated prompt builders.

---

# 10. Guardrails

Keep existing guardrails.

Only extend them if required to support routing decisions.

Do not move business logic into guardrails.

---

# 11. Evaluation Framework

Split evaluation.

Classification

- Request Type
- Product Area
- Status

Routing

- Retrieval Decision Accuracy
- Escalation Accuracy
- Out-of-Scope Accuracy
- Greeting Accuracy

Knowledge (RAGAS)

Run only if retrieval_required == True

Metrics

- Context Precision
- Context Recall
- Faithfulness
- Answer Correctness
- Answer Relevancy

Generation

Aggregate RAGAS generation metrics.

System

- Average Latency
- Average Retrieved Chunks
- Retrieval Skip Rate
- Token Usage (optional)

Never mix routing tickets into RAGAS averages.

---

# 12. Runner Changes

Pseudo-code

for sample:

    result = agent.invoke()

    classification.evaluate()

    if result.retrieval_required:
        ragas.evaluate()
    else:
        routing.evaluate()

aggregate()

generate_reports()

---

# 13. Streamlit

Do not redesign.

Keep one app.py.

Evaluation page

Sections

- Classification
- Routing
- Knowledge
- Generation
- System

Move all evaluation controls into Evaluation tab.

Sidebar only contains global configuration.

---

# 14. Logging

Log

- Decision outcome
- Retrieval skipped
- Retrieval executed
- Number of chunks
- RAGAS executed
- RAGAS skipped
- Routing metrics
- Runtime

---

# 15. Testing

Verify

✓ Greetings skip retrieval

✓ Out-of-scope skips retrieval

✓ Site outage skips retrieval

✓ Knowledge tickets retrieve

✓ Top-K respected

✓ RAGAS only runs for knowledge tickets

✓ Dashboard metrics correct

---

# 16. Definition of Done

The implementation is complete when:

- Architecture matches this document.
- Retrieval pipeline remains unchanged.
- Decision Gate works.
- Prompt Builder supports both modes.
- Evaluation separates Routing and Knowledge.
- Streamlit reflects new evaluation structure.
- No regression in existing functionality.
