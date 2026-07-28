# Frequently Asked Questions (FAQ) — SupportSphere AI

> [!NOTE]
> **Document Header Metadata**  
> **Document Title**: SupportSphere AI Frequently Asked Questions (FAQ)  
> **Purpose**: Serves as an exhaustive technical reference containing implementation-specific answers to common architectural, operational, debugging, and configuration questions.  
> **Audience**: AI Engineers, Software Developers, System Architects, DevOps Engineers, Support Managers.  
> **Last Updated**: `2026-07-28`  
> **Related Documents**: [README](../README.md) | [Architecture](ARCHITECTURE.md) | [System Design](SYSTEM_DESIGN.md) | [Retrieval Pipeline](RETRIEVAL_PIPELINE.md) | [Decision Engine](DECISION_ENGINE.md) | [Generation Engine](GENERATION_ENGINE.md) | [Evaluation](EVALUATION.md) | [Guardrails](GUARDRAILS.md) | [Observability](OBSERVABILITY.md) | [Deployment](DEPLOYMENT.md) | [Configuration](CONFIGURATION.md) | [Contributing](CONTRIBUTING.md) | [Changelog](CHANGELOG.md)

---

## Table of Contents

- [1. Architecture & Design Rationale](#1-architecture--design-rationale)
- [2. Retrieval & Knowledge Base](#2-retrieval--knowledge-base)
- [3. Decision Engine & Triage](#3-decision-engine--triage)
- [4. Guardrails & Safety](#4-guardrails--safety)
- [5. Generation Engine & Prompts](#5-generation-engine--prompts)
- [6. Evaluation & Benchmarking](#6-evaluation--benchmarking)
- [7. Observability & Logging](#7-observability--logging)
- [8. Configuration & Customization](#8-configuration--customization)
- [9. Debugging & Troubleshooting](#9-debugging--troubleshooting)
- [10. Deployment & Operations](#10-deployment--operations)
- [Related Reading](#related-reading)
- [References](#references)
- [Future Improvements](#future-improvements)

---

## 1. Architecture & Design Rationale

#### Q1.1: Why use LangGraph instead of a standard linear LangChain chain?
LangGraph provides a formal **stateful state machine** (`StateGraph`) with explicit conditional edges. Unlike linear chains, LangGraph allows conditional branching (e.g. bypassing retrieval for static greetings), loop execution (candidate scaling loops), and step-level state inspection via `ExecutionTrace`.

#### Q1.2: Why employ a Dual-LLM architecture instead of a single LLM call?
Combining triage and answer generation into one LLM prompt forces the model to perform security checks, intent classification, tenant verification, and answer synthesis simultaneously. Decoupling into **LLM #1 (Decision Engine)** and **LLM #2 (Generation Engine)** allows LLM #1 to filter out non-retrieval tickets early, saving ~70% of inference token costs on routine queries.

#### Q1.3: Why use zero-token rules-only nodes between LLM stages?
Nodes 3 (`retrieval_validation`) and 5 (`response_validation`) use pure Python helper functions (`verify_company`, `calculate_grounding_score`) rather than LLM evaluators. This eliminates $0.005+$ USD per-evaluation cost and executes in $< 5\text{ms}$ versus $2,000\text{ms}+$ for LLM judges.

#### Q1.4: Why select Streamlit for the user interface?
Streamlit allows rapid Python-native UI rendering, session state management (`st.session_state`), and interactive tab components without requiring a separate JavaScript frontend framework.

#### Q1.5: How does SupportSphere AI enforce state immutability across graph nodes?
Nodes do not mutate global variables; they receive a copy of `SupportState`, update specific keys, append an immutable `StageRecord` to `execution_trace`, and return the updated state dictionary.

---

## 2. Retrieval & Knowledge Base

#### Q2.1: Why Hybrid Search instead of pure vector search?
Pure vector search relies on dense semantic distance (`gemini-embedding-001`). It excels at natural language paraphrasing, but misses exact technical keywords (error codes, specific API query params). Parallel BM25 search guarantees keyword match precision, while vector search guarantees semantic recall.

#### Q2.2: Why use Reciprocal Rank Fusion (RRF) instead of score normalization?
Cosine similarity scores and BM25 relevance scores operate on different numerical scales. RRF merges rank positions ($RRF\_Score = \sum \frac{1}{60 + rank}$) without needing brittle score normalization multipliers.

#### Q2.3: Why is `FINAL_TOP_K` set to 5 by default?
Injecting more than 5 chunks increases prompt context size, introducing context noise that degrades RAGAS Faithfulness by ~8% and increasing prompt token costs without improving answer recall.

#### Q2.4: Why use FlashRank for reranking?
FlashRank executes local ONNX cross-encoder reranking (`ms-marco-MiniLM-L-12-v2`) on CPU in ~30ms, eliminating additional cloud network latency and third-party API costs.

#### Q2.5: How does the candidate deduplication algorithm work?
`HybridSearch.search()` enforces 3-way deduplication: ID uniqueness check, full text string uniqueness check, and a per-URL cap (`MAX_CHUNKS_PER_URL = 2`) to ensure diverse context sources.

#### Q2.6: What happens if deduplication drops candidate chunks below Top-K?
`HybridSearch` triggers a candidate expansion scaling loop, doubling the query search limit ($20 \rightarrow 40 \rightarrow 80$) for up to 3 attempts until `FINAL_TOP_K` unique chunks are satisfied.

#### Q2.7: Why use header-aware markdown chunking instead of character splitting?
Splitting documentation strictly by character count breaks procedural step lists across chunk boundaries. `HeaderChunker` splits text at markdown headers (`#`, `##`), maintaining topic coherence.

#### Q2.8: How are raw markdown files stored?
Raw support documentation files are stored under domain folders in `data/raw/` (e.g. `data/raw/claude/`, `data/raw/hackerrank/`, `data/raw/visa/`).

#### Q2.9: What embedding model and dimension does the vector store use?
Google Gemini `gemini-embedding-001` with `768` vector dimensions and cosine similarity metric.

#### Q2.10: Where is the BM25 index stored?
The Rank-BM25 index is serialized and stored as a pickle binary at `data/artifacts/bm25_index.pkl`.

---

## 3. Decision Engine & Triage

#### Q3.1: What are the 11 intent classes recognized by LLM #1?
`GREETING`, `THANK_YOU`, `PRODUCT_SUPPORT`, `ACCOUNT`, `BILLING`, `BUG`, `PLATFORM_INCIDENT`, `FEATURE_REQUEST`, `OUT_OF_SCOPE`, `SAFETY_REFUSAL`, `EMPTY_INPUT`.

#### Q3.2: Which intent classes require vector retrieval?
Only `PRODUCT_SUPPORT`, `ACCOUNT`, `BILLING` (general inquiries), and `FEATURE_REQUEST` set `retrieval_required = True`. All other intents use static template responses.

#### Q3.3: How does domain company reference resolution work?
Node 1 checks the ticket issue description for enterprise domain signals:
- **HackerRank**: assessment, candidate, test, question bank, coding test.
- **Claude**: Anthropic, Sonnet, Haiku, Opus, Bedrock.
- **Visa**: Visa, Traveller Cheque, Commercial Card, Visa Direct, Merchant.

#### Q3.4: What happens if the detected company conflicts with the selected company hint?
Node 1 sets `company_match = False` and logs a warning. Node 3 (`verify_company`) subsequently blocks any retrieved context that does not match the target ticket company.

#### Q3.5: What output schema does LLM #1 use?
Output is constrained to the `DecisionResult` Pydantic model (`safe`, `intent`, `company`, `routing_strategy`, `retrieval_required`, `reason`).

---

## 4. Guardrails & Safety

#### Q4.1: Why use NVIDIA NeMo Guardrails?
NeMo Guardrails provides a programmable policy framework (Colang 2.0) for evaluating input safety before sending user prompts to LLMs.

#### Q4.2: What Colang policies are configured in `rails.co`?
Prompt injection detection, jailbreak prevention, PII leakage protection (passwords, API keys, SSNs), roleplay attack prevention, and spam filtering.

#### Q4.3: How does `verify_company()` custom action prevent data leaks?
Node 3 programmatically iterates through all retrieved context chunks and checks `chunk.metadata.company` against the ticket's verified company. If any mismatch is found, it returns `BLOCK`.

#### Q4.4: How does `calculate_grounding_score()` output gate work?
Node 5 computes word-overlap ratio between response text and context chunks. If `grounding_score < 0.15`, the response is blocked and replaced with `SAFETY_REFUSAL_TEMPLATE`.

#### Q4.5: What token cost is incurred when a guardrail blocks a request?
Zero LLM tokens ($0.00 cost) because blocked requests route to static template strings in `src/responses/templates.py`.

---

## 5. Generation Engine & Prompts

#### Q5.1: What system prompt is used by LLM #2?
Prompt v3 ([src/ai/prompts/generation_prompt.py](file:///c:/Users/cheru/OneDrive/Desktop/GitHub_Projects/SupportSphere-AI/src/ai/prompts/generation_prompt.py)) optimized for RAGAS metrics.

#### Q5.2: How does the Generation Engine enforce anti-hallucination?
The system prompt explicitly states: *"You are a deterministic retrieval-augmented support engine. Never answer from memory. Never invent information."*

#### Q5.3: What evidence hierarchy does LLM #2 follow?
1. Exact feature match $\rightarrow$ 2. Exact workflow match $\rightarrow$ 3. Same product area $\rightarrow$ 4. Supporting context.

#### Q5.4: How are documentation citations formatted?
All responses must include inline citations and append the authoritative documentation URL extracted from chunk metadata.

#### Q5.5: When does LLM #2 escalate a ticket?
If documentation is missing, contradictory, or requires backend database investigation, LLM #2 sets `status = "Escalated"` and outputs human escalation instructions.

---

## 6. Evaluation & Benchmarking

#### Q6.1: What metrics are evaluated in RAGAS?
Faithfulness, Answer Relevancy, Answer Correctness, Context Precision, and Context Recall.

#### Q6.2: How are LLM token costs calculated?
Input/output tokens per stage are multiplied by rates in `src/billing/catalog.py` plus `$0.00002` per Pinecone vector query.

#### Q6.3: What is a pricing snapshot?
Each evaluation run embeds a snapshot of `catalog.py` rates (`PRICING_VERSION = "2026-07-18"`) into the experiment JSON file so historical cost calculations remain reproducible.

#### Q6.4: Where are evaluation benchmark files stored?
In `data/experiments/` as `experiment_<timestamp>.json`.

#### Q6.5: How do I compare multiple evaluation runs?
Open Streamlit Tab 4 (Evaluation) to view multi-axis radar charts and side-by-side metric comparison tables.

---

## 7. Observability & Logging

#### Q7.1: Where are log files written?
Log files are written to `logs/`: `application.log` (workflow milestones & audit summaries), `pipeline.log` (stage telemetry), and `errors.log` (exceptions).

#### Q7.2: What is the purpose of `FrameworkFilter`?
Suppresses noisy third-party frameworks (`nemoguardrails`, `pinecone`, `httpx`) and normalizes `src.*` logger names to `"SupportSphere"`.

#### Q7.3: What is a Ticket Execution Summary Block?
A formatted 55-line audit summary printed to `application.log` after every ticket completes, showing tokens, latency, cost, and decisions per node.

#### Q7.4: How does `ExecutionTrace` work?
Each graph node appends a `StageRecord` dataclass to `state["execution_trace"]`, which feeds the Streamlit timeline inspector.

#### Q7.5: Is LangSmith tracing supported?
Yes. Set `LANGCHAIN_TRACING_V2=true` in `.env` to stream traces directly to LangSmith.

---

## 8. Configuration & Customization

#### Q8.1: How do I add a new LLM provider?
Add pricing rates to `_LLM_CATALOG` in `src/billing/catalog.py` and extend `LLMClient._create_client()` in `src/ai/client.py`.

#### Q8.2: How do I add support for a new company tenant?
Add raw markdown files to `data/raw/<company>/`, update domain keywords in `decision_prompt.py`, and run `uv run python -m vectordb_ingestion.ingestion.ingest`.

#### Q8.3: How do I change retrieval Top-K dynamically?
Adjust the "Top-K Chunks" slider in the Streamlit sidebar, or set `FINAL_TOP_K` in `.env`.

#### Q8.4: How do I add a custom guardrail rule?
Add Colang flow definitions in `src/guardrails/config/rails.co` and register custom Python actions in `src/guardrails/config/actions.py`.

#### Q8.5: How do I change the default reranker?
Select a reranker in the Streamlit sidebar (None, Cross-Encoder, FlashRank, LLM) or set `RERANKER` in `.env`.

---

## 9. Debugging & Troubleshooting

#### Q9.1: How do I debug retrieval quality?
Inspect the `retrieval_trace` dictionary returned in `AgentResponse` or check Node 2 debug logs in `logs/pipeline.log`.

#### Q9.2: How do I debug NeMo Guardrail policy blocks?
Check `logs/application.log` for `[Node 1: Decision Engine]` entries showing `check_input()` policy results.

#### Q9.3: What should I do if Pinecone returns connection errors?
Verify `PINECONE_API_KEY` and `PINECONE_INDEX_NAME` in `.env`. The retrieval engine will gracefully fall back to BM25-only search.

#### Q9.4: How do I resolve Windows `WinError 10054` warnings?
This socket noise is automatically patched in `src/app.py` via `asyncio.proactor_events` patching.

#### Q9.5: Why does Streamlit show empty charts on first launch?
Ensure at least one benchmark evaluation run has been saved to `data/experiments/`.

---

## 10. Deployment & Operations

#### Q10.1: How do I run unit tests?
Execute `uv run pytest` from the project root.

#### Q10.2: How do I run the application using Docker?
Build and run via Docker Compose: `docker compose up -d --build`.

#### Q10.3: How do I deploy to Google Cloud Run?
Submit container to Google Artifact Registry and deploy using `gcloud run deploy`.

#### Q10.4: What container port is exposed?
Port `8501`.

#### Q10.5: What health check endpoint is available?
`http://localhost:8501/_stcore/health` (returns `200 OK`).

---

## Related Reading

- [README](../README.md) — Project overview & documentation index.
- [Architecture Guide](ARCHITECTURE.md) — 5-node state machine topology.
- [System Design Specification](SYSTEM_DESIGN.md) — Design patterns & principles.
- [Configuration Guide](CONFIGURATION.md) — Environment variables matrix.

---

## References

1. SupportSphere AI Source Codebase & Configuration Modules.
2. LangGraph, LangChain, NeMo Guardrails, Pinecone, and RAGAS Documentation.

---

## Future Improvements

1. **Search Bar in FAQ**: Add client-side fuzzy text search to the Streamlit FAQ UI tab.
