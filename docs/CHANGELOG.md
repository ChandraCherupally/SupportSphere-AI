# Changelog & Release Notes — SupportSphere AI

> [!NOTE]
> **Document Header Metadata**  
> **Document Title**: SupportSphere AI Changelog & Release Notes  
> **Purpose**: Documents all notable changes, feature additions, structural updates, bug fixes, security enhancements, and deprecations across versions of SupportSphere AI following the Keep a Changelog standard and Semantic Versioning (SemVer).  
> **Audience**: Developers, Systems Administrators, DevOps Engineers, Release Managers, End Users.  
> **Last Updated**: `2026-07-28`  
> **Related Documents**: [README](../README.md) | [Architecture](ARCHITECTURE.md) | [System Design](SYSTEM_DESIGN.md) | [Retrieval Pipeline](RETRIEVAL_PIPELINE.md) | [Decision Engine](DECISION_ENGINE.md) | [Generation Engine](GENERATION_ENGINE.md) | [Evaluation](EVALUATION.md) | [Guardrails](GUARDRAILS.md) | [Observability](OBSERVABILITY.md) | [Deployment](DEPLOYMENT.md) | [Configuration](CONFIGURATION.md) | [Contributing](CONTRIBUTING.md) | [FAQ](FAQ.md)

---

## 1. Versioning Standards & Principles

All versions of **SupportSphere AI** strictly adhere to **Semantic Versioning 2.0.0** (`MAJOR.MINOR.PATCH`):

- **MAJOR (`X.0.0`)**: Incompatible API or architectural changes breaking backwards compatibility.
- **MINOR (`0.Y.0`)**: Backwards-compatible new features, retrieval algorithms, or LLM provider additions.
- **PATCH (`0.0.Z`)**: Backwards-compatible bug fixes, logging adjustments, or documentation updates.

Changes are grouped into six standard categories defined by [Keep a Changelog](https://keepachangelog.com/en/1.0.0/):
- **`Added`**: New features, providers, or workflows.
- **`Changed`**: Modifications to existing functionality.
- **`Deprecated`**: Soon-to-be removed features.
- **`Removed`**: Removed features.
- **`Fixed`**: Bug fixes and error resolutions.
- **`Security`**: Security enhancements, vulnerability patches, or guardrails updates.

---

## 2. Changelog Entries

### [Unreleased]

#### Added
- Future SSE streaming token generator for Streamlit UI.
- Async document ingestion queue using Celery and Redis.

---

### [0.1.0] - 2026-07-28

> **Initial Production Release**  
> Production multi-tenant support triage and grounded RAG platform featuring a stateful 5-node LangGraph pipeline, dual-LLM architecture, NeMo Guardrails safety policies, Hybrid Retrieval (BM25 + Pinecone Vector Search), RAGAS benchmark evaluator, and 4-tab Streamlit dashboard.

#### Added
- **LangGraph Stateful 5-Node Workflow**: Stateful state machine topology (`decision_engine` $\rightarrow$ `retrieval` $\rightarrow$ `retrieval_validation` $\rightarrow$ `generation` $\rightarrow$ `response_validation`) with conditional routing.
- **Dual-LLM Architecture**: Decoupled reasoning (LLM #1 — Decision Engine) from synthesis (LLM #2 — Generation Engine) across OpenAI, Google Gemini, and Groq providers.
- **NVIDIA NeMo Guardrails Integration**: Colang 2.0 policy execution screening for prompt injection, jailbreaks, PII leakage, and abusive language.
- **Hybrid Retrieval Engine**: Parallel Rank-BM25 keyword search and Pinecone dense vector similarity search (`gemini-embedding-001`), fused via Reciprocal Rank Fusion (RRF).
- **Plug-and-Play Reranker Factory**: Reranking factory supporting FlashRank (local ONNX), CrossEncoder (`ms-marco-MiniLM-L-6-v2`), LLM-rescoring, and NoOp passthrough.
- **Zero-Token Rules Validation Nodes**: Node 3 (`verify_company`, `filter_duplicate_chunks`) and Node 5 (`calculate_grounding_score`) executing pure Python rules at `$0.00` token cost.
- **RAGAS Benchmark Evaluator**: Evaluation framework tracking Faithfulness, Answer Relevancy, Answer Correctness, Context Precision, and Context Recall across ground truth datasets.
- **Experiment Manager & Pricing Snapshots**: JSON experiment persistence embedding versioned pricing catalogs (`2026-07-18`) for reproducible cost tracking.
- **Node-Attributed Logging**: Multi-file logging architecture (`application.log`, `pipeline.log`, `errors.log`) with explicit node context tags (`[Node 1: Decision Engine]`).
- **4-Tab Streamlit Dashboard**: Overview Dashboard, Single Ticket Triage Inspector, Batch CSV Processing, and RAGAS Benchmark Evaluator with live sidebar runtime configuration.
- **Comprehensive Unit Test Suite**: 18 automated unit tests covering retrieval modes, company detection, model validation, safety guardrails, and evaluators.

#### Changed
- Standardized all application loggers to `logger = logging.getLogger("SupportSphere")` and implemented `FrameworkFilter` for clean log outputs.
- Renamed ingestion package from `vectordb_ingestion_pipeline` to `vectordb_ingestion` and updated all import statements across `src/` and `vectordb_ingestion/`.
- Updated system prompt v3 in `generation_prompt.py` for RAGAS metric optimization.

#### Fixed
- Fixed text invisibility in Streamlit Dark Mode by adding explicit high-contrast node label styling (`color:#...`) to Mermaid workflow diagrams in `dashboard.py`.
- Fixed `N/A` ticket subject and issue log formatting by introducing `normalized_subject` and `normalized_issue` fallbacks in `logger.py` and `support_agent.py`.
- Fixed Windows asyncio Proactor BasePipeTransport `WinError 10054` socket reset noise on application shutdown.

#### Security
- Added `verify_company()` custom action in Node 3 to programmatically block cross-tenant retrieved chunks before prompt injection.
- Added Colang 2.0 input safety rules blocking system prompt extraction, credential leakage, and persona subversion attacks.

---

## 3. Template for Future Release Entries

Contributors should use the following format when drafting release notes:

```markdown
### [X.Y.Z] - YYYY-MM-DD

> **Summary**: One-paragraph summary of the release.

#### Added
- Feature description...

#### Changed
- Change description...

#### Deprecated
- Deprecation notice...

#### Removed
- Removal notice...

#### Fixed
- Bug fix description...

#### Security
- Security patch description...
```

---

## Related Reading

- [README](../README.md) — Master project overview and documentation index.
- [System Architecture Guide](ARCHITECTURE.md) — Topology and design principles.
- [Contributing Guidelines](CONTRIBUTING.md) — Commit standards and PR process.

---

## References

1. Semantic Versioning 2.0.0 (`semver.org`).
2. Keep a Changelog 1.0.0 (`keepachangelog.com`).

---

## Future Improvements

1. **Automated Changelog Generator**: Integrate `git-cliff` or `semantic-release` to generate changelog entries automatically from conventional commit messages.
