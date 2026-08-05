# ❓ Frequently Asked Questions & Troubleshooting — SupportSphere AI

Welcome to the **SupportSphere AI FAQ & Troubleshooting Guide**. This document answers common technical questions and provides quick fixes for common development and deployment issues.

---

## 📋 Frequently Asked Questions

### 1. What LLM providers are supported?
SupportSphere AI natively supports three major providers:
- **OpenAI** (`gpt-5.4-mini`, `gpt-4o-mini`, `gpt-5`)
- **Google Gemini** (`gemini-2.5-flash-lite`, `gemini-3.1-flash-lite`, `gemini-1.5-flash`)
- **Groq** (`llama-3.3-70b-versatile`)

You can select separate providers for the **Decision Engine** (LLM #1) and **Generation Engine** (LLM #2).

---

### 2. How does tenant isolation work?
When a customer ticket arrives, the Decision Engine identifies the tenant company (e.g. `HackerRank`, `Claude`, `Visa`). The Retrieval Engine filters Pinecone vector search metadata by `company == tenant`. Node 3 (Retrieval Validation) acts as a strict zero-token gate, ensuring no document chunk from another company is passed to the generation prompt.

---

### 3. What embedding model is used for document retrieval?
The vector pipeline uses Google's `gemini-embedding-001` model (768 dimensions) to create dense embeddings for Pinecone cloud vector search.

---

## 🛠️ Troubleshooting Guide

### Error: `Permission denied: /home/ubuntu/supportsphere-ai/logs/application.log`
- **Cause**: Container process user (`appuser` UID 1001) created log files that host SSH user (`ubuntu` UID 1000) tries to modify.
- **Fix**: Run `sudo chmod -R 777 ~/supportsphere-ai/data ~/supportsphere-ai/logs` on the host server. (This step is automated in `.github/workflows/cd.yml`).

---

### Error: `No space left on device` during CD deployment on EC2
- **Cause**: Docker accumulated unused build layers, stopped containers, or old images.
- **Fix**: Run the automated storage cleanup commands on EC2:
  ```bash
  docker system prune -af --volumes
  docker builder prune -af
  sudo apt-get clean && sudo rm -rf /var/lib/apt/lists/*
  ```

---

### Error: `Missing API Key for Decision Provider` in Streamlit UI
- **Cause**: The selected provider (e.g. Google) does not have an API key configured in `.env` or the sidebar UI.
- **Fix**: Open the **API Credentials** expander in the Streamlit left sidebar and paste your active API key.

---

### Error: `404 NOT_FOUND` for Google Generative AI Embeddings
- **Cause**: Passing `models/text-embedding-004` to newer `google-genai` SDK versions causing URL prefixing issues.
- **Fix**: Use `model="gemini-embedding-001"` in `GoogleGenerativeAIEmbeddings`.

---

## 🔄 Re-ingesting Documentation & Disaster Recovery

If the Pinecone vector index is deleted or updated documentation files are added to `data/raw/`, run the ingestion script to re-embed and rebuild indexes:

```bash
uv run python -m vectordb_ingestion.ingestion.ingest
```

This script automatically:
1. Parses markdown files in `data/raw/`.
2. Generates embeddings using `gemini-embedding-001`.
3. Upserts dense vectors into Pinecone.
4. Serializes the BM25 index to `data/artifacts/bm25_index.pkl`.
