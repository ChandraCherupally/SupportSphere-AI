# 📊 Benchmark Evaluation & Configuration Guide — SupportSphere AI

Welcome to the **SupportSphere AI Evaluation & Configuration Guide**. This document summarizes our empirical performance findings across 46 automated experiments, explains the RAGAS evaluation metrics, and details configuration settings.

---

## 📋 Table of Contents
1. [Evaluation Framework Overview](#1-evaluation-framework-overview)
2. [RAGAS Quality Metrics Explained](#2-ragas-quality-metrics-explained)
3. [Key Performance Summary (46 Experiments)](#3-key-performance-summary-46-experiments)
4. [Top Recommended Configurations](#4-top-recommended-configurations)
5. [Configuration Specification (.env & Sidebar)](#5-configuration-specification-env--sidebar)

---

## 1. Evaluation Framework Overview

SupportSphere AI includes an automated benchmark evaluation suite using **RAGAS** (Retrieval-Augmented Generation Assessment). 

- **Evaluation Dataset**: `data/eval_dataset.csv` (contains ground-truth tickets, expected request types, product areas, and human expert reference responses).
- **RAGAS Judge LLM**: Configurable LLM judge (OpenAI GPT-5.4-mini or Google Gemini-2.5-flash-lite) that scores model outputs without human bias.
- **Experiment Manager**: Saves complete experiment snapshots to `data/experiments/exp_*.json`.

---

## 2. RAGAS Quality Metrics Explained

All RAGAS metrics are scored on a scale from **0.0 to 1.0** (higher is better):

| Metric | What It Measures | Target Score |
|---|---|---|
| **Faithfulness** | Does the generated answer contain *only* facts present in retrieved documents? (Anti-hallucination) | $\ge 0.80$ |
| **Answer Relevancy** | Is the answer directly relevant to the customer's question? | $\ge 0.80$ |
| **Context Precision** | Were the retrieved document chunks relevant to the question? | $\ge 0.70$ |
| **Context Recall** | Did the search find *all* information needed to answer the question? | $\ge 0.50$ |
| **Answer Correctness** | How closely does the AI answer match the human expert reference answer? | $\ge 0.65$ |
| **Avg RAGAS Score** | Overall composite score across all 5 metrics above | $\ge 0.75$ |

---

## 3. Key Performance Summary (46 Experiments)

Over **46 experiments** were conducted testing different LLM providers, retrieval modes, rerankers, and context sizes:

### Overall Score Range
- **Highest Observed RAGAS Score**: **`0.842`** (Gemini-2.5-Flash-Lite + Gemini-3.1-Flash-Lite, Vector + Cross-Encoder)
- **Fastest Average Latency**: **`3.83 seconds`**
- **Lowest Cost per Ticket**: **`$0.00062`**
- **Intent Classification Accuracy**: **`100%`** across nearly all experiments.

### Top 3 Experiments

| Rank | Experiment File | Config | Avg RAGAS | Faithfulness | Latency | Cost/Ticket |
|---|---|---|---|---|---|---|
| 🥇 **1st** | `exp_20260729_100146.json` | Gemini-2.5-Flash-Lite + Gemini-3.1-Flash-Lite · Vector + Cross-Encoder | **0.842** | 0.798 | 3.83s | $0.00078 |
| 🥈 **2nd** | `exp_20260729_095307.json` | Gemini-2.5-Flash-Lite + Gemini-3.1-Flash-Lite · Hybrid + Cross-Encoder | **0.840** | 0.861 | 4.05s | $0.00080 |
| 🥉 **3rd** | `exp_20260727_115012.json` | GPT-5.4-mini + Gemini-3.1-Flash-Lite · Hybrid + FlashRank | **0.826** | 0.816 | 6.36s | $0.00292 |

---

## 4. Top Recommended Configurations

### Recommended Production Setup (Best Quality + Low Cost)
```ini
DECISION_PROVIDER=google
DECISION_MODEL=gemini-2.5-flash-lite
GENERATION_PROVIDER=google
GENERATION_MODEL=gemini-3.1-flash-lite
SEARCH_MODE=hybrid
FINAL_TOP_K=5
RERANKER=cross_encoder
```
- **Performance**: RAGAS ~0.84, Latency ~4.0s, Cost ~$0.0008 / ticket.

---

## 5. Configuration Specification (.env & Sidebar)

### Core Environment Variables (`.env`)

```ini
# LLM Providers
OPENAI_API_KEY=sk-proj-...
GOOGLE_API_KEY=AIzaSy...
GROQ_API_KEY=gsk_...

# Vector Database (Pinecone)
PINECONE_API_KEY=pcsk_...
PINECONE_INDEX_NAME=support-sphere-ai

# Pipeline Defaults
DECISION_PROVIDER=openai
DECISION_MODEL=gpt-5.4-mini
GENERATION_PROVIDER=openai
GENERATION_MODEL=gpt-5.4-mini
SEARCH_MODE=hybrid
FINAL_TOP_K=5
DEFAULT_RERANKER=none
```
