"""
Centralized project configuration.
Only this file should read environment variables.
Everything else imports constants from here.
"""

from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

# =============================================================================
# Environment
# =============================================================================

load_dotenv()

# =============================================================================
# Project Paths
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = DATA_DIR / "artifacts"
PROMPTS_DIR = BASE_DIR / "ai"

DECISION_PROVIDER = os.getenv("DECISION_PROVIDER", os.getenv("LLM_PROVIDER", "google")).lower()
DECISION_MODEL = os.getenv("DECISION_MODEL", "gemini-2.5-flash-lite")

GENERATION_PROVIDER = os.getenv("GENERATION_PROVIDER", os.getenv("LLM_PROVIDER", "google")).lower()
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "gemini-2.5-flash")

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "google").lower()
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))

_INITIAL_PROVIDER = os.getenv("LLM_PROVIDER", "google").lower()
_INITIAL_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")

LLM_CONFIG = {
    "decision": {
        "provider": DECISION_PROVIDER,
        "model": DECISION_MODEL,
    },
    "generation": {
        "provider": GENERATION_PROVIDER,
        "model": GENERATION_MODEL,
    }
}

# =============================================================================
# API Keys
# =============================================================================

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_API_KEY_EMBED = os.getenv("GOOGLE_API_KEY_EMBED")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

# =============================================================================
# Embeddings
# =============================================================================

EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER","google").lower()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION","768"))
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE","100"))

# =============================================================================
# Pinecone
# =============================================================================

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME","support-sphere-ai")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE","")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD","aws")
PINECONE_REGION = os.getenv("PINECONE_REGION","us-east-1")
PINECONE_METRIC = "cosine"

# =============================================================================
# Chunking
# =============================================================================

MAX_SECTION_SIZE = int(os.getenv("MAX_SECTION_SIZE","1200"))
SECTION_OVERLAP = int(os.getenv("SECTION_OVERLAP","150"))

# =============================================================================
# Retrieval
# =============================================================================

SEARCH_MODE = os.getenv("SEARCH_MODE", "hybrid").lower()
BM25_INDEX_PATH = ARTIFACTS_DIR / "bm25_index.pkl"
VECTOR_TOP_K = int(os.getenv("VECTOR_TOP_K", "20"))
BM25_TOP_K = int(os.getenv("BM25_TOP_K", "20"))
FINAL_TOP_K = int(os.getenv("FINAL_TOP_K", "10"))

# =============================================================================
# Context Builder
# =============================================================================

MAX_CONTEXT_CHUNKS = int(os.getenv("MAX_CONTEXT_CHUNKS","10"))
MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS","6000"))

# =============================================================================
# Validation
# =============================================================================

SUPPORTED_LLM_PROVIDERS = {
    "google",
    "openai",
    "anthropic",
    "groq",
    "ollama",
}

if DECISION_PROVIDER not in SUPPORTED_LLM_PROVIDERS:
    raise ValueError(f"Unsupported DECISION_PROVIDER: {DECISION_PROVIDER}")

if GENERATION_PROVIDER not in SUPPORTED_LLM_PROVIDERS:
    raise ValueError(f"Unsupported GENERATION_PROVIDER: {GENERATION_PROVIDER}")

SUPPORTED_EMBEDDING_PROVIDERS = {"google",}

if EMBEDDING_PROVIDER not in SUPPORTED_EMBEDDING_PROVIDERS:
    raise ValueError(f"Unsupported EMBEDDING_PROVIDER: {EMBEDDING_PROVIDER}")

# =============================================================================
# Reranker
# =============================================================================

DEFAULT_RERANKER = "none"
AVAILABLE_RERANKERS = ("none","cross_encoder", "flashrank","llm",)
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_RERANKER_MODEL = "gemini-2.5-flash"
FLASHRANK_MODEL = "ms-marco-MiniLM-L-12-v2"
FLASHRANK_MAX_LENGTH = 512


