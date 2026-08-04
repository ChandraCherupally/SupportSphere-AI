"""
- Centralized project configuration. 
- Only this file should read environment variables. 
- Everything else imports constants from here.
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

# =============================================================================
# Application & LLM Defaults
# =============================================================================

LLM_PROVIDER: str = "openai"
LLM_MODEL: str = "gpt-5.4-mini"

DECISION_PROVIDER: str = LLM_PROVIDER
DECISION_MODEL: str = LLM_MODEL

GENERATION_PROVIDER: str = LLM_PROVIDER
GENERATION_MODEL: str = LLM_MODEL

LLM_TEMPERATURE: float = 0.0

# =============================================================================
# API Keys
# =============================================================================

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "") or ""
GOOGLE_API_KEY_EMBED = os.getenv("GOOGLE_API_KEY_EMBED", "") or ""
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "") or ""
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "") or ""


def get_api_key(provider: str) -> str:
    """Return the global API key for the given provider."""
    p = provider.lower()
    if p == "google":
        return GOOGLE_API_KEY
    if p == "openai":
        return OPENAI_API_KEY
    if p == "groq":
        return GROQ_API_KEY
    return ""


# =============================================================================
# Embeddings
# =============================================================================

EMBEDDING_PROVIDER: str = "google"
EMBEDDING_MODEL: str = "gemini-embedding-001"
EMBEDDING_DIMENSION: int = 768
EMBED_BATCH_SIZE: int = 100

# =============================================================================
# Vector Store
# =============================================================================

VECTOR_DB = os.getenv("VECTOR_DB", "pinecone").lower()

# =============================================================================
# Pinecone
# =============================================================================

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "support-sphere-ai")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
PINECONE_METRIC = "cosine"

# =============================================================================
# Chunking
# =============================================================================

MAX_SECTION_SIZE: int = 1200
SECTION_OVERLAP: int = 150

# =============================================================================
# Retrieval
# =============================================================================

SEARCH_MODE: str = "hybrid"
BM25_INDEX_PATH = ARTIFACTS_DIR / "bm25_index.pkl"
VECTOR_TOP_K: int = 20
BM25_TOP_K: int = 20
FINAL_TOP_K: int = 10

# =============================================================================
# Context Builder
# =============================================================================

MAX_CONTEXT_CHUNKS: int = 10
MAX_CONTEXT_TOKENS: int = 6000

# =============================================================================
# Validation
# =============================================================================

SUPPORTED_LLM_PROVIDERS = {"google", "openai", "groq"}

if DECISION_PROVIDER not in SUPPORTED_LLM_PROVIDERS:
    raise ValueError(f"Unsupported DECISION_PROVIDER: {DECISION_PROVIDER}")

if GENERATION_PROVIDER not in SUPPORTED_LLM_PROVIDERS:
    raise ValueError(f"Unsupported GENERATION_PROVIDER: {GENERATION_PROVIDER}")

SUPPORTED_EMBEDDING_PROVIDERS = {"google"}

if EMBEDDING_PROVIDER not in SUPPORTED_EMBEDDING_PROVIDERS:
    raise ValueError(f"Unsupported EMBEDDING_PROVIDER: {EMBEDDING_PROVIDER}")

# =============================================================================
# Reranker
# =============================================================================

DEFAULT_RERANKER = "none"
AVAILABLE_RERANKERS = ("none", "cross_encoder", "flashrank","llm")
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_RERANKER_MODEL = "gemini-2.5-flash"
FLASHRANK_MODEL = "ms-marco-MiniLM-L-12-v2"
FLASHRANK_MAX_LENGTH = 512
