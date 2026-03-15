"""
config.py
─────────
Centralized configuration for the FinTech GraphRAG pipeline.
All settings are loaded from environment variables (.env file).
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────
# Neo4j
# ──────────────────────────────────────────
NEO4J_URI  = os.getenv("NEO4J_URI",  "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASS", "password")

# ──────────────────────────────────────────
# Qdrant
# ──────────────────────────────────────────
QDRANT_URL     = os.getenv("QDRANT_URL",     "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")

# ──────────────────────────────────────────
# Groq / LLM
# ──────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
LLM_MODEL    = "llama-3.3-70b-versatile"

# ──────────────────────────────────────────
# Embeddings
# ──────────────────────────────────────────
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_DIMENSION  = 384

# ──────────────────────────────────────────
# Qdrant Collections
# ──────────────────────────────────────────
COLLECTION_ENTITIES  = "llm_entities"
COLLECTION_SUMMARIES = "llm_summaries"

# ──────────────────────────────────────────
# Pipeline Settings
# ──────────────────────────────────────────
BATCH_SIZE          = 5    # Companies per LLM batch
NUM_COMPANIES       = 40   # Total synthetic companies
SUBGRAPH_HOPS       = 2    # Neo4j traversal depth
RETRIEVAL_TOP_K     = 5    # Qdrant top-K entities
SUMMARY_TOP_K       = 3    # Qdrant top-K summaries
