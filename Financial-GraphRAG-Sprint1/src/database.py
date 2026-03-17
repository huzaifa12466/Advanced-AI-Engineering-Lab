"""
database.py
───────────
Database connection managers for Neo4j and Qdrant.
Provides singleton-style access across the pipeline.
"""

from neo4j import GraphDatabase
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

import instructor
from groq import Groq as GroqClient
from llama_index.llms.groq import Groq as LlamaGroq

from src.config import (
    NEO4J_URI, NEO4J_USER, NEO4J_PASS,
    QDRANT_URL, QDRANT_API_KEY,
    GROQ_API_KEY, LLM_MODEL,
    EMBED_MODEL_NAME
)


class Neo4jGraph:
    """
    Neo4j database connection wrapper.
    Provides helper methods for running Cypher queries.
    """

    def __init__(self, uri: str = NEO4J_URI, auth: tuple = (NEO4J_USER, NEO4J_PASS)):
        self.driver = GraphDatabase.driver(uri, auth=auth)

    def close(self):
        """Close the Neo4j connection."""
        if self.driver:
            self.driver.close()

    def run_query(self, query: str, parameters: dict = None) -> list:
        """
        Execute a Cypher read query and return all records.

        Neo4j's `session.run()` returns a lazy Result generator — records are
        only fetched from the driver as you iterate. We consume the generator
        with an explicit loop inside the session context so the connection stays
        open while data is being read. Assigning the result directly or calling
        list() outside the `with` block would close the session before the
        records are materialised.

        Args:
            query:      Cypher query string
            parameters: Query parameters dict

        Returns:
            List of Neo4j records
        """
        with self.driver.session() as session:
            result = session.run(query, parameters or {})
            records = []
            for record in result:   # consume the generator while session is live
                records.append(record)
            return records

    def session(self):
        """Return a raw session for complex transactions."""
        return self.driver.session()

    def verify_connection(self) -> bool:
        """Test the connection and return True if successful."""
        try:
            result = self.run_query("RETURN 'connected' AS status")
            return result[0]["status"] == "connected"
        except Exception as e:
            print(f"Neo4j connection failed: {e}")
            return False


def get_qdrant_client() -> QdrantClient:
    """
    Create and return a Qdrant client instance.

    Returns:
        QdrantClient connected to configured cloud instance
    """
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


def get_embed_model() -> SentenceTransformer:
    """
    Load and return the sentence embedding model.

    Returns:
        SentenceTransformer model (all-MiniLM-L6-v2)
    """
    return SentenceTransformer(EMBED_MODEL_NAME)


def get_llm():
    """
    Return LlamaIndex Groq LLM wrapper for simple completions.

    Returns:
        LlamaGroq LLM instance
    """
    return LlamaGroq(model=LLM_MODEL, api_key=GROQ_API_KEY)


def get_instructor_client():
    """
    Return Instructor-patched Groq client for structured output.

    Returns:
        Instructor-wrapped Groq client
    """
    raw_client = GroqClient(api_key=GROQ_API_KEY)
    return instructor.patch(raw_client, mode=instructor.Mode.JSON)
