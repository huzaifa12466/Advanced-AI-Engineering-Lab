"""
retrieval.py
────────────
Retrieval functions for the GraphRAG pipeline.
Handles Qdrant vector search and Neo4j subgraph traversal.
"""

from typing import List, Optional

from qdrant_client import QdrantClient, models
from qdrant_client.http import models as rest

from src.database import Neo4jGraph
from src.config import RETRIEVAL_TOP_K, SUMMARY_TOP_K


def qdrant_entity_retriever(
    qdrant:          QdrantClient,
    collection_name: str,
    query_vector:    list,
    limit:           int = RETRIEVAL_TOP_K
) -> List[str]:
    """
    Retrieve top-K entity Neo4j IDs from Qdrant via semantic search.

    This is the core Qdrant → Neo4j bridge:
    Embedded query → similar entities → neo4j_id from payload

    Args:
        qdrant:          QdrantClient instance
        collection_name: Entity collection name
        query_vector:    Embedded query vector
        limit:           Top-K results to return

    Returns:
        List of Neo4j node IDs
    """
    result = qdrant.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=limit
    )
    return [res.payload['neo4j_id'] for res in result.points]


def fetch_subgraph_from_neo4j(graph_db: Neo4jGraph, entity_ids: List[str]) -> List[str]:
    """
    Fetch multi-hop subgraph from Neo4j for given entity IDs.

    Traverses up to 2 hops from seed entities to capture
    direct relationships and second-degree connections.

    Args:
        graph_db:   Neo4jGraph instance
        entity_ids: List of Neo4j node IDs (from Qdrant)

    Returns:
        List of human-readable triplet strings for LLM context
    """
    query = """
    MATCH p = (n)-[*1..2]->(m)
    WHERE n.id IN $entity_ids
    UNWIND relationships(p) AS r
    RETURN DISTINCT
        startNode(r).name  AS source,
        type(r)            AS relationship,
        r.impact_percentage AS impact,
        endNode(r).name    AS target
    """
    with graph_db.session() as session:
        results = session.run(query, entity_ids=entity_ids)
        return [
            f"{r['source']} --[{r['relationship']} | impact: {r['impact']}%]--> {r['target']}"
            for r in results
        ]


def search_qdrant_summaries(
    query:           str,
    collection_name: str,
    qdrant:          QdrantClient,
    embed_model,
    level:           Optional[int] = None,
    limit:           int = SUMMARY_TOP_K
) -> str:
    """
    Search community summaries in Qdrant with optional level filtering.

    Level filtering ensures only relevant granularity is retrieved:
      level=0 → fine-grained sector summaries
      level=1 → strategic industry summaries
      None    → search all levels

    Args:
        query:           User query string
        collection_name: Summaries collection name
        qdrant:          QdrantClient instance
        embed_model:     SentenceTransformer model
        level:           Community level filter (0, 1, or None)
        limit:           Top-K summaries to retrieve

    Returns:
        Concatenated summary string for LLM context
    """
    query_vector = embed_model.encode(query).tolist()
    query_filter = None

    if level is not None:
        query_filter = models.Filter(
            must=[rest.FieldCondition(key="level", match=rest.MatchValue(value=level))]
        )

    result = qdrant.query_points(
        collection_name=collection_name,
        query=query_vector,
        query_filter=query_filter,
        limit=limit
    )
    return "\n\n---\n\n".join([res.payload['text'] for res in result.points])
