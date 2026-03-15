"""
pipeline.py
───────────
Master GraphRAG controller.
Orchestrates the full query → retrieval → answer pipeline.
"""

from src.retrieval import (
    qdrant_entity_retriever,
    fetch_subgraph_from_neo4j,
    search_qdrant_summaries
)
from src.config import (
    COLLECTION_ENTITIES,
    COLLECTION_SUMMARIES,
    LLM_MODEL
)


# ──────────────────────────────────────────
# ANSWER GENERATION
# ──────────────────────────────────────────

def answer_from_subgraph(query: str, subgraph: list, llm) -> str:
    """
    Generate answer using Neo4j subgraph context (LOCAL mode).

    Strictly grounds the LLM in graph data to prevent hallucination.

    Args:
        query:    User query string
        subgraph: List of relationship triplet strings from Neo4j
        llm:      LlamaIndex LLM instance

    Returns:
        LLM-generated answer string
    """
    graph_context = "\n".join(subgraph) if subgraph else "No graph data found."

    prompt = f"""
    ROLE: You are a professional FinTech Graph Analyst.

    GRAPH CONTEXT (Entities & Relationships from Knowledge Graph):
    {graph_context}

    USER QUERY: {query}

    INSTRUCTIONS:
    1. Answer ONLY based on the graph context above
    2. Reference specific relationship types and impact percentages
    3. If the answer is not in the graph, say "Not found in graph"
    4. Keep the answer factual and concise
    """
    return llm.complete(prompt).text


def answer_from_summaries(query: str, context: str, llm) -> str:
    """
    Generate answer using community summary context (GLOBAL mode).

    Uses hierarchical summaries for broad industry-level questions.

    Args:
        query:   User query string
        context: Concatenated community summary strings from Qdrant
        llm:     LlamaIndex LLM instance

    Returns:
        LLM-generated answer string
    """
    prompt = f"""
    ROLE: You are a senior FinTech Industry Analyst.

    INDUSTRY CONTEXT (Community Summaries):
    {context}

    USER QUERY: {query}

    INSTRUCTIONS:
    1. Answer using the industry context provided
    2. Highlight cross-sector patterns and strategic themes
    3. Provide actionable insights where relevant
    4. Keep the tone professional and analytical
    """
    return llm.complete(prompt).text


def get_query_level(query: str, llm) -> int:
    """
    Classify a GLOBAL query into community summary level.

    Args:
        query: User query string
        llm:   LlamaIndex LLM instance

    Returns:
        0 = specific sector query → Level 0 summaries
        1 = broad strategic query → Level 1 summaries
    """
    prompt = f"""
    Classify this FinTech query:
    - Level 1: Broad, strategic, macro, or industry-wide questions
    - Level 0: Specific sector, company group, or regional questions

    Query: "{query}"

    Respond with ONLY the digit 0 or 1:"""

    response = llm.complete(prompt).text
    try:
        return int(response.strip())
    except ValueError:
        return 1  # Default to strategic level on parse failure


# ──────────────────────────────────────────
# MASTER CONTROLLER
# ──────────────────────────────────────────

def graphrag_query(
    query:         str,
    graph_db,
    qdrant_client,
    embed_model,
    llm,
    verbose:       bool = True
) -> str:
    """
    Master GraphRAG controller — full query pipeline.

    Routing Logic:
      LOCAL  → Qdrant entity search → Neo4j subgraph (2-hop) → LLM
      GLOBAL → Level classification → Qdrant filtered summary search → LLM

    Args:
        query:         User query string
        graph_db:      Neo4jGraph instance
        qdrant_client: QdrantClient instance
        embed_model:   SentenceTransformer model
        llm:           LlamaIndex LLM instance
        verbose:       Print routing decisions if True

    Returns:
        Final answer string
    """

    # ── Step 1: Route the query ──
    router_prompt = f"""
    You are a Routing Engine for a FinTech GraphRAG system.

    Select GLOBAL if:
    - Query asks for industry trends, sector overviews, or macro analysis
    - Example: "What are the risks in the payments industry?"

    Select LOCAL if:
    - Query asks about specific companies, direct relationships, or exact data
    - Example: "What is Visa's impact percentage on PayPal?"

    Respond with ONLY 'GLOBAL' or 'LOCAL'.
    Query: "{query}"
    """
    route = llm.complete(router_prompt).text.strip().upper()

    # ── Step 2a: LOCAL path ──
    if "LOCAL" in route:
        if verbose:
            print("📍 Route: LOCAL (Neo4j Subgraph Retrieval)")

        # Embed query
        query_vector = embed_model.encode(query).tolist()

        # Find similar entities in Qdrant
        entity_ids = qdrant_entity_retriever(qdrant_client, COLLECTION_ENTITIES, query_vector)
        if verbose:
            print(f"   Qdrant → {len(entity_ids)} matching entities")

        # Fetch subgraph from Neo4j
        subgraph = fetch_subgraph_from_neo4j(graph_db, entity_ids)
        if verbose:
            print(f"   Neo4j  → {len(subgraph)} relationships retrieved")

        return answer_from_subgraph(query, subgraph, llm)

    # ── Step 2b: GLOBAL path ──
    else:
        if verbose:
            print("🌍 Route: GLOBAL (Community Summary Retrieval)")

        # Determine summary level
        level = get_query_level(query, llm)
        if verbose:
            label = "strategic" if level == 1 else "sector-specific"
            print(f"   Level  → {level} ({label})")

        # Search filtered summaries
        context = search_qdrant_summaries(
            query, COLLECTION_SUMMARIES, qdrant_client, embed_model, level=level
        )

        return answer_from_summaries(query, context, llm)
