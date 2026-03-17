"""
ingestion.py
────────────
Neo4j and Qdrant ingestion pipeline.
Converts raw company data into graph nodes/relationships
and embeds entities for vector search.
"""

import uuid
from typing import List, Dict, Tuple

from qdrant_client.models import PointStruct
from qdrant_client import QdrantClient, models

from src.models import CompanyDetails
from src.database import Neo4jGraph
from src.config import EMBED_DIMENSION, COLLECTION_ENTITIES


# ──────────────────────────────────────────
# GRAPH OBJECT BUILDER
# ──────────────────────────────────────────

def build_graph_objects(companies: List[CompanyDetails]) -> Tuple[dict, list]:
    """
    Convert CompanyDetails list into graph nodes and relationships.

    Each company generates:
      - One Company node
      - One CEO node
      - One CEO_OF relationship
      - N company-to-company relationships

    Args:
        companies: List of CompanyDetails from data generation

    Returns:
        Tuple: (nodes_dict, relationships_list)
        nodes_dict: {name -> {uid, name, label, properties}}
    """
    nodes: Dict[str, dict] = {}
    relationships: List[dict] = []

    for company in companies:

        # ── Company Node ──
        company_uid = str(uuid.uuid4())
        nodes[company.company_name] = {
            "uid":   company_uid,
            "name":  company.company_name,
            "label": "Company",
            "properties": {
                "sector":      company.sector,
                "profit_loss": company.profit_loss
            }
        }

        # ── CEO Node ──
        ceo_uid = str(uuid.uuid4())
        if company.ceo not in nodes:
            nodes[company.ceo] = {
                "uid":   ceo_uid,
                "name":  company.ceo,
                "label": "CEO",
                "properties": {
                    "previous_companies": company.ceo_previous_companies
                }
            }

        # ── CEO → Company Relationship ──
        relationships.append({
            "source_name": company.ceo,
            "target_name": company.company_name,
            "type":        "CEO_OF",
            "impact":      0.0
        })

        # ── Company → Company Relationships ──
        for connected, impact, rel_type in zip(
            company.connected_companies,
            company.impact_percentage,
            company.relationship_type
        ):
            # Create target node if not seen before
            if connected not in nodes:
                nodes[connected] = {
                    "uid":        str(uuid.uuid4()),
                    "name":       connected,
                    "label":      "Company",
                    "properties": {}
                }

            relationships.append({
                "source_name": company.company_name,
                "target_name": connected,
                "type":        rel_type.upper().replace(" ", "_"),
                "impact":      impact
            })

    print(f"✅ Built {len(nodes)} nodes and {len(relationships)} relationships")
    return nodes, relationships


# ──────────────────────────────────────────
# NEO4J INGESTION
# ──────────────────────────────────────────

def ingest_to_neo4j(all_nodes: dict, all_rels: list, graph_db: Neo4jGraph):
    """
    Push all nodes and relationships into Neo4j using MERGE (upsert).
    MERGE guarantees no duplicates even on multiple runs.

    Args:
        all_nodes: Dict of node data keyed by entity name
        all_rels:  List of relationship dicts
        graph_db:  Neo4jGraph instance
    """
    with graph_db.session() as session:

        # ── 1. Push Company Nodes ──
        companies = [n for n in all_nodes.values() if n['label'].lower() == 'company']
        session.run("""
            UNWIND $batch AS row
            MERGE (n:Company {id: row.uid})
            SET n.name        = row.name,
                n.sector      = row.properties.sector,
                n.profit_loss = row.properties.profit_loss
        """, batch=companies)
        print(f"  ✅ Pushed {len(companies)} Company nodes")

        # ── 2. Push CEO Nodes ──
        ceos = [n for n in all_nodes.values() if n['label'].lower() == 'ceo']
        session.run("""
            UNWIND $batch AS row
            MERGE (n:CEO {id: row.uid})
            SET n.name = row.name
        """, batch=ceos)
        print(f"  ✅ Pushed {len(ceos)} CEO nodes")

        # ── 3. Push Relationships ──
        pushed = 0
        for rel in all_rels:
            source = all_nodes.get(rel['source_name'])
            target = all_nodes.get(rel['target_name'])
            if not source or not target:
                continue
            session.run(f"""
                MATCH (a {{id: $source_id}})
                MATCH (b {{id: $target_id}})
                MERGE (a)-[r:{rel['type']}]->(b)
                SET r.impact_percentage = $impact
            """,
            source_id=source['uid'],
            target_id=target['uid'],
            impact=rel['impact'])
            pushed += 1

        print(f"  ✅ Pushed {pushed} relationships")
    print("\n🎉 Neo4j ingestion complete!")


# ──────────────────────────────────────────
# QDRANT INGESTION
# ──────────────────────────────────────────

def create_qdrant_collection(client: QdrantClient, name: str, dimension: int = EMBED_DIMENSION):
    """
    Create a Qdrant collection if it does not already exist.

    Args:
        client:    QdrantClient instance
        name:      Collection name
        dimension: Embedding vector size
    """
    try:
        client.get_collection(name)
        print(f"  ⚡ '{name}' already exists — skipping")
    except Exception:
        client.recreate_collection(
            collection_name=name,
            vectors_config=models.VectorParams(
                size=dimension,
                distance=models.Distance.COSINE
            )
        )
        print(f"  ✅ Created collection '{name}' (dim={dimension})")


def get_all_entities_from_neo4j(graph_db: Neo4jGraph) -> List[dict]:
    """
    Fetch all Company and CEO nodes from Neo4j, including their properties.
    Properties are used to build a rich embedding string.

    Returns:
        List of dicts: {name, label, id, sector, profit_loss, previous_companies}
    """
    results = graph_db.run_query("""
        MATCH (n)
        WHERE (n:CEO OR n:Company)
        RETURN DISTINCT n.name            AS name,
                        labels(n)[0]      AS label,
                        n.id              AS id,
                        n.sector          AS sector,
                        n.profit_loss     AS profit_loss
        ORDER BY n.name
    """)
    entities = []
    for r in results:
        entities.append({
            "name":         r["name"],
            "label":        r["label"],
            "id":           r["id"],
            "sector":       r["sector"],
            "profit_loss":  r["profit_loss"],
        })
    return entities


def ingest_entities_to_qdrant(graph_db: Neo4jGraph, collection_name: str,
                               qdrant: QdrantClient, embed_model):
    """
    Embed all Neo4j entities and store in Qdrant.

    Critical: neo4j_id is stored in payload to enable the
    Qdrant → Neo4j bridge during retrieval.

    Args:
        graph_db:        Neo4jGraph instance
        collection_name: Target Qdrant collection
        qdrant:          QdrantClient instance
        embed_model:     SentenceTransformer model
    """
    entities = get_all_entities_from_neo4j(graph_db)
    print(f"  📦 Ingesting {len(entities)} entities...")

    for entity in entities:
        # ── Build a rich text string combining entity identity + properties ──
        # Embedding the name alone loses all context. By joining the entity's
        # label, sector, and financial performance into one string, the vector
        # captures semantic meaning far beyond just the company name.
        parts = [f"{entity['name']} ({entity['label']})"]
        if entity.get("sector"):
            parts.append(f"Sector: {entity['sector']}")
        if entity.get("profit_loss") is not None:
            direction = "Profit" if entity["profit_loss"] >= 0 else "Loss"
            parts.append(f"{direction}: {abs(entity['profit_loss'])}M USD")

        embed_text = " | ".join(parts)  # e.g. "Visa (Company) | Sector: Payments | Profit: 17273M USD"

        embedding = embed_model.encode(embed_text).tolist()
        qdrant.upsert(
            collection_name=collection_name,
            points=[
                PointStruct(
                    id=str(uuid.uuid4()),   # Qdrant's own ID
                    vector=embedding,
                    payload={
                        "name":      entity['name'],
                        "label":     entity['label'],
                        "neo4j_id":  entity['id'],    # ← Bridge to Neo4j
                        "embed_text": embed_text       # ← Store for inspection
                    }
                )
            ]
        )
    print(f"  ✅ All entities ingested into '{collection_name}'")


def ingest_summaries_to_qdrant(level0_summaries: list, level1_summaries: list,
                                collection_name: str, qdrant: QdrantClient, embed_model):
    """
    Embed and store community summaries in Qdrant with level metadata.

    Level payload enables filtered search:
      level=0 → sector-specific queries
      level=1 → strategic industry queries

    Args:
        level0_summaries: Fine-grained community summaries
        level1_summaries: Strategic community summaries
        collection_name:  Target Qdrant collection
        qdrant:           QdrantClient instance
        embed_model:      SentenceTransformer model
    """
    all_points = []

    for s in level0_summaries:
        vec = embed_model.encode(s['summary']).tolist()
        all_points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=vec,
            payload={"text": s['summary'], "level": 0, "community_id": s['id']}
        ))

    for s in level1_summaries:
        vec = embed_model.encode(s['summary']).tolist()
        all_points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=vec,
            payload={"text": s['summary'], "level": 1, "community_id": s['parent_id']}
        ))

    # Index for fast level filtering
    qdrant.create_payload_index(
        collection_name=collection_name,
        field_name="level",
        field_schema=models.PayloadSchemaType.INTEGER
    )

    qdrant.upsert(collection_name=collection_name, points=all_points)
    print(f"  ✅ Stored {len(all_points)} summaries "
          f"({len(level0_summaries)} L0 + {len(level1_summaries)} L1)")
