"""
main.py
───────
FinTech GraphRAG — Single entry point.

Usage:
    # Run full setup pipeline (first time)
    python main.py --setup

    # Run a single query
    python main.py --query "What is Visa's impact on PayPal?"

    # Run demo queries
    python main.py --demo
"""

import argparse
import sys

from src.config import (
    COLLECTION_ENTITIES,
    COLLECTION_SUMMARIES,
    EMBED_DIMENSION
)
from src.database import (
    Neo4jGraph,
    get_qdrant_client,
    get_embed_model,
    get_llm,
    get_instructor_client
)
from src.data_generation import run_data_generation
from src.ingestion import (
    build_graph_objects,
    ingest_to_neo4j,
    create_qdrant_collection,
    ingest_entities_to_qdrant,
    ingest_summaries_to_qdrant
)
from src.community import (
    run_community_detection,
    generate_all_summaries_level0,
    generate_level1_summaries
)
from src.pipeline import graphrag_query


# ──────────────────────────────────────────
# SETUP PIPELINE
# ──────────────────────────────────────────

def run_setup():
    """
    Full one-time setup pipeline.

    Steps:
        1. Connect to databases
        2. Generate synthetic FinTech data
        3. Push to Neo4j
        4. Run Leiden community detection
        5. Generate community summaries
        6. Push entities + summaries to Qdrant
    """

    print("\n" + "=" * 60)
    print("  FinTech GraphRAG — Setup Pipeline")
    print("=" * 60)

    # ── Step 1: Connect ──
    print("\n📡 Step 1: Connecting to databases...")
    db             = Neo4jGraph()
    qdrant         = get_qdrant_client()
    embed_model    = get_embed_model()
    llm            = get_llm()
    instructor_client = get_instructor_client()

    if not db.verify_connection():
        print("❌ Neo4j connection failed. Check your .env file.")
        sys.exit(1)
    print("✅ All connections established!")

    # ── Step 2: Generate Data ──
    print("\n📊 Step 2: Generating synthetic FinTech data...")
    all_companies = run_data_generation(instructor_client)

    # ── Step 3: Push to Neo4j ──
    print("\n🗄️  Step 3: Pushing to Neo4j...")
    all_nodes, all_rels = build_graph_objects(all_companies)
    ingest_to_neo4j(all_nodes, all_rels, db)

    # ── Step 4: Community Detection ──
    print("\n🔍 Step 4: Running Leiden community detection...")
    run_community_detection(db)

    # ── Step 5: Generate Summaries ──
    print("\n📝 Step 5: Generating community summaries...")
    level0_summaries = generate_all_summaries_level0(db, llm)
    level1_summaries = generate_level1_summaries(level0_summaries, db, llm)
    print(f"✅ {len(level0_summaries)} Level-0 + {len(level1_summaries)} Level-1 summaries")

    # ── Step 6: Push to Qdrant ──
    print("\n📦 Step 6: Pushing to Qdrant...")
    create_qdrant_collection(qdrant, COLLECTION_ENTITIES,  EMBED_DIMENSION)
    create_qdrant_collection(qdrant, COLLECTION_SUMMARIES, EMBED_DIMENSION)
    ingest_entities_to_qdrant(db, COLLECTION_ENTITIES, qdrant, embed_model)
    ingest_summaries_to_qdrant(level0_summaries, level1_summaries,
                                COLLECTION_SUMMARIES, qdrant, embed_model)

    # ── Done ──
    nodes_total = db.run_query("MATCH (n) RETURN count(n) AS c")[0]['c']
    rels_total  = db.run_query("MATCH ()-[r]->() RETURN count(r) AS c")[0]['c']

    print("\n" + "=" * 60)
    print("  ✅ Setup Complete!")
    print("=" * 60)
    print(f"""
  📊 Final Stats:
     Neo4j Nodes:         {nodes_total}
     Neo4j Relationships: {rels_total}
     Qdrant Entities:     {qdrant.count(COLLECTION_ENTITIES).count}
     Qdrant Summaries:    {qdrant.count(COLLECTION_SUMMARIES).count}

  🚀 Now run:
     python main.py --demo
     python main.py --query "Your question here"
    """)


# ──────────────────────────────────────────
# QUERY PIPELINE
# ──────────────────────────────────────────

def run_query(query: str):
    """
    Run a single query through the GraphRAG pipeline.

    Args:
        query: User query string
    """
    print("\n" + "=" * 60)
    print(f"  Query: {query}")
    print("=" * 60 + "\n")

    db          = Neo4jGraph()
    qdrant      = get_qdrant_client()
    embed_model = get_embed_model()
    llm         = get_llm()

    answer = graphrag_query(
        query=query,
        graph_db=db,
        qdrant_client=qdrant,
        embed_model=embed_model,
        llm=llm,
        verbose=True
    )

    print(f"\n💬 Answer:\n{answer}\n")


# ──────────────────────────────────────────
# DEMO
# ──────────────────────────────────────────

def run_demo():
    """Run 2 demo queries — one LOCAL, one GLOBAL."""

    db          = Neo4jGraph()
    qdrant      = get_qdrant_client()
    embed_model = get_embed_model()
    llm         = get_llm()

    demo_queries = [
        {
            "type":  "LOCAL",
            "query": "What is the exact relationship between Visa and PayPal?"
        },
        {
            "type":  "GLOBAL",
            "query": "What are the major trends and risks in the FinTech payments industry?"
        }
    ]

    for i, item in enumerate(demo_queries, 1):
        print("\n" + "=" * 60)
        print(f"  TEST {i}: {item['type']} Query")
        print("=" * 60)
        print(f"  Query: {item['query']}\n")

        answer = graphrag_query(
            query=item['query'],
            graph_db=db,
            qdrant_client=qdrant,
            embed_model=embed_model,
            llm=llm,
            verbose=True
        )
        print(f"\n💬 Answer:\n{answer}\n")


# ──────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FinTech Adaptive Hierarchical GraphRAG"
    )

    parser.add_argument(
        "--setup",
        action="store_true",
        help="Run full setup pipeline (data generation + Neo4j + Qdrant)"
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Run a single query through the pipeline"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run 2 demo queries (LOCAL + GLOBAL)"
    )

    args = parser.parse_args()

    if args.setup:
        run_setup()
    elif args.query:
        run_query(args.query)
    elif args.demo:
        run_demo()
    else:
        parser.print_help()
