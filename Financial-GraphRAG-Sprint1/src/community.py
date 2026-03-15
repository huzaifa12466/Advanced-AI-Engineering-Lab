"""
community.py
────────────
Community detection using Neo4j GDS Leiden algorithm
and LLM-powered community summarization.
"""

from collections import defaultdict
from typing import List

from src.database import Neo4jGraph
from src.config import LLM_MODEL


# ──────────────────────────────────────────
# COMMUNITY DETECTION
# ──────────────────────────────────────────

def run_community_detection(graph_db: Neo4jGraph):
    """
    Run Leiden community detection via Neo4j GDS.

    Produces hierarchical community levels:
      n.community[0] → Level 0 (fine-grained clusters)
      n.community[1] → Level 1 (strategic groups)

    IMPORTANT: includeIntermediateCommunities MUST be True.
    Setting it to False causes a NullPointerException in Neo4j GDS
    due to an uninitialized DendrogramManager on that code path.

    Args:
        graph_db: Neo4jGraph instance
    """

    # ── Drop existing projection ──
    graph_db.run_query("CALL gds.graph.drop('finance_projection', false)")
    print("  🗑️  Dropped old projection")

    # ── Create graph projection ──
    graph_db.run_query("""
        CALL gds.graph.project(
            'finance_projection',
            ['Company', 'CEO'],
            {
                ALL: {
                    type: '*',
                    orientation: 'UNDIRECTED',
                    properties: {
                        impact_percentage: {
                            property: 'impact_percentage',
                            defaultValue: 1.0
                        }
                    }
                }
            }
        )
    """)
    print("  ✅ Graph projection created")

    # ── Run Leiden Algorithm ──
    graph_db.run_query("""
        CALL gds.leiden.write(
            'finance_projection',
            {
                writeProperty: 'community',
                includeIntermediateCommunities: true,
                relationshipWeightProperty: 'impact_percentage',
                maxLevels: 3,
                gamma: 1.0,
                theta: 0.01
            }
        )
    """)
    print("  ✅ Leiden algorithm complete")

    # ── Verify ──
    stats = graph_db.run_query("""
        MATCH (n)
        WHERE n.community IS NOT NULL
        RETURN size(n.community) AS levels, count(n) AS nodes
        LIMIT 1
    """)
    if stats:
        print(f"  📊 Levels: {stats[0]['levels']}, Nodes: {stats[0]['nodes']}")

    print("\n🎉 Community detection complete!")


# ──────────────────────────────────────────
# COMMUNITY SUMMARIZATION
# ──────────────────────────────────────────

def get_community_data(graph_db: Neo4jGraph, comm_id: int) -> dict:
    """
    Fetch all nodes and relationships within a community.

    Args:
        graph_db: Neo4jGraph instance
        comm_id:  Level 0 community ID

    Returns:
        Dict with 'nodes' and 'relationships' lists
    """
    query = """
    MATCH (n)-[r]->(m)
    WHERE n.community[0] = $comm_id AND m.community[0] = $comm_id
    RETURN
        collect(DISTINCT n.name + ' (' + labels(n)[0] + ')') AS nodes,
        collect(DISTINCT
            n.name + ' --[' + type(r) +
            ' {impact: ' + toString(coalesce(r.impact_percentage, 0)) + '%}]--> ' +
            m.name
        ) AS relationships
    """
    with graph_db.session() as session:
        result = session.run(query, {"comm_id": comm_id})
        record = result.single()
        if record:
            return {
                "nodes":         record["nodes"],
                "relationships": record["relationships"]
            }
    return {"nodes": [], "relationships": []}


def generate_community_summary(llm, comm_id: int, nodes: list,
                                relationships: list, level: int) -> str:
    """
    Use LLM to generate a textual summary for a community.

    Args:
        llm:           LlamaIndex LLM instance
        comm_id:       Community ID
        nodes:         List of node strings
        relationships: List of relationship strings
        level:         Community level (0=fine, 1=strategic)

    Returns:
        Summary string
    """
    level_desc = "fine-grained sector cluster" if level == 0 else "strategic industry group"
    prompt = f"""
    You are a FinTech industry analyst. Summarize this {level_desc} (Community #{comm_id}).

    ENTITIES:
    {chr(10).join(nodes)}

    RELATIONSHIPS:
    {chr(10).join(relationships)}

    Write a concise 3-5 sentence summary covering:
    1. The sector/theme that unites these entities
    2. Key relationships and their business impact
    3. Overall strategic significance in FinTech
    """
    return llm.complete(prompt).text


def generate_all_summaries_level0(graph_db: Neo4jGraph, llm) -> List[dict]:
    """
    Generate LLM summaries for all Level 0 communities.

    Only processes communities with more than 1 node.

    Returns:
        List of dicts: {id, summary, nodes}
    """
    clusters = graph_db.run_query("""
        MATCH (n)
        WITH n.community[0] AS comm_id, collect(n.name) AS names
        WHERE size(names) > 1
        RETURN comm_id, names
    """)

    summaries = []
    for cluster in clusters:
        comm_id = cluster['comm_id']
        data    = get_community_data(graph_db, comm_id)
        summary = generate_community_summary(llm, comm_id, data['nodes'],
                                              data['relationships'], level=0)
        summaries.append({"id": comm_id, "summary": summary, "nodes": cluster['names']})
        print(f"  ✅ Level 0 — Community {comm_id} ({len(cluster['names'])} nodes)")

    return summaries


def generate_level1_summaries(level0_summaries: list, graph_db: Neo4jGraph, llm) -> List[dict]:
    """
    Aggregate Level 0 summaries into Level 1 strategic summaries.

    Reads the community hierarchy from Neo4j to map
    Level 0 → Level 1 parent communities.

    Args:
        level0_summaries: Output of generate_all_summaries_level0()
        graph_db:         Neo4jGraph instance
        llm:              LlamaIndex LLM instance

    Returns:
        List of Level 1 summary dicts: {parent_id, summary}
    """
    # Map Level 0 → Level 1
    mapping = graph_db.run_query("""
        MATCH (n)
        WHERE n.community[0] IS NOT NULL AND n.community[1] IS NOT NULL
        RETURN DISTINCT n.community[0] AS child, n.community[1] AS parent
    """)

    parent_groups  = defaultdict(list)
    summary_lookup = {s['id']: s['summary'] for s in level0_summaries}

    for rec in mapping:
        child_summary = summary_lookup.get(rec['child'], "")
        if child_summary:
            parent_groups[rec['parent']].append(child_summary)

    level1_summaries = []
    for parent_id, child_summaries in parent_groups.items():
        combined = "\n".join(child_summaries)
        prompt = f"""
        You are a senior FinTech strategist. Synthesize these sector summaries
        into a strategic industry-level overview:

        {combined}

        Write a 4-6 sentence strategic summary covering:
        1. Cross-sector dynamics and interdependencies
        2. Major industry themes and risks
        3. Investment and competitive implications
        """
        summary = llm.complete(prompt).text
        level1_summaries.append({"parent_id": parent_id, "summary": summary})
        print(f"  ✅ Level 1 — Parent community {parent_id}")

    return level1_summaries
