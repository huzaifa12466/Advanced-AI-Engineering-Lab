"""
models.py
─────────
All Pydantic schemas used across the pipeline.
Defines data contracts for LLM extraction and graph storage.
"""

import uuid
from typing import List, Dict, Optional
from pydantic import BaseModel, Field


# ──────────────────────────────────────────
# DATA GENERATION SCHEMAS
# ──────────────────────────────────────────

class CompaniesList(BaseModel):
    """Schema for extracting company names and sectors from LLM."""
    companies: List[str] = Field(description="List of unique company names")
    sectors:   List[str] = Field(description="Corresponding sector for each company")


class CompanyDetails(BaseModel):
    """Detailed profile for a single FinTech company."""
    company_name:           str         = Field(description="Company name")
    sector:                 str         = Field(description="Industry sector")
    ceo:                    str         = Field(description="Current CEO full name")
    ceo_previous_companies: List[str]   = Field(description="Companies where CEO worked before")
    ceo_previous_positions: List[str]   = Field(description="Positions CEO held previously")
    profit_loss:            float       = Field(description="Profit (+) or loss (-) in millions USD")
    connected_companies:    List[str]   = Field(description="Companies this one is connected to (3-5)")
    impact_percentage:      List[float] = Field(description="Impact % on each connected company")
    relationship_type:      List[str]   = Field(description="Relationship type e.g. PARTNER, COMPETITOR")


class CEOHistory(BaseModel):
    """CEO career history for building person nodes in the graph."""
    ceo_name:         str       = Field(description="Full CEO name")
    current_company:  str       = Field(description="Current company")
    current_position: str       = Field(description="Current title e.g. CEO, MD")
    previous_roles:   List[str] = Field(description="Previous positions e.g. CFO at Company X")
    years_experience: int       = Field(description="Total years of industry experience")


# ──────────────────────────────────────────
# GRAPH STORAGE SCHEMAS
# ──────────────────────────────────────────

class GraphNode(BaseModel):
    """
    Represents a node to be inserted into Neo4j.
    The `uid` is auto-generated and excluded from LLM prompts.
    """
    uid:        str  = Field(default_factory=lambda: str(uuid.uuid4()))
    name:       str
    label:      str                  # 'Company' or 'CEO'
    properties: Dict = Field(default_factory=dict)


class GraphRelationship(BaseModel):
    """Represents a directed relationship between two Neo4j nodes."""
    source_name:       str
    target_name:       str
    relationship_type: str
    impact_percentage: float = Field(default=0.0)


# ──────────────────────────────────────────
# COMMUNITY SCHEMAS
# ──────────────────────────────────────────

class CommunitySummary(BaseModel):
    """A generated summary for a single community."""
    community_id: int
    level:        int    # 0 = fine-grained, 1 = strategic
    summary:      str
    node_names:   List[str] = Field(default_factory=list)


# ──────────────────────────────────────────
# QUERY ROUTING SCHEMAS
# ──────────────────────────────────────────

class QueryRoute(BaseModel):
    """Router decision for incoming user query."""
    route:  str  # 'LOCAL' or 'GLOBAL'
    reason: str  = Field(description="Brief reason for routing decision")
