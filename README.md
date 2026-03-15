# ⚡ Advanced AI Engineering Lab

> Designing and implementing **production-grade AI systems** for Financial Intelligence.  
> Core focus: GraphRAG pipelines, Multi-Agent systems + MCP, Data Engineering, LLM Fine-tuning, Real-time streaming, Evaluation (RAGAS • LangSmith), Monitoring (Prometheus • Grafana), LLMOps, Cloud deployment — end to end, production ready.

[![stack](https://img.shields.io/badge/Neo4j%20•%20Qdrant%20•%20Groq%20•%20Instructor-blue?style=flat-square)](.)
[![algo](https://img.shields.io/badge/Leiden%20•%20GraphRAG%20•%20Hierarchical%20Retrieval-purple?style=flat-square)](.)
[![updated](https://img.shields.io/badge/updated-daily-brightgreen?style=flat-square)](.)
[![sprint](https://img.shields.io/badge/sprint-01%20Complete%20✅-red?style=flat-square)](./Financial-GraphRAG-Sprint1)

---

## 🧠 What This Lab Is

This is not a course project. This is an active engineering lab where I design, build, and benchmark AI systems focused on **Financial Intelligence** — knowledge graph construction, hierarchical GraphRAG pipelines, agentic reasoning, real-time data processing, LLM evaluation, and production monitoring.

Every sprint ships working, tested code. Every technical decision is documented. Every architecture is built from scratch — no black-box frameworks.

**Domain:** Financial Intelligence — company networks, market relationships, investment analysis  
**Philosophy:** Production-first. Full control over every layer. No pip-install-and-pray.

---

## 📂 Lab Structure

```
Advanced-AI-Engineering-Lab/
│
├── Financial-GraphRAG-Sprint1/   ✅ Complete
│   │   Focus: Adaptive Hierarchical GraphRAG engine
│   │          with Neo4j + Qdrant + Leiden community detection
│   ├── src/
│   │   ├── config.py
│   │   ├── database.py
│   │   ├── models.py
│   │   ├── data_generation.py
│   │   ├── ingestion.py
│   │   ├── community.py
│   │   ├── retrieval.py
│   │   └── pipeline.py
│   ├── notebooks/
│   ├── assets/
│   ├── main.py
│   ├── requirements.txt
│   ├── .env.example
│   └── README.md
│
└── README.md                     ← This file
```

> Each sprint is self-contained with its own `src/`, `notebooks/`, `assets/`, and full `README.md`. New sprints ship when current sprint is complete and tested.

---

## 🔴 Current Sprint

### ✅ Sprint 01 — Financial GraphRAG Engine

**Goal:** Build a production-grade GraphRAG system for FinTech that goes beyond Microsoft GraphRAG and LlamaIndex — full control, no black boxes.

![Architecture](Financial-GraphRAG-Sprint1/assets/architecture.png)

**What was built:**

```
User Query
    │
    ▼
┌─────────────────┐
│  Query Router   │  ← LLM decides: LOCAL or GLOBAL
└────────┬────────┘
         │
   ┌─────┴──────┐
   │            │
LOCAL         GLOBAL
   │            │
Qdrant       Level Router
Entity       (0 or 1)
Search           │
   │         Qdrant
Neo4j        Summary
Subgraph     Search
(2-hop)      (filtered)
   │            │
   └─────┬──────┘
         │
    ┌────▼────┐
    │   LLM   │
    └─────────┘
```

**Key engineering decisions:**

| Decision | Why |
|---|---|
| **Neo4j MERGE** over CREATE | Idempotent writes — pipeline re-runs safely, no duplicate nodes |
| **Qdrant payload filtering** over full scan | ~70% token reduction vs sending all summaries to LLM |
| **includeIntermediateCommunities: true** | Fixes NullPointerException bug in Neo4j GDS Leiden — DendrogramManager not initialized on false path |
| **neo4j_id in Qdrant payload** | Direct 1:1 bridge — eliminates silent ID mismatch bug at scale |
| **Custom pipeline** over LlamaIndex/Microsoft | Full control over parsing, chunking, retrieval — production debuggable |

**Results:**
```
LOCAL query:  "Visa → PayPal relationship?"
→ FINANCIAL_IMPACT | 11.2% ✅ (exact graph fact, zero hallucination)

GLOBAL query: "FinTech payments industry trends?"
→ Strategic synthesis from Level 1 community summaries ✅
```

---

## 🗺️ Roadmap

| Sprint | Focus | Status |
|---|---|---|
| **Sprint 01** | Adaptive Hierarchical GraphRAG — Neo4j + Qdrant + Leiden | ✅ Complete |
| **Sprint 02** | GraphRAG + LightRAG Hybrid — entity-level embeddings, no community summaries | 🔄 Next |
| **Sprint 03** | Agentic RAG — LangGraph + MCP tool integration | ⏳ Planned |
| **Sprint 04** | Real-time pipeline — Kafka + Redis Streams + fraud detection | ⏳ Planned |
| **Sprint 05** | LLM Evaluation — RAGAS + LangSmith + custom FinTech metrics | ⏳ Planned |
| **Sprint 06** | LLM Fine-tuning — QLoRA + Unsloth + MLflow Registry | ⏳ Planned |
| **Sprint 07** | Production API — FastAPI + JWT + Prometheus + Grafana | ⏳ Planned |
| **Sprint 08** | Cloud Deployment — Docker + GitHub Actions + AWS | ⏳ Planned |

---

## 🛠️ Full Stack (End Goal)

**AI & Knowledge Layer**
```
Knowledge Graph    Neo4j Aura — Leiden community detection, multi-hop Cypher traversal
Vector Store       Qdrant Cloud — payload filtering, hierarchical level search
Structured Output  instructor + Pydantic — auto-retry, zero broken JSON
Agent Framework    LangGraph • MCP Protocol • ReAct pattern
Evaluation         RAGAS • LangSmith • MLflow — every result is measured
Fine-tuning        QLoRA • Unsloth • PEFT • MLflow Registry
```

**Data Engineering Layer**
```
Databases          PostgreSQL + asyncpg + Alembic
Streaming          Kafka • Redis Streams (semantic caching)
Orchestration      Apache Airflow — DAGs, scheduled pipelines
Async Python       asyncio + asyncio.gather()
```

**Production & Infra Layer**
```
API                FastAPI + JWT + rate limiting + WebSockets
Testing            pytest + pytest-asyncio — 80%+ coverage
Infra              Docker Compose • GitHub Actions CI/CD • AWS
Monitoring         Prometheus + Grafana — custom metrics + alerting
Security           AWS Secrets Manager • Presidio PII masking
```

---

## 📬 Connect

**LinkedIn:** [linkedin.com/in/huzaifa-qureshi-ai](https://linkedin.com/in/huzaifa-qureshi-ai)  
**GitHub:** [github.com/huzaifa12466](https://github.com/huzaifa12466)

---

<p align="center"><i>Build from scratch. Understand everything. Ship what works.</i></p>
