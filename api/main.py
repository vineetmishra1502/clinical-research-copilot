"""
api/main.py — FastAPI Serving Layer for Pharma RAG Pipeline
=============================================================
Python 3.13 | FastAPI + Pydantic

Exposes the Agentic RAG pipeline as HTTP endpoints.
Same pipeline calls as mcp_server.py but over standard HTTP —
no transport-level timeout issues.

Endpoints:
  POST /research    — full multi-agent pipeline → cited research brief
  POST /search      — retrieval only → raw graded evidence chunks
  GET  /drugs       — lists available drugs in knowledge base
  GET  /health      — health check (DB + API connectivity)

Architecture position:
  Client (curl, Postman, frontend)
        ↓  HTTP JSON
  api/main.py
        ↓  calls Python functions directly
  agents.run_research()  OR  retriever.retrieve()
        ↓  (full pipeline beneath)
  query_rewriter → retriever → grader → agents → synthesizer

Pydantic response models:
  Internal dataclasses (AgentOutput, RetrievedChunk) are converted to
  Pydantic models at the API boundary for JSON serialization.
  This is the only place where dataclass → Pydantic conversion happens.

Run:
  uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
  or: python api/main.py

Langfuse changes (marked # LANGFUSE):
  - Added uuid import
  - Added langfuse_setup import
  - POST /research: handler per request, passed to run_research()
  - POST /search: handler per request, flushed after retrieve()
  - GET /health: "langfuse" key added to api_keys (informational only)
  - health "healthy" logic changed from all(api_keys.values()) to
    explicit openai+cohere check — Langfuse is optional, not required
"""

import asyncio
import os
import sys
import time
import uuid                                                        # LANGFUSE
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add project root to Python path — api/main.py is in api/ subfolder
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

# Pipeline imports — these are our own modules in the project root
from retriever import retrieve, MetadataFilter, RetrievedChunk, close_pool
from agents import run_research, AgentOutput, AgentSection
from query_rewriter import DB_VOCABULARY
from langfuse_setup import get_callback_handler, flush_handler, langfuse_enabled  # LANGFUSE


# ─────────────────────────────────────────────────────────────────────
# 1. PYDANTIC REQUEST MODELS
# ─────────────────────────────────────────────────────────────────────

class ResearchRequest(BaseModel):
    """Request body for POST /research."""
    query: str = Field(
        ...,
        min_length=3,
        description="Clinical research question in natural language.",
        examples=[
            "What is the overall survival benefit of pembrolizumab in NSCLC?",
            "Compare pembrolizumab vs nivolumab in NSCLC survival",
            "What are the immune-related adverse events of pembrolizumab?",
        ],
    )


class SearchRequest(BaseModel):
    """Request body for POST /search."""
    query: str = Field(
        ...,
        min_length=3,
        description="Clinical question or search terms.",
        examples=[
            "pembrolizumab OS in NSCLC",
            "osimertinib EGFR mutation resistance",
        ],
    )
    drug: str | None = Field(
        default=None,
        description="Optional exact drug filter: pembrolizumab, nivolumab, osimertinib, trastuzumab, metformin",
    )
    cancer_type: str | None = Field(
        default=None,
        description="Optional cancer type filter: nsclc, breast-cancer, melanoma, etc.",
    )
    year_min: int | None = Field(
        default=None,
        description="Optional minimum publication year (e.g. 2020).",
    )
    study_type: str | None = Field(
        default=None,
        description="Optional study design filter: RCT, meta-analysis, real-world-study, etc.",
    )
    endpoints: str | None = Field(
        default=None,
        description="Optional endpoint filter: OS, PFS, ORR, AE, etc.",
    )


# ─────────────────────────────────────────────────────────────────────
# 2. PYDANTIC RESPONSE MODELS
# ─────────────────────────────────────────────────────────────────────
# These convert internal dataclasses (AgentOutput, RetrievedChunk)
# to JSON-serializable Pydantic models at the API boundary.
# This is the ONLY place where dataclass → Pydantic conversion happens.

class ChunkResponse(BaseModel):
    """One retrieved evidence chunk — JSON-serializable version of RetrievedChunk."""
    id:             int
    drug:           str
    cancer_type:    str
    study_type:     str
    evidence_level: int
    sample_size:    int | None
    journal:        str
    year:           int
    source:         str           # PMID
    doi:            str
    endpoints:      str
    pdl1_status:    str
    content:        str           # first 500 chars
    rerank_score:   float

    @classmethod
    def from_retrieved_chunk(cls, chunk: RetrievedChunk) -> "ChunkResponse":
        return cls(
            id             = chunk.id,
            drug           = chunk.drug,
            cancer_type    = chunk.cancer_type,
            study_type     = chunk.study_type,
            evidence_level = chunk.evidence_level,
            sample_size    = chunk.sample_size,
            journal        = chunk.journal,
            year           = chunk.year,
            source         = chunk.source,
            doi            = chunk.doi,
            endpoints      = chunk.endpoints,
            pdl1_status    = chunk.pdl1_status,
            content        = chunk.content[:500],
            rerank_score   = round(chunk.rerank_score, 4),
        )


class SectionResponse(BaseModel):
    """One agent section — AgentSection is already Pydantic, we just pick the fields we want."""
    title:            str
    content:          str
    citations:        list[str]
    evidence_quality: str
    limitations:      str
    verdict_note:     str

    @classmethod
    def from_agent_section(cls, section: AgentSection) -> "SectionResponse":
        return cls(
            title            = section.title,
            content          = section.content,
            citations        = section.citations,
            evidence_quality = section.evidence_quality,
            limitations      = section.limitations,
            verdict_note     = section.verdict_note,
        )


class ResearchResponse(BaseModel):
    query:               str
    report:              str
    overall_verdict:     Literal["accepted", "partial", "insufficient"]
    evidence_summary:    str
    sections:            list[SectionResponse]
    elapsed_seconds:     float
    retrieved_contexts:  list[str]  # ← ADD: exact chunks agents used (for RAGAS)

    @classmethod
    def from_agent_output(cls, output: AgentOutput, elapsed: float) -> "ResearchResponse":
        # ← ADD: collect accepted chunk content across all agents, deduplicated
        contexts: list[str] = []
        seen: set[str] = set()
        for agent_result in output.agent_results:
            for graded_chunk in agent_result.grader_output.accepted_chunks:
                content = graded_chunk.chunk.content
                if content not in seen:
                    seen.add(content)
                    contexts.append(content)

        return cls(
            query              = output.query,
            report             = output.report,
            overall_verdict    = output.overall_verdict,
            evidence_summary   = output.evidence_summary,
            sections           = [SectionResponse.from_agent_section(s) for s in output.sections],
            elapsed_seconds    = round(elapsed, 2),
            retrieved_contexts = contexts,  # ← ADD
        )

class SearchResponse(BaseModel):
    """Response for POST /search — retrieval only output."""
    query:           str
    rewritten_query: str
    confidence:      str
    filter_level:    str
    n_candidates:    int
    n_returned:      int
    chunks:          list[ChunkResponse]
    elapsed_seconds: float


class DrugInfo(BaseModel):
    """One drug's info for GET /drugs."""
    name:    str
    aliases: list[str]


class DrugsResponse(BaseModel):
    """Response for GET /drugs."""
    drugs:        list[DrugInfo]
    cancer_types: list[str]
    endpoints:    list[str]
    study_types:  list[str]


class HealthResponse(BaseModel):
    """Response for GET /health."""
    status:    str
    database:  str
    api_keys:  dict[str, bool]


# ─────────────────────────────────────────────────────────────────────
# 3. FASTAPI APP + LIFESPAN
# ─────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager — same pattern as mcp_server.py.
    Startup: nothing needed (pool created lazily on first query).
    Shutdown: release asyncpg connection pool.
    """
    yield
    await close_pool()
    print("API server shut down. DB pool closed.")


app = FastAPI(
    title="Pharma RAG API",
    description=(
        "Clinical literature research API powered by Agentic RAG. "
        "Searches 12,000+ PubMed abstracts across 5 oncology/pharma drugs."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow all origins for development
# Restrict in production to specific frontend domains
app.add_middleware(
    CORSMiddleware,              # ← Use FastAPI's CORS middleware
    allow_origins=["*"],         # ← Allow requests from ANY origin
    allow_credentials=True,      # ← Allow cookies/auth headers to be sent
    allow_methods=["*"],         # ← Allow ANY HTTP method (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],         # ← Allow ANY HTTP header in requests
)


# ─────────────────────────────────────────────────────────────────────
# 4. ENDPOINTS
# ─────────────────────────────────────────────────────────────────────

@app.post("/research", response_model=ResearchResponse)
async def research(request: ResearchRequest):
    """
    Run full multi-agent clinical research pipeline.

    Activates specialist agents (efficacy, safety, mechanism, competitor)
    based on the query. Each agent independently retrieves and grades
    evidence, then a synthesizer merges findings into a cited research brief.

    Takes 15-30 seconds depending on query complexity and number of agents.
    Each request produces one Langfuse trace when LANGFUSE_PUBLIC_KEY is set.
    """
    start      = time.perf_counter()
    request_id = str(uuid.uuid4())[:8]

    handler, invoke_config = get_callback_handler(                 # LANGFUSE
        session_id = f"research-{request_id}",
        trace_name = "research-pipeline",
        metadata   = {
            "endpoint":   "/research",
            "query":      request.query[:200],
            "request_id": request_id,
        },
    )

    try:
        output = await asyncio.wait_for(
            run_research(request.query, invoke_config=invoke_config), # LANGFUSE
            timeout=120,
        )
        elapsed = time.perf_counter() - start
        return ResearchResponse.from_agent_output(output, elapsed)

    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail="Pipeline timed out after 120 seconds. Try a simpler query.",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Research failed: {type(e).__name__}: {e}",
        )
    finally:
        flush_handler(handler)                                     # LANGFUSE


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    Search the knowledge base for clinical evidence.

    Returns raw retrieved and reranked evidence chunks with metadata.
    Faster than /research (~2-5 seconds) — no agents, no synthesis.

    Optionally pass explicit filters to narrow results.
    When no filters provided, the query rewriter extracts them automatically.
    """
    # Build MetadataFilter from explicit parameters if any provided
    filters = None
    if any([request.drug, request.cancer_type, request.year_min,
            request.study_type, request.endpoints]):
        filters = MetadataFilter(
            drug        = request.drug,
            cancer_type = request.cancer_type,
            year_min    = request.year_min,
            study_type  = request.study_type,
            endpoints   = request.endpoints,
        )

    request_id = str(uuid.uuid4())[:8]                            # LANGFUSE

    handler, _ = get_callback_handler(                            # LANGFUSE
        session_id = f"search-{request_id}",
        trace_name = "search-pipeline",
        metadata   = {
            "endpoint":    "/search",
            "query":       request.query[:200],
            "request_id":  request_id,
            "has_filters": filters is not None,
            "drug":        request.drug,
            "cancer_type": request.cancer_type,
        },
    )

    try:
        start = time.perf_counter()
        result = await retrieve(
            query        = request.query,
            filters      = filters,
            use_rewriter = (filters is None),
        )
        elapsed = time.perf_counter() - start

        chunks = [
            ChunkResponse.from_retrieved_chunk(c)
            for c in result.get("chunks", [])
        ]

        return SearchResponse(
            query           = result.get("original_query", request.query),
            rewritten_query = result.get("rewritten_query", request.query),
            confidence      = result.get("confidence", "unknown"),
            filter_level    = result.get("filter_level", "unknown"),
            n_candidates    = result.get("n_candidates", 0),
            n_returned      = result.get("n_returned", 0),
            chunks          = chunks,
            elapsed_seconds = round(elapsed, 2),
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {type(e).__name__}: {e}",
        )
    finally:
        flush_handler(handler)                                    # LANGFUSE


@app.get("/drugs", response_model=DrugsResponse)
async def list_drugs():
    """
    List all drugs available in the knowledge base with aliases,
    cancer types, endpoints, and study types.
    """
    drugs = []
    for drug in DB_VOCABULARY["drugs"]:
        aliases = [
            alias for alias, canonical
            in DB_VOCABULARY["drug_aliases"].items()
            if canonical == drug
        ]
        drugs.append(DrugInfo(name=drug, aliases=aliases))

    return DrugsResponse(
        drugs        = drugs,
        cancer_types = DB_VOCABULARY["cancer_types"],
        endpoints    = DB_VOCABULARY["endpoints"],
        study_types  = DB_VOCABULARY["study_types"],
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check — verifies database connectivity and API key presence.

    LANGFUSE: "langfuse" key added to api_keys (informational only).
    Health status requires only openai + cohere — langfuse is optional.
    Original used all(api_keys.values()) which would flip to "degraded"
    when langfuse keys are absent. New logic is explicit about what matters.
    """
    # Check database
    db_status = "unknown"
    try:
        from retriever import get_pool
        pool = await get_pool()
        async with pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        db_status = "connected"
    except Exception as e:
        db_status = f"error: {type(e).__name__}: {e}"

    # Check API keys (presence only, not validity)
    api_keys = {
        "openai":    bool(os.getenv("OPENAI_API_KEY", "").startswith("sk-")),
        "cohere":    bool(os.getenv("COHERE_API_KEY", "")),
        "langsmith": bool(os.getenv("LANGCHAIN_API_KEY", "").startswith("ls")),
        "langfuse":  langfuse_enabled(),                          # LANGFUSE
    }

    # LANGFUSE: healthy = db + openai + cohere only.
    # langfuse/langsmith are optional — not a health requirement.
    # (Original: all(api_keys.values()) — would break if langfuse absent)
    status = "healthy" if (
        db_status == "connected"
        and api_keys["openai"]
        and api_keys["cohere"]
    ) else "degraded"

    return HealthResponse(
        status   = status,
        database = db_status,
        api_keys = api_keys,
    )


# ─────────────────────────────────────────────────────────────────────
# 5. ENTRY POINT
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )