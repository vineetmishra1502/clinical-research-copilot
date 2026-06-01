"""
retriever.py — Hybrid Retrieval Layer (Python 3.13)
=====================================================
Python 3.13 optimisations applied:
  - str | None  instead of Optional[str]    — built-in union syntax
  - list[...]   instead of List[...]        — built-in generics
  - tuple[...]  instead of Tuple[...]       — built-in generics
  - dict[...]   instead of Dict[...]        — built-in generics
  - No typing imports for Optional/List/Dict/Tuple/Set

LangChain components used (consistent with rest of stack):
  - OpenAIEmbeddings.aembed_query() — async query embedding
  - CohereRerank.compress_documents() — cross-encoder reranking
  - Both auto-traced in LangSmith alongside chain traces

Architecture:
  User query
    → query_rewriter.py   (normalize + extract filters)
    → retriever.py        (dense + BM25 + weighted RRF + rerank)
    → grader.py           (evaluate + self-correct)
    → agents.py           (orchestrate)
"""

import os
import re
import asyncio
import asyncpg
from dataclasses import dataclass
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_cohere import CohereRerank
from langchain_core.documents import Document

from query_rewriter import (
    rewrite_query_with_subquestions,
    MetadataFilter,
    RewriterOutput,
    STRATEGY_WEIGHTS,
)

load_dotenv()


# ─────────────────────────────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────────────────────────────

EMBEDDING_MODEL      = "text-embedding-3-large"
RERANKER_MODEL       = "rerank-english-v3.0"
TOP_K_RETRIEVAL      = 20
TOP_K_RERANKED       = 5
MIN_RESULTS          = 5
RRF_K                = 60
SIMILARITY_THRESHOLD = 0.3

# LangChain embedding client — same model as ingestion (shared vector space)
# aembed_query() is async — non-blocking, LangSmith-traced
_embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

# LangChain Cohere reranker — cross-encoder, LangSmith-traced
# compress_documents() is the standard LangChain reranker interface
_reranker = CohereRerank(
    model          = RERANKER_MODEL,
    top_n          = TOP_K_RERANKED,
    cohere_api_key = os.getenv("COHERE_API_KEY"),
)


# ─────────────────────────────────────────────────────────────────────
# 2. RETRIEVED CHUNK DATACLASS
# ─────────────────────────────────────────────────────────────────────
# Python 3.13: str | None replaces Optional[str] throughout.
# int | None replaces Optional[int].

@dataclass
class RetrievedChunk:
    """
    Single retrieved chunk with metadata and all pipeline scores.
    Scores populated progressively:
      dense_score  → after dense_search()
      bm25_score   → after bm25_search()
      rrf_score    → after rrf_fusion()
      rerank_score → after rerank() — final ranking signal for grader
    """
    id:             int
    content:        str
    drug:           str
    cancer_type:    str
    study_type:     str
    evidence_level: int
    sample_size:    int | None
    journal:        str
    year:           int
    source:         str
    doi:            str
    endpoints:      str
    pdl1_status:    str
    funding:        str
    chunk_index:    int
    total_chunks:   int
    dense_score:    float = 0.0
    bm25_score:     float = 0.0
    rrf_score:      float = 0.0
    rerank_score:   float = 0.0


# ─────────────────────────────────────────────────────────────────────
# 3. DATABASE CONNECTION POOL
# ─────────────────────────────────────────────────────────────────────

_pool: asyncpg.Pool | None = None    # asyncpg.Pool | None — Python 3.13 syntax


async def get_pool() -> asyncpg.Pool:
    """
    Singleton asyncpg connection pool.
    Created on first call, reused for every subsequent query.

    Why asyncpg not psycopg2:
      asyncio.gather() in _hybrid_search_raw() fires dense + BM25
      simultaneously. psycopg2 is synchronous — would block the event loop
      making gather() effectively sequential. asyncpg is truly async.

    Pool (min=3, max=10):
      Opening a connection costs ~80ms. Pool keeps 3 always open.
      pool.acquire() costs ~4ms vs 80ms for a cold connection open.
    """
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(
            host     = os.getenv("DB_HOST", "localhost"),
            port     = int(os.getenv("DB_PORT", "5432")),
            database = os.getenv("DB_NAME", "pharma_rag"),
            user     = os.getenv("DB_USER", "vineet"),
            password = os.getenv("DB_PASSWORD", "devpassword"),
            min_size = 3,
            max_size = 10,
        )
    return _pool


async def close_pool() -> None:
    """Release DB connections on application shutdown."""
    global _pool
    if _pool:
        await _pool.close()
        _pool = None


# ─────────────────────────────────────────────────────────────────────
# 4. FILTER → SQL UTILITIES
# ─────────────────────────────────────────────────────────────────────

def build_filter_clause(f: MetadataFilter) -> tuple[str, list]:
    """
    Converts MetadataFilter → parameterized SQL WHERE clause.
    Returns (clause_string, params_list).

    Uses asyncpg $1, $2... positional parameters (not psycopg2 %s).
    None fields skipped — "don't filter" semantics, not "filter for NULL".
    endpoints uses LIKE '%OS%' — stored as CSV "OS,PFS,ORR" in DB.

    Applied BEFORE vector math:
      Postgres evaluates WHERE using B-Tree indexes on drug, year,
      study_type, evidence_level → identifies matching row IDs.
      HNSW cosine then runs only on those rows.
      drug + cancer_type filter → 90-95% search space reduction.
    """
    conditions: list[str] = []
    params: list           = []

    def _add(condition: str, value: object) -> None:
        idx = len(params) + 1
        conditions.append(condition.replace("?", f"${idx}"))
        params.append(value)

    if f.drug:               _add("drug = ?", f.drug)
    if f.cancer_type:        _add("cancer_type = ?", f.cancer_type)
    if f.study_type:         _add("study_type = ?", f.study_type)
    if f.year_min:           _add("year >= ?", f.year_min)
    if f.year_max:           _add("year <= ?", f.year_max)
    if f.evidence_level_max: _add("evidence_level <= ?", f.evidence_level_max)
    if f.endpoints:          _add("endpoints LIKE ?", f"%{f.endpoints}%")
    if f.pdl1_status:        _add("pdl1_status = ?", f.pdl1_status)
    if f.funding:            _add("funding = ?", f.funding)

    if not conditions:
        return "", []
    return "WHERE " + " AND ".join(conditions), params


def _row_to_chunk(
    row: asyncpg.Record,
    score_field: str,
    score_value: float,
) -> RetrievedChunk:
    """Converts asyncpg row → RetrievedChunk with one score attached."""
    chunk = RetrievedChunk(
        id             = row["id"],
        content        = row["content"],
        drug           = row["drug"] or "",
        cancer_type    = row["cancer_type"] or "",
        study_type     = row["study_type"] or "",
        evidence_level = row["evidence_level"] or 2,
        sample_size    = row["sample_size"],
        journal        = row["journal"] or "",
        year           = row["year"] or 0,
        source         = row["source"] or "",
        doi            = row["doi"] or "",
        endpoints      = row["endpoints"] or "",
        pdl1_status    = row["pdl1_status"] or "",
        funding        = row["funding"] or "",
        chunk_index    = row["chunk_index"] or 0,
        total_chunks   = row["total_chunks"] or 1,
    )
    setattr(chunk, score_field, score_value)
    return chunk


# ─────────────────────────────────────────────────────────────────────
# 5. QUERY CLASSIFICATION — sets RRF weights
# ─────────────────────────────────────────────────────────────────────

def get_rrf_weights(strategy: str) -> tuple[float, float]:
    """
    Maps a SearchStrategyLiteral string → (dense_weight, bm25_weight).

    Strategy is chosen by the LLM in query_rewriter.py as part of
    RewriterOutput.search_strategy — not by hardcoded regex here.

    Why LLM decides instead of regex:
      The old classify_query() used hardcoded regex patterns with
      arbitrary weights (0.3, 1.0) that had no empirical justification.
      The LLM reads the full query in context, understands intent, and
      picks from 5 discrete strategies with clear semantic meaning.

    Why discrete strategies not free floats:
      Free floats (0.743, 0.21) imply false precision — the LLM has
      no basis for that granularity. Five named strategies are
      validatable, explainable, and constrained by Literal type.

    Strategy → weights:
      dense_only   (1.0, 0.0) — pure concept/mechanism, no keyword noise
      dense_heavy  (0.7, 0.3) — semantic-leaning general clinical query
      equal        (0.5, 0.5) — balanced, default for most queries
      bm25_heavy   (0.3, 0.7) — specific known terms, keyword-leaning
      bm25_only    (0.0, 1.0) — exact identifier lookup (NCT, trial name)

    Falls back to equal (0.5, 0.5) for any unrecognized strategy string.
    """
    return STRATEGY_WEIGHTS.get(strategy, (0.5, 0.5))


# ─────────────────────────────────────────────────────────────────────
# 6. EMBEDDING
# ─────────────────────────────────────────────────────────────────────

async def embed_query(query: str) -> list[float]:
    """
    Embeds query using LangChain OpenAIEmbeddings (text-embedding-3-large).

    Why LangChain OpenAIEmbeddings over raw AsyncOpenAI:
    - aembed_query() is async — non-blocking event loop
    - Auto-traced in LangSmith alongside chain and retrieval traces
    - Provider-agnostic — switching to AzureOpenAI requires one line change
    - Same model as ingestion — query + document vectors share vector space

    list[float] — Python 3.13 built-in generic, no List import.
    """
    return await _embeddings.aembed_query(query)


# ─────────────────────────────────────────────────────────────────────
# 7. DENSE SEARCH
# ─────────────────────────────────────────────────────────────────────

async def dense_search(
    query_embedding: list[float],
    filters: MetadataFilter,
    top_k: int = TOP_K_RETRIEVAL,
) -> list[RetrievedChunk]:
    """
    HNSW cosine similarity search on HALFVEC(3072) column.
    Metadata filters applied in SQL WHERE BEFORE vector ranking.

    embedding_str fix:
      asyncpg cannot encode list[float] as Postgres HALFVEC.
      Convert to pgvector wire format "[v1,v2,...]" string.
      ::halfvec cast in SQL tells Postgres to parse as HALFVEC.

    Filter-first rationale:
      WHERE evaluated using B-Tree indexes → matching row IDs.
      HNSW then runs on matching rows only (not all 12,000).
      Typical reduction: 90-95% of corpus filtered before vector math.

    <=> is pgvector cosine distance (0=identical, 1=opposite).
    Score = 1 - distance → higher = more similar.
    """
    pool           = await get_pool()
    clause, params = build_filter_clause(filters)
    emb_idx        = len(params) + 1

    sql = f"""
        SELECT
            id, content, drug, cancer_type, study_type,
            evidence_level, sample_size, journal, year,
            source, doi, endpoints, pdl1_status, funding,
            chunk_index, total_chunks,
            1 - (embedding <=> ${emb_idx}::halfvec) AS score
        FROM documents
        {clause}
        ORDER BY embedding <=> ${emb_idx}::halfvec
        LIMIT {top_k};
    """

    # asyncpg cannot encode list[float] as HALFVEC.
    # Convert to pgvector wire format "[v1,v2,...]" string.
    # ::halfvec cast in SQL parses it as HALFVEC on Postgres side.
    embedding_str = "[" + ",".join(str(v) for v in query_embedding) + "]"

    async with pool.acquire() as conn:
        rows = await conn.fetch(sql, *params, embedding_str)

    return [
        _row_to_chunk(row, "dense_score", float(row["score"]))
        for row in rows
        if float(row["score"]) >= SIMILARITY_THRESHOLD
    ]


# ─────────────────────────────────────────────────────────────────────
# 8. BM25 SEARCH
# ─────────────────────────────────────────────────────────────────────

def _prepare_tsquery(query: str) -> str:
    """
    Converts natural language → Postgres tsquery syntax.
    Extracts meaningful words, removes stop words, joins with & (AND).
    "pembrolizumab overall survival NSCLC" → "pembrolizumab & overall & survival & nsclc"
    """
    words     = re.findall(r'\b[a-zA-Z0-9]{2,}\b', query.lower())
    stopwords = {
        "the","a","an","and","or","but","in","on","at","to","for",
        "of","with","by","from","is","are","was","were","be","been",
        "have","has","do","does","what","how","why","when","where",
        "which","who","that","this","these","those","can","will",
        "would","should","could","may","might","did",
    }
    meaningful = [w for w in words if w not in stopwords]
    return " & ".join(meaningful) if meaningful else query


async def bm25_search(
    query: str,
    filters: MetadataFilter,
    top_k: int = TOP_K_RETRIEVAL,
) -> list[RetrievedChunk]:
    """
    Full-text BM25 search via Postgres tsvector + GIN index.
    Same WHERE filters applied as dense search.

    SQL structure:
      If metadata WHERE exists: append "AND ts_content @@ ..."
      If no metadata WHERE: use "WHERE ts_content @@ ..."
      Combining correctly avoids SQL syntax errors from two WHERE clauses.

    ts_rank() — BM25-like scoring: term frequency + length normalization.
    @@ operator — full-text match using GIN index (fast).
    tsquery parameter appears once — $tsq_idx referenced in both
    SELECT (score) and WHERE (filter). Postgres allows same param twice.
    """
    pool           = await get_pool()
    clause, params = build_filter_clause(filters)
    tsquery        = _prepare_tsquery(query)
    tsq_idx        = len(params) + 1

    # Combine metadata WHERE with BM25 condition correctly
    if clause:
        full_clause = clause + f" AND ts_content @@ to_tsquery('english', ${tsq_idx})"
    else:
        full_clause = f"WHERE ts_content @@ to_tsquery('english', ${tsq_idx})"

    sql = f"""
        SELECT
            id, content, drug, cancer_type, study_type,
            evidence_level, sample_size, journal, year,
            source, doi, endpoints, pdl1_status, funding,
            chunk_index, total_chunks,
            ts_rank(ts_content, to_tsquery('english', ${tsq_idx})) AS score
        FROM documents
        {full_clause}
        ORDER BY score DESC
        LIMIT {top_k};
    """

    async with pool.acquire() as conn:
        try:
            rows = await conn.fetch(sql, *params, tsquery)
        except Exception:
            # tsquery syntax error on unusual input — degrade gracefully
            return []

    return [
        _row_to_chunk(row, "bm25_score", float(row["score"]))
        for row in rows
    ]


# ─────────────────────────────────────────────────────────────────────
# 9. WEIGHTED RRF FUSION
# ─────────────────────────────────────────────────────────────────────

def rrf_fusion(
    dense_results: list[RetrievedChunk],
    bm25_results:  list[RetrievedChunk],
    dense_weight:  float = 1.0,
    bm25_weight:   float = 1.0,
    k:             int   = RRF_K,
) -> list[RetrievedChunk]:
    """
    Weighted Reciprocal Rank Fusion — merges two ranked lists.

    Formula: RRF_score = dense_w × 1/(k+rank_dense)
                       + bm25_w  × 1/(k+rank_bm25)

    Why weighted not equal:
      classify_query() sets weights based on query type:
        Exact IDs (NCT, trial names): (0.3, 1.0) BM25 dominant
        Conceptual (mechanism, how):  (1.0, 0.3) Dense dominant
        General clinical:             (1.0, 1.0) Equal

    Why RRF over weighted raw-score average:
      Dense scores are cosine (0-1). BM25 scores are ts_rank (different scale).
      Averaging incomparable scales is unreliable. RRF normalises via
      rank position — scale-independent and robust.

    k=60: universal constant from original RRF paper.
      Dampens rank-1 dominance. Difference rank1 vs rank2: ~0.000264.
      Meaningful but not overwhelming. Robust to outliers.

    Deduplication: same chunk_id from both lists merged into one entry.
    Chunks in both lists accumulate score from both — highest confidence.

    dict[int, float] — Python 3.13 built-in generic, no Dict import.
    """
    rrf_scores: dict[int, float]          = {}
    chunks_map: dict[int, RetrievedChunk] = {}

    for rank, chunk in enumerate(dense_results, start=1):
        contribution         = dense_weight * (1.0 / (k + rank))
        rrf_scores[chunk.id] = rrf_scores.get(chunk.id, 0.0) + contribution
        chunks_map[chunk.id] = chunk

    for rank, chunk in enumerate(bm25_results, start=1):
        contribution         = bm25_weight * (1.0 / (k + rank))
        rrf_scores[chunk.id] = rrf_scores.get(chunk.id, 0.0) + contribution
        if chunk.id not in chunks_map:
            chunks_map[chunk.id] = chunk

    sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)

    fused = []
    for chunk_id in sorted_ids:
        chunk           = chunks_map[chunk_id]
        chunk.rrf_score = rrf_scores[chunk_id]
        fused.append(chunk)

    return fused


# ─────────────────────────────────────────────────────────────────────
# 10. COHERE RERANKER
# ─────────────────────────────────────────────────────────────────────

async def rerank(
    query:      str,
    candidates: list[RetrievedChunk],
    top_n:      int = TOP_K_RERANKED,
) -> list[RetrievedChunk]:
    """
    Cross-encoder reranking via LangChain CohereRerank.

    Two-stage pipeline:
      Stage 1 — bi-encoder (fast): cosine similarity, 20 candidates
      Stage 2 — cross-encoder (accurate): full query+doc attention, top 5

    Why LangChain CohereRerank:
    - compress_documents() is LangChain's standard reranker interface
      — same interface used in LangGraph nodes, ContextualCompressionRetriever
    - Returns Document objects with metadata["relevance_score"]
    - Auto-traced in LangSmith alongside other retrieval traces
    - Swapping reranker (FlashRank, RankGPT) requires one line change

    Reranks against original_query (user's intent), not rewritten query.
    Cohere judges relevance to what the user actually asked.

    Fallback: RRF order if Cohere API unavailable.
    """
    if not candidates:
        return []

    # Wrap content in LangChain Documents
    # Store original index in metadata — reranked order differs from input
    docs = [
        Document(
            page_content = chunk.content,
            metadata     = {"chunk_index_in_candidates": i},
        )
        for i, chunk in enumerate(candidates)
    ]

    try:
        # compress_documents: LangChain standard reranker interface
        # Returns top_n Documents sorted by Cohere relevance score
        # compress_documents() is synchronous — blocks the event loop for ~500ms.
        # asyncio.to_thread() runs it in a thread pool so the event loop
        # stays responsive (critical for MCP stdio transport).
        reranked_docs = await asyncio.to_thread(
            _reranker.compress_documents,
            docs,
            query,
        )

        reranked: list[RetrievedChunk] = []
        for doc in reranked_docs:
            original_idx       = doc.metadata["chunk_index_in_candidates"]
            chunk              = candidates[original_idx]
            chunk.rerank_score = float(doc.metadata.get("relevance_score", 0.0))
            reranked.append(chunk)

        return reranked

    except Exception as e:
        print(f"  Reranker error: {e}. Falling back to RRF order.")
        for chunk in candidates[:top_n]:
            chunk.rerank_score = chunk.rrf_score
        return candidates[:top_n]


# ─────────────────────────────────────────────────────────────────────
# 11. PARALLEL SEARCH + WEIGHTED RRF
# ─────────────────────────────────────────────────────────────────────

async def _hybrid_search_raw(
    query:           str,
    query_embedding: list[float],
    filters:         MetadataFilter,
    top_k:           int,
    rrf_weights:     tuple[float, float] = (0.5, 0.5),
) -> list[RetrievedChunk]:
    """
    Fires dense + BM25 in parallel via asyncio.gather().
    Applies LLM-chosen RRF weights from rrf_weights parameter.
    Fuses results with weighted RRF.

    rrf_weights: (dense_weight, bm25_weight) from RewriterOutput.get_rrf_weights()
    Passed in from retrieve() — not determined here by regex/heuristic.

    asyncio.gather() time = max(dense_time, bm25_time), not sum.
    With asyncpg (truly async): ~200ms each → gather takes ~200ms.
    """
    dense_weight, bm25_weight = rrf_weights

    dense_results, bm25_results = await asyncio.gather(
        dense_search(query_embedding, filters, top_k),
        bm25_search(query, filters, top_k),
    )

    return rrf_fusion(
        dense_results, bm25_results,
        dense_weight = dense_weight,
        bm25_weight  = bm25_weight,
    )


# ─────────────────────────────────────────────────────────────────────
# 12. PROGRESSIVE FALLBACK
# ─────────────────────────────────────────────────────────────────────

async def retrieve_with_fallback(
    query:           str,
    query_embedding: list[float],
    filters:         MetadataFilter,
    top_k:           int = TOP_K_RETRIEVAL,
    rrf_weights:     tuple[float, float] = (0.5, 0.5),
) -> tuple[list[RetrievedChunk], str]:
    """
    Progressive filter relaxation — 5 levels, most to least restrictive.
    Returns (candidates, filter_level_used).

    rrf_weights: passed through from retrieve() — set by LLM in rewriter.
    All fallback levels use the same weights — the query type doesn't
    change just because filters were relaxed.

    Why fallback:
      Specific filters → small intersection → few/zero BM25 keyword matches.
      Without fallback: pipeline returns empty to grader.
      With fallback: always ≥ MIN_RESULTS candidates.

    filter_level → passed to grader:
      "full"     → highest trust in retrieved evidence
      "no_filter"→ lower trust, grader applies stricter threshold

    Levels:
      full      — all filters as provided by rewriter
      relaxed   — drop pdl1_status, funding, endpoints, study_type
      minimal   — drug + cancer_type + year_min only
      drug_only — drug only (minimum meaningful context)
      no_filter — full corpus (always has results)
    """
    # L1: full filters
    results = await _hybrid_search_raw(query, query_embedding, filters, top_k, rrf_weights)
    if len(results) >= MIN_RESULTS:
        return results, "full"

    # L2: drop edge filters
    l2 = MetadataFilter(
        drug               = filters.drug,
        cancer_type        = filters.cancer_type,
        year_min           = filters.year_min,
        year_max           = filters.year_max,
        evidence_level_max = filters.evidence_level_max,
    )
    results = await _hybrid_search_raw(query, query_embedding, l2, top_k, rrf_weights)
    if len(results) >= MIN_RESULTS:
        return results, "relaxed"

    # L3: drug + cancer_type + year_min
    l3 = MetadataFilter(
        drug        = filters.drug,
        cancer_type = filters.cancer_type,
        year_min    = filters.year_min,
    )
    results = await _hybrid_search_raw(query, query_embedding, l3, top_k, rrf_weights)
    if len(results) >= MIN_RESULTS:
        return results, "minimal"

    # L4: drug only
    if filters.drug:
        l4      = MetadataFilter(drug=filters.drug)
        results = await _hybrid_search_raw(query, query_embedding, l4, top_k, rrf_weights)
        if len(results) >= MIN_RESULTS:
            return results, "drug_only"

    # L5: no filters — full corpus
    results = await _hybrid_search_raw(
        query, query_embedding, MetadataFilter(), top_k, rrf_weights
    )
    return results, "no_filter"


# ─────────────────────────────────────────────────────────────────────
# 13. MASTER RETRIEVE FUNCTION
# ─────────────────────────────────────────────────────────────────────

async def retrieve(
    query:          str,
    filters:        MetadataFilter | None = None,   # X | None — Python 3.13
    use_rewriter:   bool  = True,
    top_k:          int   = TOP_K_RETRIEVAL,
    top_n_reranked: int   = TOP_K_RERANKED,
) -> dict:
    """
    Single entry point for all callers (grader, agents, API).

    Three decision paths:
      filters provided    → skip rewriter (specialist agents use this)
      use_rewriter=True   → rewrite_query_with_subquestions() normalizes
                            query, extracts filters, returns sub_questions
      use_rewriter=False  → use original query + empty filters

    Why embed rewritten_query not original:
      Cleaner text → better vector. "Pembro OS NSC" embeds worse than
      "pembrolizumab overall survival NSCLC".

    Why rerank against original_query:
      Cohere judges relevance to user's actual intent.
      Rewriting is internal — user typed "Pembro OS NSC", Cohere scores
      against that, not the expansion.

    query_embedding returned in dict:
      Grader can reuse it for re-query without another OpenAI call.

    Returns dict[str, ...] — Python 3.13 built-in generic.
    """
    original_query = query
    sub_questions: list[str] = []
    rewriter_output: RewriterOutput | None = None
    confidence  = "medium"
    reasoning   = ""

    rrf_weights: tuple[float, float] = (0.5, 0.5)   # default: equal

    if filters is not None:
        # Explicit filters — skip rewriter (specialist agent use case)
        # Weights stay at default (0.5, 0.5) — no LLM decision available
        rewritten_query = query
    elif use_rewriter:
        # Rewriter normalizes query AND chooses search strategy
        rewriter_output, filters, sub_questions, rrf_weights = (
            await rewrite_query_with_subquestions(query)
        )
        rewritten_query = rewriter_output.rewritten_query
        confidence      = rewriter_output.confidence
        reasoning       = rewriter_output.reasoning
    else:
        filters         = MetadataFilter()
        rewritten_query = query

    # Embed rewritten query — cleaner text → better vector
    query_embedding = await embed_query(rewritten_query)

    # Primary retrieval pass — rrf_weights from LLM (or default if no rewriter)
    candidates, filter_level = await retrieve_with_fallback(
        rewritten_query, query_embedding, filters, top_k, rrf_weights
    )

    # Sub-question multi-pass (comparison queries)
    if sub_questions:
        # Embed all sub-questions in parallel
        sub_embeddings: list[list[float]] = await asyncio.gather(
            *[embed_query(sq) for sq in sub_questions]
        )

        # Retrieve for each sub-question in parallel
        # Same rrf_weights — query type doesn't change per sub-question
        sub_retrieval_results = await asyncio.gather(*[
            retrieve_with_fallback(sq, emb, filters, top_k // 2, rrf_weights)
            for sq, emb in zip(sub_questions, sub_embeddings)
        ])

        # Merge all candidates — chunks in multiple sub-questions rank higher
        all_candidates = list(candidates)
        for sub_candidates, _ in sub_retrieval_results:
            all_candidates.extend(sub_candidates)

        # Use same LLM-chosen weights for the merged sub-question fusion
        dense_weight, bm25_weight = rrf_weights
        candidates = rrf_fusion(
            all_candidates, [],
            dense_weight = dense_weight,
            bm25_weight  = bm25_weight,
        )

    # Rerank against ORIGINAL query — user's actual intent, not rewritten
    final_chunks = await rerank(original_query, candidates, top_n_reranked)

    return {
        "chunks":          final_chunks,
        "filter_level":    filter_level,
        "filters_applied": filters,
        "rewritten_query": rewritten_query,
        "sub_questions":   sub_questions,
        "confidence":      confidence,
        "reasoning":       reasoning,
        "original_query":  original_query,
        "n_candidates":    len(candidates),
        "n_returned":      len(final_chunks),
        "query_embedding": query_embedding,
        "rrf_weights":     rrf_weights,   # (dense_w, bm25_w) — for logging/debugging
    }


# ─────────────────────────────────────────────────────────────────────
# 14. QUICK TEST
# ─────────────────────────────────────────────────────────────────────

async def _test() -> None:
    """Quick smoke test — python retriever.py"""
    print("Retriever smoke test (Python 3.13)\n" + "=" * 55)

    test_cases: list[tuple[str, bool, MetadataFilter | None]] = [
        ("Pembro OS benefit in NSC patients PD-L1 high", True, None),
        ("pembrolizumab vs nivolumab NSCLC survival", True, None),
        ("KEYNOTE-189 primary endpoint overall survival", True, None),
        ("overall survival benefit", False, MetadataFilter(
            drug="pembrolizumab",
            cancer_type="nsclc",
            evidence_level_max=2,
            year_min=2018,
        )),
    ]

    for query, use_rw, explicit_filters in test_cases:
        print(f"\nQuery      : {query}")
        result = await retrieve(query, filters=explicit_filters, use_rewriter=use_rw)
        print(f"Rewritten  : {result['rewritten_query']}")
        print(f"Confidence : {result['confidence']}")
        print(f"Drug filter: {result['filters_applied'].drug}")
        print(f"Cancer     : {result['filters_applied'].cancer_type}")
        print(f"Sub-Qs     : {result['sub_questions']}")
        dense_w, bm25_w = result['rrf_weights']
        
        print(f"RRF weights: dense={dense_w}  bm25={bm25_w}")
        print(f"Candidates : {result['n_candidates']}  "
              f"Returned: {result['n_returned']}  "
              f"Level: {result['filter_level']}")

        for i, chunk in enumerate(result["chunks"], 1):
            print(f"  Chunk {i}: [{chunk.study_type:<20}] "
                  f"[{chunk.cancer_type:<20}] "
                  f"{chunk.year}  "
                  f"n={chunk.sample_size or 'N/A':<6}  "
                  f"rerank={chunk.rerank_score:.3f}")

    await close_pool()


if __name__ == "__main__":
    asyncio.run(_test())