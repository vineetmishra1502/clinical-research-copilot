# retriever.py — Documentation

## Position in Pipeline

```
query_rewriter.py
        ↓
retriever.py   ← YOU ARE HERE
        ↓
grader.py → agents.py
```

---

## Architecture Flow

```
retrieve(query, filters=None, use_rewriter=True)
        │
        ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Decision: which path?                                              │
│                                                                      │
│  filters provided   → skip rewriter (agents pass pre-built filters) │
│  use_rewriter=True  → rewrite_query_with_subquestions(query)        │
│  use_rewriter=False → raw query + empty filters (testing only)      │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ rewritten_query + filters + rrf_weights
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Step 1: embed_query(rewritten_query)                               │
│  OpenAIEmbeddings.aembed_query() → [0.2, -0.4, ...] 3072 floats   │
│  Cleaner text → better vector than raw messy query                  │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ query_embedding
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Step 2: retrieve_with_fallback() — 5 progressive levels           │
│                                                                      │
│  L1 full      → all filters (drug+cancer+year+pdl1+endpoints)       │
│  L2 relaxed   → drop pdl1_status, funding, endpoints, study_type   │
│  L3 minimal   → drug + cancer_type + year_min only                 │
│  L4 drug_only → drug only                                          │
│  L5 no_filter → full corpus                                         │
│                                                                      │
│  Stop at first level returning ≥ MIN_RESULTS (5) chunks             │
│  Return (candidates, filter_level)                                  │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ each level calls ↓
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│  _hybrid_search_raw()   asyncio.gather() fires both simultaneously   │
│                                                                       │
│  ┌──────────────────────────┐   ┌──────────────────────────────────┐ │
│  │  dense_search()          │   │  bm25_search()                   │ │
│  │                          │   │                                  │ │
│  │  build_filter_clause()   │   │  build_filter_clause()           │ │
│  │  → SQL WHERE ($1,$2...)  │   │  → SQL WHERE + ts_content @@     │ │
│  │                          │   │                                  │ │
│  │  embedding_str =         │   │  _prepare_tsquery()              │ │
│  │  "[v1,v2,...]"           │   │  "pembrolizumab & survival &     │ │
│  │  (asyncpg HALFVEC fix)   │   │   nsclc & overall"               │ │
│  │                          │   │                                  │ │
│  │  HNSW cosine similarity  │   │  GIN full-text ts_rank()         │ │
│  │  1-(embedding<=>$N::hv)  │   │  keyword frequency + position    │ │
│  │  → top 20 + dense_score  │   │  → top 20 + bm25_score           │ │
│  └────────────┬─────────────┘   └──────────────┬───────────────────┘ │
│               └──────────────┬──────────────────┘                    │
│                              ↓                                        │
│               rrf_fusion(dense, bm25, dense_w, bm25_w)               │
│               score = dw × 1/(60+rank) + bw × 1/(60+rank)           │
│               chunks in both lists → double score (consensus)        │
│               → up to 40 unique chunks sorted by rrf_score           │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼ (if sub_questions exist)
┌─────────────────────────────────────────────────────────────────────┐
│  Step 3: sub-question multi-pass (comparison queries only)          │
│                                                                      │
│  "pembro vs nivo" → sub_questions = ["pembro OS NSCLC",             │
│                                      "nivo OS NSCLC"]               │
│  Each sub-question retrieves independently → candidates merged      │
│  Chunks in multiple sub-question results get highest RRF score      │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ up to 40 candidates
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Step 4: rerank(original_query, candidates)                         │
│                                                                      │
│  CohereRerank.compress_documents()                                   │
│  Cross-encoder: full query+doc attention (not separate encoding)    │
│  Scores against ORIGINAL query — user's actual intent               │
│  40 candidates → top 5 with rerank_score                            │
└──────────────────────────────┬──────────────────────────────────────┘
                               ▼
                    5 RetrievedChunk objects
            each with all 4 scores populated:
            dense_score, bm25_score, rrf_score, rerank_score
                    + query_embedding (for grader reuse)
```

---

## Step-by-Step Process

### Step 1 — Configuration and LangChain clients

```python
_embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
_reranker   = CohereRerank(model="rerank-english-v3.0", top_n=5)
_pool       = None   # asyncpg connection pool singleton
```

Module-level singletons — created once at import, reused every call. `OpenAIEmbeddings` must use the same model as ingestion (`text-embedding-3-large`) — query and document vectors must share the same 3072-dimensional vector space for cosine similarity to be meaningful.

---

### Step 2 — `RetrievedChunk` dataclass

```python
@dataclass
class RetrievedChunk:
    id:             int
    content:        str
    drug:           str
    evidence_level: int
    sample_size:    int | None
    # ... metadata fields
    dense_score:    float = 0.0
    bm25_score:     float = 0.0
    rrf_score:      float = 0.0
    rerank_score:   float = 0.0   # ← final ranking signal for grader
```

Four score fields default to `0.0` and are populated progressively:
- `dense_score` — after `dense_search()`
- `bm25_score` — after `bm25_search()`
- `rrf_score` — after `rrf_fusion()`
- `rerank_score` — after `rerank()` — the final ranking signal passed to `grader.py`

Dataclass (not Pydantic) because it comes from asyncpg database rows — already typed by Postgres schema. No external validation needed.

---

### Step 3 — Connection pool

```python
async def get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(min_size=3, max_size=10, ...)
    return _pool
```

Singleton pattern — first call creates the pool, all subsequent calls reuse it. Opening a cold connection costs ~80ms. `pool.acquire()` costs ~4ms. With `asyncio.gather()` firing two DB queries simultaneously, pool is critical — no warm connections means both queries open cold connections serially defeating the parallelism.

`asyncpg` not `psycopg2` — `asyncpg` is a truly async Postgres driver. `psycopg2` is synchronous — calling it in `asyncio.gather()` would block the event loop making the two queries effectively sequential despite using gather. `asyncpg` suspends and yields control while waiting for Postgres, allowing both queries to actually run in parallel.

---

### Step 4 — `build_filter_clause()`

```python
def _add(condition: str, value: object) -> None:
    idx = len(params) + 1
    conditions.append(condition.replace("?", f"${idx}"))
    params.append(value)

if f.drug:        _add("drug = ?", f.drug)
if f.cancer_type: _add("cancer_type = ?", f.cancer_type)
if f.endpoints:   _add("endpoints LIKE ?", f"%{f.endpoints}%")
```

Converts `MetadataFilter` → parameterized SQL WHERE clause. `_add()` auto-increments `$N` placeholders — asyncpg positional parameter syntax. `None` fields skipped — "don't filter" semantics. `endpoints LIKE "%OS%"` because endpoints are stored as CSV `"OS,PFS,ORR"` — exact match would miss multi-endpoint rows.

Applied **before** vector math: SQL B-Tree indexes on `drug`, `cancer_type`, `year`, `evidence_level` reduce 12,000 chunks to ~400 in ~3ms. HNSW cosine then runs only on those 400 — 90-95% search space reduction before any vector computation.

`$1, $2` parameterized queries prevent SQL injection — asyncpg sends the template and values as separate packets, Postgres never interprets values as SQL code.

---

### Step 5 — Dense search

```python
embedding_str = "[" + ",".join(str(v) for v in query_embedding) + "]"

rows = await conn.fetch(sql, *params, embedding_str)
```

**The `embedding_str` fix:** asyncpg has no encoder for Postgres `HALFVEC` — a custom type added by the pgvector extension. Passing `list[float]` directly raises a `TypeError`. Converting to the pgvector wire format string `"[v1,v2,...]"` lets asyncpg send it as a string. The `::halfvec` cast in the SQL tells Postgres to parse it as a HALFVEC vector.

`1 - (embedding <=> $N::halfvec)` — `<=>` is pgvector's cosine distance operator (0=identical, 1=opposite). Subtract from 1 to get cosine similarity (1=identical, 0=opposite). Chunks below `SIMILARITY_THRESHOLD=0.3` are filtered out before RRF.

---

### Step 6 — BM25 search

```python
def _prepare_tsquery(query: str) -> str:
    words     = re.findall(r'\b[a-zA-Z0-9]{2,}\b', query.lower())
    stopwords = {"the", "a", "an", "and", "or", ...}
    meaningful = [w for w in words if w not in stopwords]
    return " & ".join(meaningful)
    # "pembrolizumab overall survival NSCLC" → "pembrolizumab & overall & survival & nsclc"
```

Postgres `to_tsquery()` requires specific syntax — raw natural language fails. Stop words removed to reduce noise. Terms joined with `&` (AND — all must appear).

**SQL clause construction fix:** Two separate `WHERE` clauses produce invalid SQL. The correct construction:

```python
if clause:    # metadata filters exist
    full_clause = clause + f" AND ts_content @@ to_tsquery('english', ${tsq_idx})"
else:         # no metadata filters
    full_clause = f"WHERE ts_content @@ to_tsquery('english', ${tsq_idx})"
```

`tsq_idx` — the tsquery parameter index comes after all filter parameters. Referenced twice in the SQL (SELECT for scoring, WHERE for filtering) — Postgres allows the same `$N` parameter to appear multiple times.

---

### Step 7 — `get_rrf_weights()` and weighted RRF

```python
def get_rrf_weights(strategy: str) -> tuple[float, float]:
    return STRATEGY_WEIGHTS.get(strategy, (0.5, 0.5))
```

**Why LLM decides, not regex:** The old `classify_query()` used hardcoded regex patterns with arbitrary weights (`0.3`, `1.0`). The LLM in `query_rewriter.py` reads the full query in context and picks from 5 discrete strategies. Free floats imply false precision. Named strategies are validatable and explainable.

```python
def rrf_fusion(dense_results, bm25_results,
               dense_weight=1.0, bm25_weight=1.0, k=60):
    for rank, chunk in enumerate(dense_results, start=1):
        score += dense_weight × (1.0 / (60 + rank))
    for rank, chunk in enumerate(bm25_results, start=1):
        score += bm25_weight  × (1.0 / (60 + rank))
```

**Why RRF not score averaging:** Dense scores are cosine (0–1). BM25 scores are ts_rank (different scale, domain-specific). Averaging incomparable scales is unreliable. RRF normalizes via rank position — scale-independent, robust, proven in retrieval literature.

**Why k=60:** From the original RRF paper. Dampens rank-1 dominance without eliminating the benefit. Difference between rank 1 and rank 2: `1/61 - 1/62 ≈ 0.00026`. Meaningful but not overwhelming.

**Chunk deduplication:** Same `chunk.id` appearing in both dense and BM25 results accumulates score from both — natural consensus signal. A chunk retrieved by both methods is more reliably relevant than one retrieved by only one.

---

### Step 8 — Progressive fallback

```python
# L1: full filters → L2: drop edge filters → L3: minimal → L4: drug only → L5: no filter
results = await _hybrid_search_raw(query, embedding, filters, top_k, rrf_weights)
if len(results) >= MIN_RESULTS:
    return results, "full"
```

**Why 5 levels:** A query for "pembrolizumab NSCLC TPS>=50% industry-funded RCT post-2021" might match only 2 chunks. Without fallback the pipeline returns empty to the grader. Progressive relaxation drops the most specific filters first (pdl1_status, funding) while keeping the most meaningful (drug, cancer_type).

`filter_level` returned alongside results — the grader uses it to adjust faithfulness thresholds. `"no_filter"` retrieval means the chunks are less targeted and the grader requires stronger evidence quality to compensate.

`rrf_weights` passed through all 5 fallback levels unchanged — the query type doesn't change because filters were relaxed.

---

### Step 9 — Cohere reranker

```python
docs = [
    Document(page_content=chunk.content, metadata={"chunk_index_in_candidates": i})
    for i, chunk in enumerate(candidates)
]

reranked_docs = _reranker.compress_documents(documents=docs, query=query)

for doc in reranked_docs:
    original_idx       = doc.metadata["chunk_index_in_candidates"]
    chunk.rerank_score = float(doc.metadata.get("relevance_score", 0.0))
```

**Two-stage pipeline:** Bi-encoder (dense + BM25) computes query and document embeddings separately — fast but less accurate. Cross-encoder (Cohere) reads query and document together with full attention — slower but significantly more accurate. Running cross-encoder on all 12,000 chunks would be too slow (~minutes). Running it on 40 RRF candidates takes ~500ms.

`chunk_index_in_candidates` stored in metadata — Cohere returns documents in a different order than input. The index maps results back to original `RetrievedChunk` objects.

**Why rerank against `original_query`, not `rewritten_query`:** Cohere judges relevance to what the user actually asked. "Pembro OS NSCLC" is the user's intent. "pembrolizumab overall survival NSCLC" is an internal optimization for retrieval. Reranking against the expansion could over-weight expanded terms.

---

### Step 10 — `retrieve()` master function

```python
return {
    "chunks":          final_chunks,
    "filter_level":    filter_level,
    "filters_applied": filters,
    "rewritten_query": rewritten_query,
    "rrf_weights":     rrf_weights,
    "query_embedding": query_embedding,   # ← reused by grader for re-query
    ...
}
```

`query_embedding` included in the return dict. When the grader decides to re-query, it can pass the embedding back to avoid a second OpenAI embedding call for the same original query. ~200ms saved per re-query attempt.

---

## Data contracts

| Input | Source | Purpose |
|---|---|---|
| `rewritten_query` | `query_rewriter.py` | embedding + BM25 text |
| `MetadataFilter` | `query_rewriter.py` | SQL WHERE clause |
| `rrf_weights` | `query_rewriter.py` | RRF fusion weighting |
| `sub_questions` | `query_rewriter.py` | multi-pass retrieval |

| Output | Type | Consumer |
|---|---|---|
| `chunks` | `list[RetrievedChunk]` | `grader.grade_chunks()` |
| `filter_level` | `str` | `grader` faithfulness threshold |
| `query_embedding` | `list[float]` | `grader` re-query reuse |
| `rrf_weights` | `tuple[float, float]` | logging / debugging |

---

## Key decisions

**Why hybrid (dense + BM25):** Dense excels at semantic similarity — "OS improvement" finds "survival benefit." BM25 excels at exact term matching — "KEYNOTE-189" and "NCT02578680" require character-for-character matches. Queries about clinical trials use both types of language simultaneously.

**Why asyncpg:** Truly async Postgres driver — `asyncio.gather()` fires dense and BM25 queries simultaneously. `psycopg2` is synchronous — gather would be sequential despite the code appearing parallel.

**Why `embedding_str` conversion:** asyncpg has no encoder for the pgvector `HALFVEC` extension type. String conversion + `::halfvec` SQL cast is the correct pattern for sending vector data through asyncpg.

