# api/main.py — Documentation

## Position in Pipeline

```
All pipeline files (query_rewriter, retriever, grader, agents)
        ↓
api/main.py   ← YOU ARE HERE (HTTP interface layer)
        ↓
Client (curl, Postman, browser, frontend app)
```

---

## Architecture Flow

```
Client (curl / Postman / frontend)
        |
        |  HTTP JSON
        v
+---------------------------------------------------------------------+
|  api/main.py — FastAPI application                                  |
|                                                                      |
|  +----------------------------+    +----------------------------+   |
|  |  POST /research            |    |  POST /search              |   |
|  |                            |    |                            |   |
|  |  "Answer my question"      |    |  "Show me papers"          |   |
|  |  Full: ~15-30 seconds      |    |  Fast: ~2-5 seconds        |   |
|  |  Cited markdown report     |    |  Raw chunks + scores       |   |
|  |                            |    |                            |   |
|  |  Calls:                    |    |  Calls:                    |   |
|  |  agents.run_research()     |    |  retriever.retrieve()      |   |
|  |    supervisor -> agents    |    |    rewriter -> embed       |   |
|  |    -> synthesizer          |    |    -> dense + bm25         |   |
|  |                            |    |    -> rrf -> rerank        |   |
|  +----------------------------+    +----------------------------+   |
|                                                                      |
|  +----------------------------+    +----------------------------+   |
|  |  GET /drugs                |    |  GET /health               |   |
|  |                            |    |                            |   |
|  |  Lists available drugs     |    |  DB + API key check        |   |
|  |  with aliases, cancer      |    |  Returns: healthy /        |   |
|  |  types, endpoints          |    |  degraded + details        |   |
|  |                            |    |                            |   |
|  |  Reads: DB_VOCABULARY      |    |  Tests: asyncpg pool       |   |
|  +----------------------------+    +----------------------------+   |
|                                                                      |
|  Lifespan:                                                          |
|  shutdown -> close_pool() releases asyncpg DB connections            |
+---------------------------------------------------------------------+
```

---

## Step-by-Step Process

### Step 1 — Pydantic request models

Two request models validate incoming JSON:

**`ResearchRequest`** — just a query string with `min_length=3`. The supervisor decides everything else (which agents, which filters).

**`SearchRequest`** — query string plus optional explicit filters (drug, cancer_type, year_min, study_type, endpoints). When filters are provided, the query rewriter is skipped. When no filters provided, the rewriter extracts them from the query text.

### Step 2 — Pydantic response models (the dataclass -> Pydantic boundary)

This is where internal dataclasses are converted to JSON-serializable Pydantic models.

**`ChunkResponse`** — converts `RetrievedChunk` (dataclass) for the `/search` response. Content truncated to 500 chars. `rerank_score` rounded to 4 decimals.

**`SectionResponse`** — converts `AgentSection` (already Pydantic, but we pick only the fields we want to expose).

**`ResearchResponse`** — converts `AgentOutput` (dataclass). Contains the full markdown report, verdict, evidence summary, and all sections. Includes `elapsed_seconds` for latency visibility.

Each response model has a `from_*()` classmethod that does the conversion. This keeps the conversion logic co-located with the response schema.

### Step 3 — FastAPI app with lifespan

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    yield                    # app runs during yield
    await close_pool()       # teardown: release DB connections

app = FastAPI(title="Pharma RAG API", lifespan=lifespan)
```

Same lifespan pattern as `mcp_server.py`. The asyncpg connection pool is released on shutdown.

CORS middleware added with `allow_origins=["*"]` for development. Restrict to specific domains in production.

### Step 4 — POST /research endpoint

```python
@app.post("/research", response_model=ResearchResponse)
async def research(request: ResearchRequest):
    output = await asyncio.wait_for(run_research(request.query), timeout=120)
    return ResearchResponse.from_agent_output(output, elapsed)
```

Calls `agents.run_research()` — the full LangGraph pipeline. Wrapped in `asyncio.wait_for(timeout=120)` — returns HTTP 504 on timeout instead of hanging. `elapsed_seconds` included in response for latency visibility.

### Step 5 — POST /search endpoint

```python
@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    result = await retrieve(query=request.query, filters=filters, use_rewriter=(filters is None))
    return SearchResponse(chunks=[ChunkResponse.from_retrieved_chunk(c) for c in result["chunks"]], ...)
```

Calls `retriever.retrieve()` directly — no grading, no agents, no synthesis. If explicit filters provided, builds `MetadataFilter` from them and skips the rewriter. Returns raw chunks with all metadata and scores.

### Step 6 — GET /drugs endpoint

Reads `DB_VOCABULARY` from `query_rewriter.py`. Returns all drugs with aliases, cancer types, endpoints, and study types. Static data — no pipeline call, instant response.

### Step 7 — GET /health endpoint

Tests database connectivity by acquiring a pool connection and running `SELECT 1`. Checks API key presence (not validity — just whether they exist in env vars). Returns `"healthy"` or `"degraded"` with details.

---

## How to run

### Install dependencies

```bash
pip install fastapi uvicorn
```

### Start the server

```bash
# From project root
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Or directly
python api/main.py
```

### Test the endpoints

```bash
# Health check
curl http://localhost:8000/health

# List drugs
curl http://localhost:8000/drugs

# Search (fast, ~2-5s)
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "pembrolizumab overall survival NSCLC"}'

# Search with explicit filters
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "overall survival", "drug": "pembrolizumab", "cancer_type": "nsclc", "year_min": 2020}'

# Full research (slow, ~15-30s)
curl -X POST http://localhost:8000/research \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the overall survival benefit of pembrolizumab in NSCLC?"}'
```

### Interactive docs

FastAPI auto-generates interactive API documentation:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

---

## Key decisions

**Why FastAPI over Flask:** FastAPI is natively async — `run_research()` and `retrieve()` are async functions. Flask would require wrapping them in `asyncio.run()` or using a compatibility layer. FastAPI also auto-generates OpenAPI docs from the Pydantic models.

**Why Pydantic response models instead of returning dicts:** Type safety, auto-documentation, and validation. FastAPI uses the `response_model` to generate the OpenAPI schema, validate the response, and produce interactive docs. Returning a raw dict would lose all of this.

**Why `from_*()` classmethods for conversion:** Keeps dataclass -> Pydantic conversion co-located with the response schema. When you add a field to `ChunkResponse`, the conversion logic is right there — not in a separate converter module.

**Why `asyncio.wait_for(timeout=120)` on /research:** Without a timeout, a complex query with multiple retrying agents could run indefinitely. HTTP clients (browsers, curl) have their own timeouts and would disconnect, but the server-side computation would continue wasting resources. The 120-second timeout returns a clean HTTP 504 error instead.

**Why CORS with `allow_origins=["*"]`:** Development convenience — any frontend can call the API. In production, restrict to the specific frontend domain.

**Why the health endpoint tests `SELECT 1`:** Verifying the asyncpg pool can acquire a connection and execute a trivial query proves end-to-end database connectivity. Checking API key presence (not calling the API) is a fast signal that env vars are configured correctly.
