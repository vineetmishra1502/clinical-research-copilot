# Clinical Research Copilot

**LangGraph-powered multi-agent research platform for oncology evidence synthesis.**

![](./media/Clinical_Research_Copilot.mp4)

A production-grade multi-agent Retrieval-Augmented Generation system that searches, grades, and synthesizes clinical evidence from 12,000+ PubMed abstracts across 5 oncology/pharma drugs. Built with LangGraph, pgvector, and hybrid retrieval (dense + BM25 + Cohere rerank).

Ask a clinical question → the supervisor generates domain-focused queries and dispatches specialist agents (efficacy, safety, mechanism, competitor) in parallel → each agent independently retrieves, grades, and writes cited findings → a synthesizer merges everything into a research brief with evidence quality assessment.

## Architecture

```
User query: "What is the OS benefit of pembrolizumab in NSCLC?"
                            |
                            v
          +-------------------------------------+
          |        LangGraph supervisor          |
          |                                      |
          |  Analyses query, selects agents,     |
          |  generates focused_query per agent:  |
          |  "pembro NSCLC" becomes              |
          |    efficacy: "pembrolizumab OS NSCLC" |
          |    safety:   "pembrolizumab AE NSCLC" |
          |                                      |
          |  Send() dispatches in parallel       |
          +----+-----+-----+-----+--------------+
               |     |     |     |
               v     v     v     v
          +--------+ +------+ +--------+ +----------+
          |Efficacy| |Safety| |Mechanm.| |Competitor|
          +---+----+ +--+---+ +---+----+ +----+-----+
              |         |         |            |
              +---------+---------+------------+
                   Each agent runs independently:
                            |
                            v
                   +------------------+
                   | Query rewriter   |  Structured output (gpt-4o-mini)
                   | Alias normalization ("keytruda" -> "pembrolizumab")
                   | MetadataFilter extraction (drug, cancer, endpoints)
                   | Sub-question decomposition for comparisons
                   | RRF strategy selection (5 discrete weights)
                   | Three-layer protection: Literal types -> validators -> safe fallback
                   +--------+---------+
                            |
                   rewritten_query + MetadataFilter + sub_questions + rrf_weights
                            |
              +-------------+-------------+
              |     B-Tree metadata        |  WHERE drug=$1 AND cancer_type=$2
              |     pre-filter (3ms)       |  12,000 chunks -> ~400
              +-------------+-------------+
                            |
                            v
          +-------------------------------------+
          |         Hybrid retriever             |
          |                                      |
          |  +-------------+   +-------------+   |
          |  | Dense search |   | BM25 search |   |  asyncio.gather() — parallel
          |  | HNSW cosine  |   | GIN ts_rank |   |
          |  | (pgvector)   |   | (Postgres)  |   |
          |  +------+------+   +------+------+   |
          |         |                  |          |
          |         v                  v          |
          |     Weighted RRF fusion               |
          |     (LLM-selected strategy weights)   |
          |     Chunks in both lists = consensus  |
          |                |                      |
          |                v                      |
          |     Cohere cross-encoder rerank       |  asyncio.to_thread() — non-blocking
          |     40 candidates -> top 5            |
          |     Scores vs ORIGINAL query          |
          +----------------+----------------------+
                           |
                  5 RetrievedChunks
                           |
                           v
          +-------------------------------------+
          |       Self-correcting grader         |
          |                                      |
          |  Relevancy:    LLM (semantic)        |  5 parallel LLM calls
          |  Faithfulness: Formula (structural)  |  evidence_level + sample_size
          |                                      |
          |  Both thresholds must independently  |
          |  pass (AND, not average)             |
          |                                      |
          |  If < 3 accepted:                    |
          |    Diagnoses failure pattern:         |
          |    relevancy low -> LLM rewrites     |
          |    faithfulness low -> rules tighten  |
          |    both low -> acknowledge gap        |
          |                                      |
          |  Progressive fallback (5 levels)     |
          |  + compensating thresholds            |
          |  (0.40 -> 0.50 -> 0.55 -> 0.60 ->   |
          |   0.65 as filters relax)             |
          +----------------+---------------------+
                           |
                  GraderOutput (verdict + accepted_chunks)
                           |
                           v
                  Section writer (per agent)
                  Verdict propagates into prompt:
                  "accepted" -> full claims
                  "partial"  -> "limited evidence suggests..."
                           |
                           v
          +-------------------------------------+
          |           Synthesizer                |
          |                                      |
          |  Merges all agent sections           |
          |  Worst-case verdict across agents    |
          |  Cited markdown research brief       |
          |  Evidence quality + limitations      |
          +----------------+---------------------+
                           |
                           v
               +------------------------+
               |    FastAPI + Streamlit  |
               |  POST /research         |
               |  POST /search           |
               |  GET  /drugs            |
               |  GET  /health           |
               |  Streamlit UI with      |
               |  pipeline stepper       |
               +------------------------+
```

## Key features

**Metadata-enriched ingestion** — During ingestion, GPT-4o-mini extracts 8 structured fields from every abstract (drug, cancer type, study design, evidence level, sample size, endpoints, PD-L1 status, funding). Stored as typed Postgres columns with B-Tree indexes. At query time, metadata filtering is a 3ms index lookup — not an LLM call.

**Structured query rewriting with sub-question decomposition** — Normalizes drug aliases ("keytruda" → "pembrolizumab"), extracts metadata filters via structured output with Literal type constraints, selects RRF strategy from 5 discrete options, and decomposes comparison queries into sub-questions ("pembro vs nivo" → two independent retrieval passes merged via RRF).

**Three-layer LLM output protection** — Layer 1: Literal types constrain the LLM at generation time. Layer 2: Pydantic field_validators catch edge cases post-generation. Layer 3: try/except safe_fallback ensures the pipeline never crashes. Each layer catches different failure modes.

**Hybrid retrieval with metadata pre-filtering** — B-Tree WHERE clause reduces 12,000 chunks to ~400 before vector search starts. Dense (HNSW cosine via pgvector) + BM25 (GIN full-text via Postgres) run in parallel via `asyncio.gather()`, fused with weighted Reciprocal Rank Fusion. Cohere cross-encoder reranks against the original query (not rewritten — user intent matters more).

**Dual-dimension evidence grading** — Relevancy scored by LLM (semantic judgment — 5 parallel calls), faithfulness scored by formula (evidence hierarchy: RCT=1.0, cohort=0.65, case report=0.25 + sample_size boost). Both thresholds must independently pass (AND, not average). A case report about the perfect topic still fails faithfulness.

**Self-correcting retrieval loop** — When evidence quality is low, the grader diagnoses the failure pattern: relevancy low → LLM rewrites query with more specific terms; faithfulness low → deterministic rules tighten evidence filters; both low → acknowledge data limitation. Progressive fallback across 5 filter levels with compensating quality thresholds (0.40 → 0.65).

**Supervisor with focused query generation** — The LangGraph supervisor doesn't just route — it generates a domain-focused query per agent. "pembrolizumab NSCLC" becomes "pembrolizumab overall survival efficacy NSCLC" for the efficacy agent and "pembrolizumab adverse events toxicity" for the safety agent.

**Specialist agents with domain-specific filters** — LangGraph `Send()` API dispatches agents in parallel. Each agent has mutually exclusive evidence requirements: efficacy requires evidence_level ≤ 2 (no case reports), safety allows case reports (rare AEs only appear there), competitor removes the drug filter (cross-drug search). `operator.add` merges results concurrently. 2 agents take the same wall-clock time as 1.

**Verdict propagation** — Evidence quality flows end-to-end: grader verdict → injected into agent prompt ("limited evidence suggests...") → synthesizer uses worst-case verdict across agents → final report reflects limitations. The system never makes overconfident claims from weak evidence.

## Tech stack

| Component | Technology |
|---|---|
| Language | Python 3.13 |
| LLM | OpenAI gpt-4o-mini (agents), gpt-4o-mini (synthesis) |
| Embeddings | OpenAI text-embedding-3-large (3072 dimensions) |
| Reranker | Cohere rerank-english-v3.0 |
| Vector DB | PostgreSQL 16 + pgvector (HNSW + HALFVEC) |
| Full-text | PostgreSQL GIN index + ts_rank (BM25-equivalent) |
| Orchestration | LangGraph (StateGraph + Send API) |
| Structured output | LangChain with_structured_output + Pydantic |
| API | FastAPI + Pydantic response models |
| Frontend | Streamlit (pipeline stepper, evidence cards, PubMed links) |
| Data source | PubMed E-utilities API (12,000+ abstracts) |

## Performance

| Metric | Value | Notes |
|---|---|---|
| Search latency (warm) | 4.8s | With explicit filters, rewriter skipped |
| Search latency (cold) | 17.4s | First request — pool + model warmup |
| Research (1 agent) | 28.7s | Full pipeline: rewrite + retrieve + grade + synthesize |
| Research (2 agents) | 26.6s | Parallel agents — ~same as 1 agent, not 2x |
| Rerank accuracy | 0.999 | Cohere cross-encoder confidence on top results |
| Candidates per search | 40 | 20 dense + 20 BM25, deduplicated via RRF |
| Final chunks | 5 | After reranking |
| Evidence levels | L1 + L2 | RCTs and meta-analyses correctly prioritized |

## Quick start

### Prerequisites

- Python 3.13
- Docker Desktop (for PostgreSQL)
- API keys: OpenAI, Cohere

### 1. Clone and setup

```bash
git clone https://github.com/yourusername/clinical-research-copilot.git
cd clinical-research-copilot
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

### 2. Environment variables

Create `.env` in the project root:

```bash
# LLM APIs
OPENAI_API_KEY=sk-...
COHERE_API_KEY=...

# PostgreSQL (Docker container)
DB_HOST=localhost
DB_PORT=5432
DB_NAME=pharma_rag
DB_USER=vineet
DB_PASSWORD=devpassword

# LangSmith (optional — tracing)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls__...
LANGCHAIN_PROJECT=pharma-rag
```

### 3. Start PostgreSQL with pgvector

```bash
docker run -d --name pgvector-dev \
  -e POSTGRES_USER=vineet \
  -e POSTGRES_PASSWORD=devpassword \
  -e POSTGRES_DB=pharma_rag \
  -p 5432:5432 \
  pgvector/pgvector:pg16
```

### 4. Create schema and ingest data

```bash
python setup_db.py     # Creates tables, indexes (HNSW, GIN, B-Tree)
python ingest.py       # Fetches and embeds 12,000+ PubMed abstracts (~30 min)
```

### 5. Start the API server

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 6. Start the Streamlit UI

```bash
# In a second terminal
streamlit run streamlit_app.py
```

Open http://localhost:8501 for the full research interface with pipeline stepper, evidence cards, and PubMed-linked citations.

### 7. Test

```bash
python api_test.py     # Runs all 7 endpoint tests
```

Or open http://localhost:8000/docs for interactive Swagger UI.

## Streamlit UI

The Streamlit frontend provides two modes:

**Deep Research** — Type a clinical question, watch the 6-stage pipeline stepper animate (Supervisor → Agent Dispatch → Evidence Retrieval → Evidence Grading → Section Writing → Report Synthesis), then read the full cited research brief with evidence quality badges, agent sections, and PubMed-linked citations.

**Evidence Search** — Fast evidence retrieval with collapsible filters (drug, cancer type, endpoint, study design, publication year). Returns ranked evidence chunks as cards with metadata grids, evidence level badges, rerank scores, and expandable abstracts.

Features:
- Real-time 6-stage pipeline progress stepper
- Color-coded evidence level badges (Level 1 green, Level 2 blue, Level 3 amber)
- Metadata grid per chunk (drug, cancer type, study design, journal, year, sample size, endpoint, PMID)
- PubMed-linked citations throughout report and sections
- Verdict-colored evidence summary (green=accepted, amber=partial, red=insufficient)
- Sidebar with pipeline explanation, knowledge base stats, and API settings
- Tech stack display: LangGraph, FastAPI, OpenAI, Cohere Rerank, pgvector, PostgreSQL, Pydantic, Streamlit

## API endpoints

### POST /research — full multi-agent pipeline

```bash
curl -X POST http://localhost:8000/research \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the overall survival benefit of pembrolizumab in NSCLC?"}'
```

Response:

```json
{
  "query": "What is the overall survival benefit of pembrolizumab in NSCLC?",
  "overall_verdict": "accepted",
  "evidence_summary": "EFFICACY: verdict=accepted | chunks=5 | RCTs=2 | attempts=1",
  "elapsed_seconds": 28.72,
  "sections": [
    {
      "title": "Efficacy Evidence for Pembrolizumab in NSCLC",
      "evidence_quality": "Based on 4 RCTs and 2 real-world studies (n=300 to n=1,966)",
      "citations": [
        "[PMID: 42046364] Immunotherapy (2026) — meta-analysis",
        "[PMID: 37465924] Immunotherapy (2023) — RCT",
        "[PMID: 34048946] Journal of Thoracic Oncology (2021) — RCT"
      ]
    }
  ],
  "report": "# Executive Summary\n\nPembrolizumab has shown significant improvements in overall survival (OS) for patients with non-small cell lung cancer (NSCLC), particularly in those with higher PD-L1 expression. The KEYNOTE-010 study and real-world data support its efficacy..."
}
```

### POST /search — fast evidence retrieval

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "pembrolizumab NSCLC", "drug": "pembrolizumab", "year_min": 2020, "endpoints": "OS"}'
```

Response includes 5 evidence chunks with metadata (PMID, journal, year, study type, sample size, evidence level) and Cohere rerank scores.

### GET /drugs — knowledge base contents

```bash
curl http://localhost:8000/drugs
```

```json
{
  "drugs": [
    {"name": "pembrolizumab", "aliases": ["keytruda", "pembro", "mk-3475"]},
    {"name": "nivolumab", "aliases": ["opdivo", "nivo"]},
    {"name": "osimertinib", "aliases": ["tagrisso", "osi", "azd9291"]},
    {"name": "trastuzumab", "aliases": ["herceptin"]},
    {"name": "metformin", "aliases": ["glucophage"]}
  ],
  "cancer_types": ["nsclc", "sclc", "breast-cancer", "melanoma", "..."],
  "endpoints": ["OS", "PFS", "ORR", "DFS", "EFS", "CR", "PR", "DoR", "TTP", "AE"],
  "study_types": ["RCT", "meta-analysis", "phase-3-trial", "real-world-study", "..."]
}
```

### GET /health — health check

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "healthy",
  "database": "connected",
  "api_keys": {"openai": true, "cohere": true, "langsmith": true}
}
```

## Example queries

| Query | Agents activated | Verdict | Time |
|---|---|---|---|
| "What is the OS benefit of pembrolizumab in NSCLC?" | efficacy | accepted | 28.7s |
| "Compare pembrolizumab vs nivolumab in NSCLC survival" | efficacy, competitor | partial | 26.6s |
| "Adverse events of osimertinib in EGFR-mutant NSCLC" | safety | accepted | ~25s |
| "How does pembrolizumab work against PD-L1?" | mechanism | accepted | ~22s |

## Project structure

```
agentic-rag-pharma/
├── .env                          # API keys and DB credentials
├── setup_db.py                   # PostgreSQL schema + indexes
├── ingest.py                     # PubMed data ingestion pipeline
├── query_rewriter.py             # LLM query normalization + filter extraction
├── retriever.py                  # Hybrid retrieval: dense + BM25 + RRF + rerank
├── grader.py                     # Self-correcting evidence grading loop
├── agents.py                     # LangGraph multi-agent orchestration
├── streamlit_app.py              # Streamlit frontend (pipeline stepper, evidence cards)
├── api/
│   └── main.py                   # FastAPI serving layer
├── api_test.py                   # API endpoint test suite (7 tests)
├── docs/
│   ├── setup_db.md               # Schema documentation
│   ├── ingest.md                 # Ingestion pipeline documentation
│   ├── INGESTION_PIPELINE.md     # Detailed ingestion flow
│   ├── query_rewriter.md         # Rewriter architecture
│   ├── retriever.md              # Retrieval pipeline documentation
│   ├── grader.md                 # Grading loop documentation
│   ├── agents.md                 # Multi-agent architecture
│   ├── api_main.md               # API documentation
│   └── PHARMA_KNOWLEDGE.md       # Clinical domain reference
├── requirements.txt
└── README.md
```

## Architectural decisions

**Metadata enrichment at ingestion, not query time.** GPT-4o-mini extracts 8 structured fields during ingestion — drug, cancer type, study design, evidence level, sample size, endpoints, PD-L1 status, funding. These become B-Tree-indexed columns. At query time, filtering is a 3ms index lookup instead of an LLM call per query. Extract once, query cheaply forever.

**Co-located vectors and metadata in one database.** pgvector inside PostgreSQL — not a separate vector DB. One query does metadata filtering (B-Tree WHERE) AND vector search (HNSW cosine) in the same transaction. No cross-database consistency problems.

**Pydantic for LLM output, dataclass for internal state.** `FiltersOutput`, `RewriterOutput`, `GradingResult`, `AgentSection` are Pydantic — they come from untrusted LLM generation and need validation. `MetadataFilter`, `RetrievedChunk`, `GraderOutput`, `AgentOutput` are dataclasses — they are built by trusted internal code from already-validated values. No redundant validation.

**Supervisor generates focused queries, not just routes.** The supervisor doesn't just select agents — it rewrites the query for each agent's domain. "pembrolizumab NSCLC" becomes "pembrolizumab overall survival efficacy NSCLC" for the efficacy agent and "pembrolizumab adverse events toxicity" for the safety agent. Each agent retrieves with a query optimized for its domain.

**Sub-question decomposition for comparisons.** "pembro vs nivo NSCLC" decomposes into two independent retrieval passes — one per drug. Results merged via RRF. Chunks relevant to both drugs rank highest through consensus scoring.

**LLM chooses RRF weights, not regex.** The old approach used hardcoded regex patterns with arbitrary weights. The LLM reads the full query and picks from 5 discrete strategies (`dense_only`, `dense_heavy`, `equal`, `bm25_heavy`, `bm25_only`). Discrete strategies are validatable and explainable — free floats imply false precision.

**Relevancy is LLM, faithfulness is formula.** "Does this chunk answer the query?" requires semantic understanding — only an LLM can judge that "OS improvement" and "survival benefit" mean the same thing. "Is this evidence from a reliable study design?" is an arithmetic check on `evidence_level` — a formula is more reliable, cheaper, and faster than an LLM reading the same integer.

**Both thresholds must independently pass.** A case report (faithfulness=0.25) about the exact right topic (relevancy=0.95) would average to 0.67 and pass. But citing a single case report as efficacy evidence is clinically wrong. Independent AND thresholds prevent this.

**Specialist agents exist for clinical reasons.** Efficacy requires `evidence_level_max=2` (no case reports). Safety requires no evidence level filter (case reports are valid for rare adverse events). Competitor removes the drug filter entirely (cross-drug comparison). These requirements are mutually exclusive in a single filter — one agent cannot satisfy all three.

**Failure diagnosis before retry.** When evidence quality is low, the grader classifies the failure pattern: relevancy low → LLM rewrites query; faithfulness low → deterministic rules tighten filters; both low → acknowledge limitation. Different problems need different fixes — blind retry wastes tokens and returns the same bad results.

**Progressive fallback with compensating quality gates.** 5 filter levels (full → relaxed → minimal → drug_only → no_filter). As filters loosen, faithfulness thresholds tighten (0.40 → 0.65). Wider net, tighter mesh — the system never returns low-quality evidence just because strict filters had no results.

## Knowledge base

| Metric | Value |
|---|---|
| Total chunks | 12,000+ |
| Unique papers (PMIDs) | 9,249 |
| Drugs indexed | 5 |
| Cancer types | 39+ |
| Study designs | 15 |
| Clinical endpoints | 10 (OS, PFS, ORR, DFS, EFS, CR, PR, DoR, TTP, AE) |
| Evidence levels | 3 (RCT/meta-analysis, cohort/real-world, case report) |
| Embedding dimensions | 3,072 (text-embedding-3-large, HALFVEC) |
| Indexes | HNSW (dense), GIN (BM25), B-Tree (metadata) |

## Drugs and clinical context

| Drug | Class | Primary cancers | Key trials |
|---|---|---|---|
| Pembrolizumab | Anti-PD-1 checkpoint inhibitor | NSCLC, melanoma | KEYNOTE-024, KEYNOTE-189, KEYNOTE-010 |
| Nivolumab | Anti-PD-1 checkpoint inhibitor | NSCLC, melanoma, RCC | CheckMate-017, CheckMate-057 |
| Osimertinib | EGFR TKI (3rd generation) | EGFR-mutant NSCLC | FLAURA, AURA3 |
| Trastuzumab | Anti-HER2 monoclonal antibody | HER2+ breast, gastric | HERA, CLEOPATRA |
| Metformin | Biguanide (anti-diabetic) | Type 2 diabetes, cancer research | ADOPT, repurposing studies |

## License

MIT
