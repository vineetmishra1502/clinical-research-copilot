# ingest.py — Ingestion Pipeline

**File:** `pipeline/ingest.py`  
**Run:** After `setup_db.py`. Takes 30–45 minutes for full ingestion.  
**Depends on:** Postgres running, `.env` configured, `setup_db.py` already executed.

---

## What It Does

Fetches 10,000 real clinical abstracts from PubMed across 5 drugs, extracts 20 metadata fields per article, splits abstracts into searchable chunks, generates semantic embeddings, and stores everything in Postgres.

**Input:** 5 drug name strings  
**Output:** ~12,000–15,000 chunks in the `documents` table, each with embedding + metadata  
**Cost:** ~$2–4 total (OpenAI embeddings + LLM fallback extraction)

---

## Flow

```
For each of 5 drugs:
│
├── FETCH
│     ├── esearch → get 2000 PMIDs (lightweight, IDs only)
│     └── efetch  → get full XML content in batches of 200
│
├── PARSE
│     ├── XML fields   → year, journal, authors, doi (direct extraction)
│     ├── Regex fields → sample_size, pdl1_status, funding, clinical_trial_id
│     └── Hybrid fields → cancer_type, study_type, endpoints, drug_combination
│                              │
│                              ├── regex pass (free, instant)
│                              └── ONE combined LLM call if regex fails
│
├── CHUNK  (structure-aware, 3 cases)
│     ├── Case 1: ≤600 tokens      → single chunk     (~60% of abstracts)
│     ├── Case 2: section headers  → split on headers (~30% of abstracts)
│     └── Case 3: long, no headers → recursive split  (~10% of abstracts)
│
├── EMBED
│     └── text-embedding-3-large, batches of 100 → 3072-dim vectors
│
└── STORE
      └── bulk INSERT into Postgres via executemany()
            → trigger auto-fills ts_content on every row
```

---

## Drug Selection Rationale

5 specific drugs, not a broad search. Depth beats breadth for RAG quality.

| Drug | Class | Company | Why included |
|---|---|---|---|
| pembrolizumab | PD-1 inhibitor | Merck | Largest checkpoint inhibitor literature |
| nivolumab | PD-1 inhibitor | BMS | Direct comparator to pembrolizumab |
| osimertinib | EGFR inhibitor | AstraZeneca | Flagship targeted therapy |
| trastuzumab | HER2 inhibitor | Roche | 25yr blockbuster, rich literature |
| metformin | Biguanide | — | Adds diabetes angle, cross-therapeutic |

To add a 6th drug: append one string to `DRUGS = [...]`. The pipeline adapts automatically.

---

## Metadata Extraction — Three Methods

### Method 1 — Direct XML (5 fields)

`year`, `journal`, `authors`, `doi` are structured fields in PubMed's XML. Extracted by navigating the nested record. No regex, no LLM, no ambiguity.

`clinical_trial_id` — checks `DataBankList` first (the official XML field where NCBI stores registered trial IDs), then falls back to regex `NCT\d{8}` in the abstract text. Two sources because not all trials are formally registered in PubMed's DataBankList even if the abstract text mentions the NCT number.

### Method 2 — Pure Regex (4 fields)

`sample_size`, `pdl1_status`, `funding`, and (fallback for) `clinical_trial_id` use regex only. No LLM fallback because:

- `sample_size` — numeric patterns are deterministic. 12 patterns cover every reporting style seen in real abstracts.
- `pdl1_status` — 5 fixed values (TPS<1%, TPS1-49%, TPS>=50%, PD-L1-positive, PD-L1-negative). Vocabulary is closed and stable.
- `funding` — 3 categories (industry, NIH, non-profit). Stable keyword sets.

### Method 3 — Hybrid Regex + LLM (4 fields)

`cancer_type`, `study_type`, `endpoints`, `drug_combination` use regex first then LLM fallback. These fields have open or semi-open vocabularies that regex alone cannot cover reliably.

---

## Hybrid Extraction — Design

### Why Not Regex Alone

A regex pattern list is a closed vocabulary. New cancer types get approved every year (cholangiocarcinoma, Merkel cell, uveal melanoma). New research designs emerge. Non-standard endpoint names exist. Regex silently defaults to `"oncology"` or `"observational"` — the data is ingested but the metadata filter is wrong.

### Why Not LLM Alone

10,000 abstracts × 4 fields × 1 call each = 40,000 API calls, ~3.2 seconds per abstract, ~$8 total. Unnecessary for fields where regex works reliably 85–90% of the time.

### The Hybrid Logic

```
For each of 4 hybrid fields:
  run regex → if matched: done (free)
            → if no match: add to missing_fields list

If missing_fields is not empty:
  Fire ONE combined LLM call for ALL missing fields simultaneously
  (never N separate calls — always 1 maximum per abstract)

Validate LLM output:
  cancer_type:      sanitize format only — open vocabulary, any disease accepted
  study_type:       must match VALID_STUDY_TYPES — reject unknowns → "observational"
  endpoints:        must match VALID_ENDPOINTS — drop unknowns silently
  drug_combination: sanitize format only — open vocabulary, any drug accepted

Apply safe defaults — pipeline never crashes regardless of LLM output
```

**Result:** Regex handles ~87% of abstracts for free. LLM fires for ~13%. Total LLM cost for all 10,000 abstracts: ~$0.30.

### Why One Combined LLM Call

The LLM reads the abstract once and returns all missing fields in a single JSON response. One HTTP request instead of up to 4. The prompt is built dynamically — only instructions for missing fields are included. A focused prompt with fewer instructions produces better accuracy.

`response_format={"type": "json_object"}` — OpenAI JSON mode guarantees parseable output. Without it, the model might include explanation text or markdown backticks, causing `json.loads()` to fail.

### Controlled vs Open Vocabulary Fields

| Field | Vocabulary | LLM constrained? | Validation |
|---|---|---|---|
| `cancer_type` | Open | No | Format only |
| `study_type` | Closed (14 values) | Yes | Reject → default |
| `endpoints` | Closed (10 values) | Yes | Drop unknowns |
| `drug_combination` | Open | No | Format only |

For `study_type`, if the LLM returns `"randomized-trial"` (not in `VALID_STUDY_TYPES`), it defaults to `"observational"`. This is intentional — a known conservative default is better than an unknown arbitrary value.

---

## Chunking Strategy

### Why Not Semantic Chunking

Would require ~150,000 extra embedding API calls just for chunking (10,000 abstracts × ~15 sentences each). Quality gain is marginal — PubMed abstracts are already semantically organized by section. Semantic chunking is the right choice for full-text papers; not for 250-token abstracts.

### Why Not Fixed-Size Chunking

Ignores content structure. A 500-token cut that lands mid-sentence loses context. A cut between Methods and Results separates closely related clinical findings.

### The Three Cases

**Case 1 — ≤600 tokens → single chunk**

Most PubMed abstracts are 200–400 tokens. Splitting them destroys context. The entire abstract is one coherent thought. Case reports always land here — narrative structure, no section labels.

**Case 2 — Section headers found → split on boundaries**

Structured abstracts use labeled sections (Background, Methods, Results, Conclusions). These are natural semantic boundaries. The section label is preserved in each chunk (`"Methods:\ncontent..."`) so the chunk retains context even in isolation.

Section pattern handles:
- Title Case AND UPPERCASE (re.IGNORECASE)
- 40+ label variants from real NLM data
- Colon required — prevents mid-sentence word matches
- `is_url_artifact()` — filters broken URLs that create false boundaries

**Case 3 — Long, no headers → RecursiveCharacterTextSplitter**

Reviews and narrative summaries without section labels. Splits on paragraphs first, then lines, then sentences, then words — always at the most meaningful boundary. `chunk_overlap=50` tokens prevents information loss at boundaries.

---

## Embedding

`text-embedding-3-large` — 3072 dimensions. Chosen over `text-embedding-3-small` (1536 dims) because it consistently ranks higher on the MTEB retrieval benchmark, particularly on scientific text. Cost difference on 15,000 chunks: ~$1.50.

Stored as `HALFVEC(3072)` — half-precision reduces memory per vector by 50% while preserving cosine similarity rankings. Required because pgvector's HNSW index has a 2,000-dimension limit on the full-precision `VECTOR` type.

Batches of 100 chunks per API call. `sleep(0.1)` between batches = 600 req/min, safely under OpenAI's 3,000 RPM tier-1 limit.

**Zero-vector fallback:** Failed batches store zero vectors. Zero vectors have near-zero cosine similarity with any real query — they are never retrieved. The pipeline continues rather than crashing.

---

## Storage

`executemany()` sends all rows in one database round-trip. For 2,000 chunks per drug, this is thousands of times faster than calling `execute()` in a loop.

`ON CONFLICT DO NOTHING` makes ingestion idempotent — re-running the pipeline does not create duplicates.

The `ts_content` column is NOT included in the INSERT statement. The trigger in `setup_db.py` fills it automatically from `content` on every row.

`sample_size` is stored as `NULL` when not found — not as `0`. This allows: `WHERE sample_size IS NOT NULL` to filter for studies that report patient counts.

---

## Key Caveats

**ENTREZ_EMAIL must be set**  
NCBI requires an email in `Entrez.email`. Without it, requests are rejected. No registration needed — just an email address. Set in `.env` as `ENTREZ_EMAIL=your@email.com`.

**NCBI rate limit is 3 req/sec without API key**  
The `sleep(0.4)` between batches enforces this. Removing it triggers HTTP 429 errors mid-ingestion. If you have an NCBI API key, you can increase to 10 req/sec and reduce sleep to `0.11`.

**Structured abstracts normalize label case**  
PubMed XML stores labels as `"BACKGROUND"` (uppercase). Journals display them as `"Background"` (Title Case). The parser calls `.title()` on every label to normalize to Title Case before chunking, so `SECTION_PATTERN` works consistently regardless of source format.

**evidence_level is derived from study_type**  
The mapping is hardcoded in `EVIDENCE_LEVEL_MAP`. If the LLM returns a `study_type` not in the map, `evidence_level` defaults to `2`. Correcting `study_type` via SQL UPDATE requires manually updating `evidence_level` in the same statement.

**Correcting mislabeled metadata does not require re-embedding**  
Vectors represent semantic content — they never change. Only metadata columns need SQL UPDATE. Example: correcting `cancer_type` for bile duct papers:
```sql
UPDATE documents
SET cancer_type = 'cholangiocarcinoma'
WHERE content ILIKE '%bile duct%'
  AND cancer_type = 'oncology';
```

**drug_combination regex is directional**  
The combo patterns (`"combined with X"`, `"plus X"`, `"+X"`) detect the partner drug relative to the searched drug. `"nivolumab plus ipilimumab"` correctly extracts `ipilimumab` as the partner. But if the sentence reads `"ipilimumab combined with nivolumab"` in a paper retrieved under the `nivolumab` search, the extracted partner would be `nivolumab` — a self-reference. The LLM fallback handles this better for ambiguous phrasing.

---

## Monitoring During Run

The pipeline prints per-drug stats at each stage:

```
================================================================
  Processing: PEMBROLIZUMAB
================================================================
  Searching PubMed: 'pembrolizumab[Title/Abstract]'
  Total on PubMed: 24,891 | Fetching: 2000
  Retrieved 1923 abstracts with content

  Pure regex fields:
    DOI         : 1756/1923
    NCT ID      : 892/1923
    Sample size : 1401/1923

  Hybrid fields:
    Regex only  : 1651/1923 (86%) — $0.00
    LLM used    : 272/1923  (14%) — ~$0.054

  Chunking cases:
    Case 1 (short)    : 1184
    Case 2 (sections) : 561
    Case 3 (recursive): 178

  2314 chunks | 1.2 avg/abstract
```

Watch the hybrid breakdown (86%/14%). If LLM usage climbs above 25%, the regex patterns need expanding.

---

## How to Run

```bash
# Activate virtual environment
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Mac/Linux

# Run full ingestion
python pipeline/ingest.py

# Verify after completion
docker exec -it pgvector-dev psql -U vineet -d pharma_rag \
  -c "SELECT drug, COUNT(*), COUNT(DISTINCT source) FROM documents GROUP BY drug;"
```

**What comes before:** `setup_db.py` — creates the schema this pipeline writes to.  
**What comes next:** `retriever.py` — hybrid dense + BM25 search over the data just ingested.
