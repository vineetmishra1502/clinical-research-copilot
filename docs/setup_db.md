# setup_db.py — Database Schema

**File:** `pipeline/setup_db.py`  
**Run:** Once before ingestion. Safe to re-run — `IF NOT EXISTS` guards on all objects.  
**To reset:** `DROP TABLE IF EXISTS documents CASCADE;` then re-run.

---

## What It Does

Creates the entire Postgres schema that the ingestion pipeline writes to and the retrieval layer reads from. No application code touches schema after this runs — only data flows in and out.

---

## Flow

```
Connect to Postgres
      │
      ├── Enable extensions
      │     ├── vector    → adds HALFVEC column type + HNSW index support
      │     └── pg_trgm   → trigram similarity for fuzzy text matching
      │
      ├── Create documents table
      │     └── 20 columns across 6 groups (see schema below)
      │
      ├── Create indexes
      │     ├── HNSW index     → fast approximate vector search
      │     ├── GIN index      → full-text keyword search (BM25)
      │     └── B-Tree indexes → fast metadata filtering
      │
      └── Create ts_content trigger
            → auto-fills keyword search column on every INSERT
```

---

## Schema — 6 Column Groups

```
GROUP 1 — Core content
  content        TEXT NOT NULL     the actual text chunk
  embedding      HALFVEC(3072)     semantic vector (3072 dims, half precision)
  ts_content     TSVECTOR          auto-filled by trigger — powers BM25 search

GROUP 2 — Publication identity
  source         TEXT              PubMed ID (PMID)
  doi            TEXT              direct link to paper
  clinical_trial_id TEXT           NCT number — deduplication across publications

GROUP 3 — Chunk position
  chunk_index    INT               position of this chunk within its article
  total_chunks   INT               total chunks from that article

GROUP 4 — Drug and disease context
  drug           TEXT              primary drug searched
  drug_combination TEXT            partner drug(s) if combination therapy
  cancer_type    TEXT              specific disease (nsclc, breast-cancer...)
  indication     TEXT              mirrors cancer_type, kept for compatibility

GROUP 5 — Study characteristics
  study_type     TEXT              RCT, meta-analysis, real-world-study...
  sample_size    INT               patient count (NULL if not found)
  endpoints      TEXT              OS,PFS,ORR — comma-separated
  pdl1_status    TEXT              TPS<1%, TPS>=50%, PD-L1-positive...
  funding        TEXT              industry, NIH, non-profit, unknown
  evidence_level INT               1=RCT/meta (strongest), 2=cohort, 3=case

GROUP 6 — Publication info
  year           INT               publication year
  journal        TEXT              venue name
  authors        TEXT              first 3 authors comma-separated
```

---

## Index Decisions

### HNSW — vector search

```sql
CREATE INDEX USING hnsw (embedding halfvec_cosine_ops)
WITH (m = 16, ef_construction = 64);
```

HNSW (Hierarchical Navigable Small World) builds a multi-layer graph over embeddings. Searches navigate the graph rather than scanning every row — O(log n) instead of O(n).

`halfvec_cosine_ops` — the operator class must match the column type. Using `vector_cosine_ops` on a `HALFVEC` column fails silently with wrong results. This was a real bug encountered during setup.

`m = 16` — connections per node. Higher = better recall, more memory.  
`ef_construction = 64` — search width during index build. Higher = more accurate graph, slower to build. Both are standard production defaults for corpora under 100K rows.

### GIN — full-text (BM25) search

```sql
CREATE INDEX USING GIN (ts_content);
```

GIN (Generalized Inverted Index) works like a book index — for every word, it stores which rows contain it. A keyword search looks up the word and gets back matching row IDs instantly. This is the BM25 side of hybrid retrieval.

### B-Tree — metadata filtering

```sql
CREATE INDEX ON documents (drug);
CREATE INDEX ON documents (year);
CREATE INDEX ON documents (study_type);
CREATE INDEX ON documents (evidence_level);
```

Without these, `WHERE drug = 'pembrolizumab' AND year >= 2020` requires a full table scan. With B-Tree indexes, the same query resolves in milliseconds. These four columns are the most frequently filtered in retrieval queries.

---

## The ts_content Trigger

```sql
CREATE TRIGGER ts_content_trigger
BEFORE INSERT OR UPDATE ON documents
FOR EACH ROW EXECUTE FUNCTION update_ts_content();
```

`ts_content` is never written by application code. This trigger fires automatically before every INSERT and UPDATE, calling `to_tsvector('english', content)` which:

- Lowercases all words
- Removes English stop words (the, a, is, was...)
- Stems words (treated → treat, showing → show)
- Stores as: `'pembrolizumab':1 'treat':2 'nsclc':3`

**Why this matters for hybrid retrieval:** The retrieval layer can do BM25 keyword search without any extra application logic. The searchable representation is always in sync with the content — updating a chunk's content automatically updates its keyword index.

---

## Key Caveats

**HALFVEC, not VECTOR**  
`text-embedding-3-large` produces 3,072-dimensional vectors. pgvector's HNSW and IVFFlat indexes both have a hard 2,000-dimension limit on the `VECTOR` type. `HALFVEC` stores dimensions in 16-bit half precision, halving memory usage and bypassing the limit. The precision loss has negligible impact on cosine similarity rankings.

**halfvec_cosine_ops, not vector_cosine_ops**  
The HNSW index operator class must match the column type. `HALFVEC` columns require `halfvec_cosine_ops`. Using the wrong operator class causes a silent type mismatch — the index builds but retrieval results are incorrect.

**CASCADE on DROP**  
`DROP TABLE IF EXISTS documents CASCADE` is required when resetting because the indexes and trigger depend on the table. Without `CASCADE`, the drop fails if dependent objects exist.

**evidence_level is derived, not stored independently**  
`evidence_level` is computed from `study_type` during ingestion. If `study_type` is corrected via an UPDATE query, `evidence_level` should be updated in the same statement — it does not auto-recompute.

---

## How to Run

```bash
# Ensure pgvector container is running
docker ps | grep pgvector-dev

# Run setup
python pipeline/setup_db.py

# Verify columns
docker exec -it pgvector-dev psql -U vineet -d pharma_rag -c "\d documents"
```

**What comes next:** `ingest.py` — fetches PubMed abstracts and populates this schema.
