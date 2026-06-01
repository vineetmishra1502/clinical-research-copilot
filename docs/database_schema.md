# Database Schema — `documents` Table

PostgreSQL table for storing pharma/oncology literature chunks with semantic embeddings, full-text search, and rich metadata for filtered retrieval.

---

## Extensions

| Extension | Purpose |
|-----------|---------|
| `vector` | Adds `HALFVEC` column type and HNSW / IVFFlat index support for semantic search |
| `pg_trgm` | Trigram similarity for fuzzy text matching; boosts full-text search on partial words |

---

## Column Reference

### Identity

| Column | Type | Description |
|--------|------|-------------|
| `id` | `SERIAL PRIMARY KEY` | Auto-incrementing row identifier |

---

### Core Content

| Column | Type | Description |
|--------|------|-------------|
| `content` | `TEXT NOT NULL` | The actual text chunk — typically a concatenation of the article title and an abstract section |
| `embedding` | `HALFVEC(3072)` | 3072-dimensional semantic vector produced by the embedding model. `HALFVEC` uses half-precision floats, halving memory vs `VECTOR` and bypassing the 2000-dimension HNSW index limit |
| `ts_content` | `TSVECTOR` | Pre-tokenised full-text search representation of `content`. **Never set manually** — auto-filled by a trigger on every `INSERT`/`UPDATE` using `to_tsvector('english', content)`, which lowercases, removes English stop words, and stems terms |

---

### Publication Identity

| Column | Type | Description |
|--------|------|-------------|
| `source` | `TEXT` | PubMed ID (PMID) — the unique article identifier assigned by NCBI |
| `doi` | `TEXT` | Digital Object Identifier — direct link to the full paper (e.g. `10.1056/NEJMoa2034892`) |
| `clinical_trial_id` | `TEXT` | NCT number or named trial (e.g. `KEYNOTE-189`). Extracted from `DataBankList` and abstract text. Critical for deduplication — prevents double-counting the same trial reported across multiple publications |

---

### Chunk Position

| Column | Type | Description |
|--------|------|-------------|
| `chunk_index` | `INT` | Zero-based position of this chunk within its source article |
| `total_chunks` | `INT` | Total number of chunks produced from that article. Together with `chunk_index`, allows reconstruction of original article order |

---

### Drug Context

| Column | Type | Description |
|--------|------|-------------|
| `drug` | `TEXT` | Primary drug that was searched (e.g. `pembrolizumab`, `nivolumab`) |
| `drug_combination` | `TEXT` | Full combination regimen if applicable (e.g. `pembrolizumab + carboplatin + pemetrexed`). Enables filtering monotherapy vs. combination therapy queries |

---

### Disease Context

| Column | Type | Description |
|--------|------|-------------|
| `cancer_type` | `TEXT` | Specific disease entity (e.g. `NSCLC`, `breast cancer`). Extracted from abstract text via regex. Replaces a generic `oncology` label and enables precise disease-level filtering |
| `indication` | `TEXT` | Kept for backward compatibility and non-oncology cases (e.g. diabetes); broader than `cancer_type` |

---

### Study Characteristics

| Column | Type | Description |
|--------|------|-------------|
| `study_type` | `TEXT` | Study design: `RCT`, `meta-analysis`, `real-world`, `case-report`, etc. Inferred from title and abstract keyword matching |
| `sample_size` | `INT` | Patient count extracted via regex (e.g. `1966`). Larger samples indicate stronger evidence — `n=1966` vs `n=12` |
| `endpoints` | `TEXT` | Comma-separated outcome measures (e.g. `OS, PFS, ORR`). Enables filtering efficacy-focused vs. safety-focused queries |
| `pdl1_status` | `TEXT` | PD-L1 biomarker expression level. Critical filter for checkpoint inhibitor studies where response rates differ strongly by expression tier |
| `funding` | `TEXT` | Funding source: `industry`, `NIH`, `non-profit`, etc. Used as an evidence quality signal |

---

### Evidence Quality

| Column | Type | Description |
|--------|------|-------------|
| `evidence_level` | `INT` | Strength of evidence tier used by the grader node to weight retrieved chunks: `1` = RCT / meta-analysis (strongest), `2` = cohort / real-world study, `3` = case report / case series (weakest) |

---

### Publication Info

| Column | Type | Description |
|--------|------|-------------|
| `year` | `INT` | Publication year |
| `journal` | `TEXT` | Journal name |
| `authors` | `TEXT` | First three authors, comma-separated (e.g. `Smith J, Jones A, Lee B`) |

---

## Indexes

| Index | Type | Columns | Purpose |
|-------|------|---------|---------|
| `documents_embedding_idx` | HNSW | `embedding` (`halfvec_cosine_ops`) | Approximate nearest-neighbour semantic search. `m=16` neighbours per node, `ef_construction=64` search width at build time |
| `documents_fts_idx` | GIN | `ts_content` | Full-text / BM25 keyword search. GIN builds an inverted index (word → row list) over the pre-tokenised `ts_content` column |
| `documents_drug_idx` | B-Tree | `drug` | Fast equality filter on drug name |
| `documents_year_idx` | B-Tree | `year` | Fast range filter on publication year |
| `documents_study_type_idx` | B-Tree | `study_type` | Fast equality filter on study design |
| `documents_evidence_level_idx` | B-Tree | `evidence_level` | Fast filter on evidence tier for grading |

---

## Trigger: `ts_content_trigger`

Fires **BEFORE INSERT OR UPDATE** on every row. Calls `update_ts_content()`, which runs:

```sql
NEW.ts_content := to_tsvector('english', NEW.content);
```

This means you never need to populate `ts_content` manually — insert or update `content` and the tsvector is computed automatically.
