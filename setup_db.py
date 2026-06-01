"""
setup_db.py — Database Schema Setup
=====================================
Run this ONCE before ingestion.
Creates the documents table with full metadata schema,
HNSW vector index, GIN full-text index, and auto-fill trigger.

Run again safely — IF NOT EXISTS guards prevent duplicate creation.
To reset: DROP TABLE IF EXISTS documents; then re-run.
"""

import os
import sys
import psycopg2
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()


def get_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD")
    )


def setup():
    conn = get_connection()
    conn.autocommit = True
    cur  = conn.cursor()

    # ── EXTENSIONS ────────────────────────────────────────────────────
    print("Enabling extensions...")

    # vector: adds HALFVEC column type and HNSW/IVFFlat index support
    # Without this, HALFVEC(3072) does not exist as a data type
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    # pg_trgm: trigram similarity for fuzzy text matching
    # Boosts full-text search performance on partial word matches
    cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")

    print("  vector    ✓")
    print("  pg_trgm   ✓")

    # ── DOCUMENTS TABLE ───────────────────────────────────────────────
    print("\nCreating documents table...")

    cur.execute("DROP TABLE IF EXISTS documents CASCADE;")

    cur.execute("""
        CREATE TABLE IF NOT EXISTS documents (

            -- ── Identity ─────────────────────────────────────────────
            id                SERIAL PRIMARY KEY,

            -- ── Core content ──────────────────────────────────────────
            -- content: the actual text chunk (title + abstract section)
            -- embedding: 3072-dim semantic vector — HALFVEC halves memory
            --            vs VECTOR and has no 2000-dim HNSW index limit
            -- ts_content: auto-filled by trigger — powers BM25 keyword search
            content           TEXT        NOT NULL,
            embedding         HALFVEC(3072),
            ts_content        TSVECTOR,

            -- ── Publication identity ──────────────────────────────────
            -- source: PubMed ID (PMID) — unique article identifier
            -- doi: direct link to full paper e.g. 10.1056/NEJMoa2034892
            -- clinical_trial_id: NCT number or trial name (KEYNOTE-189)
            --   extracted from DataBankList and abstract text
            --   critical for deduplication — prevents double-counting
            --   same trial reported across multiple publications
            source            TEXT,
            doi               TEXT,
            clinical_trial_id TEXT,

            -- ── Chunk position ────────────────────────────────────────
            -- chunk_index: position of this chunk within its source article
            -- total_chunks: total chunks from that article
            -- Together they let you reconstruct original article order
            chunk_index       INT,
            total_chunks      INT,

            -- ── Drug context ──────────────────────────────────────────
            -- drug: primary drug searched (pembrolizumab, nivolumab...)
            -- drug_combination: what it was combined with if applicable
            --   e.g. "pembrolizumab + carboplatin + pemetrexed"
            --   enables filtering monotherapy vs combo therapy queries
            drug              TEXT,
            drug_combination  TEXT,

            -- ── Disease context ───────────────────────────────────────
            -- cancer_type: specific disease e.g. NSCLC, breast cancer
            --   extracted from abstract text via regex
            --   replaces generic "oncology" — enables precise filtering
            -- indication: kept for backward compatibility and diabetes cases
            cancer_type       TEXT,
            indication        TEXT,

            -- ── Study characteristics ─────────────────────────────────
            -- study_type: RCT, meta-analysis, real-world, case-report...
            --   inferred from title + abstract keyword matching
            -- sample_size: patient count extracted via regex
            --   n=1966 is much stronger evidence than n=12
            -- endpoints: comma-separated outcome measures
            --   OS, PFS, ORR — filters efficacy vs safety queries
            -- pdl1_status: PD-L1 biomarker expression level
            --   critical filter for checkpoint inhibitor studies
            -- funding: industry vs NIH vs non-profit
            --   evidence quality signal
            study_type        TEXT,
            sample_size       INT,
            endpoints         TEXT,
            pdl1_status       TEXT,
            funding           TEXT,

            -- ── Evidence quality ─────────────────────────────────────
            -- evidence_level: 1=RCT/meta-analysis (strongest)
            --                 2=cohort/real-world study
            --                 3=case report/case series (weakest)
            -- Used directly by the grader node to weight retrieved chunks
            -- "This chunk is level 1 evidence, n=1966" vs
            -- "This chunk is level 3 evidence, n=1"
            evidence_level    INT,

            -- ── Publication info ──────────────────────────────────────
            year              INT,
            journal           TEXT,
            -- First 3 authors comma-separated: "Smith J, Jones A, Lee B"
            authors           TEXT
        );
    """)
    print("  documents table ✓")

    # ── HNSW VECTOR INDEX ─────────────────────────────────────────────
    # HNSW = Hierarchical Navigable Small World
    # Builds a multi-layer graph over embeddings for fast ANN search
    # halfvec_cosine_ops: must match the HALFVEC column type
    #   (vector_cosine_ops would fail — wrong operator class)
    # m=16: each node connects to 16 neighbours
    #   higher m = better recall, more memory usage
    # ef_construction=64: search width during index build
    #   higher = more accurate graph, slower to build
    # No 2000-dim limit because HALFVEC bypasses it
    print("\nCreating indexes...")

    cur.execute("""
        CREATE INDEX IF NOT EXISTS documents_embedding_idx
        ON documents
        USING hnsw (embedding halfvec_cosine_ops)
        WITH (m = 16, ef_construction = 64);
    """)
    print("  HNSW vector index     ✓  (halfvec_cosine_ops, m=16, ef=64)")

    # ── GIN FULL-TEXT SEARCH INDEX ────────────────────────────────────
    # GIN = Generalised Inverted Index
    # Works like a book's index: word -> list of rows containing it
    # Powers the BM25 keyword search side of hybrid retrieval
    # ts_content is TSVECTOR (pre-tokenised) — GIN indexes it natively
    cur.execute("""
        CREATE INDEX IF NOT EXISTS documents_fts_idx
        ON documents
        USING GIN (ts_content);
    """)
    print("  GIN full-text index   ✓  (ts_content tsvector)")

    # ── METADATA FILTER INDEXES ───────────────────────────────────────
    # B-Tree indexes on frequently filtered columns
    # Enables fast: WHERE drug = 'pembrolizumab' AND year >= 2020
    # Without these, metadata filtering requires full table scan
    cur.execute("""
        CREATE INDEX IF NOT EXISTS documents_drug_idx
        ON documents (drug);
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS documents_year_idx
        ON documents (year);
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS documents_study_type_idx
        ON documents (study_type);
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS documents_evidence_level_idx
        ON documents (evidence_level);
    """)
    print("  B-Tree metadata indexes ✓  (drug, year, study_type, evidence_level)")

    # ── AUTO-FILL TRIGGER ─────────────────────────────────────────────
    # This trigger fires BEFORE every INSERT or UPDATE on documents
    # It automatically converts content -> tsvector and stores in ts_content
    # So you NEVER manually fill ts_content — just insert content text
    #
    # to_tsvector('english', content):
    #   - lowercases all words
    #   - removes English stop words (the, a, is, was...)
    #   - stems words (showing -> show, treated -> treat)
    #   - produces: 'pembrolizumab':1 'treat':2 'nsclc':3 ...
    #
    # 'english' = use English language dictionary for stemming
    # Without this trigger you'd need to manually compute tsvector
    # for every single insert — the trigger makes it automatic

    cur.execute("""
        CREATE OR REPLACE FUNCTION update_ts_content()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.ts_content := to_tsvector('english', NEW.content);
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
    """)

    cur.execute("""
        DROP TRIGGER IF EXISTS ts_content_trigger ON documents;
        CREATE TRIGGER ts_content_trigger
        BEFORE INSERT OR UPDATE ON documents
        FOR EACH ROW EXECUTE FUNCTION update_ts_content();
    """)
    print("  ts_content trigger    ✓  (auto-fills on every INSERT/UPDATE)")

    # ── VERIFY ────────────────────────────────────────────────────────
    cur.execute("""
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_name = 'documents'
        ORDER BY ordinal_position;
    """)
    columns = cur.fetchall()

    print(f"\nVerification — {len(columns)} columns created:")
    for col_name, data_type in columns:
        print(f"  {col_name:<25} {data_type}")

    cur.close()
    conn.close()

    print(f"\n{'='*55}")
    print("  setup_db.py complete — database ready for ingestion")
    print(f"{'='*55}")


if __name__ == "__main__":
    setup()
