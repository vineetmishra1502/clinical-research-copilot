"""
ingest.py — PubMed Ingestion Pipeline (Final — Full Hybrid Extraction)
=======================================================================
Fetches real clinical abstracts from PubMed for 5 oncology/pharma drugs,
applies structure-aware chunking, embeds with text-embedding-3-large,
and stores in Postgres with full metadata for filtered retrieval.

Schema alignment: matches setup_db.py columns exactly —
  content, embedding, ts_content (trigger-filled),
  source, doi, clinical_trial_id,
  chunk_index, total_chunks,
  drug, drug_combination, cancer_type, indication,
  study_type, sample_size, endpoints, pdl1_status, funding,
  evidence_level, year, journal, authors

Hybrid extraction — 4 fields use regex + LLM fallback:
  cancer_type     — 35+ types, new ones emerge yearly
  study_type      — new research designs emerge
  endpoints       — non-standard endpoint names exist
  drug_combination— complex multi-drug regimens vary in style

  Strategy per field:
    Step 1: Regex scan  — instant, free, covers 85-92% of abstracts
    Step 2: ONE combined LLM call for ALL missing fields simultaneously
            (not 4 separate calls — 4x cheaper and faster)
    Step 3: Safe defaults — never crashes the pipeline

Fields staying as pure regex (stable, closed vocabularies):
  sample_size, pdl1_status, funding, clinical_trial_id,
  year, journal, authors, doi (direct XML fields)

Chunking strategy (3 cases from real PubMed data analysis):
  Case 1: <= 600 tokens  -> single chunk     (~60% of abstracts)
  Case 2: section headers found -> split     (~30% of abstracts)
  Case 3: long, no headers -> recursive      (~10% of abstracts)
"""

import os
import re
import json
import time
import tiktoken
import psycopg2
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI
from pgvector.psycopg2 import register_vector
from Bio import Entrez
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ─────────────────────────────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────────────────────────────
load_dotenv()

Entrez.email = os.getenv("ENTREZ_EMAIL")
client       = OpenAI()

# 5 drugs chosen for domain depth and interview relevance:
# pembrolizumab + nivolumab = dominant PD-1 checkpoint inhibitors
# osimertinib   = AstraZeneca flagship EGFR inhibitor
# trastuzumab   = 25yr HER2 blockbuster, massive literature
# metformin     = adds diabetes angle, cross-therapeutic coverage
DRUGS = [
    "pembrolizumab",
    "nivolumab",
    "osimertinib",
    "trastuzumab",
    "metformin",
]

ABSTRACTS_PER_DRUG = 2000
EMBEDDING_MODEL    = "text-embedding-3-large"   # 3072 dimensions
ENCODER            = tiktoken.encoding_for_model("text-embedding-3-large")


# ─────────────────────────────────────────────────────────────────────
# 2. METADATA PATTERNS
# ─────────────────────────────────────────────────────────────────────

# ── Cancer type patterns (regex layer) ───────────────────────────────
# ORDER MATTERS: specific subtypes BEFORE general categories
# "her2-positive-breast" must match before "breast-cancer"
# otherwise HER2-specific papers are mislabeled as generic breast cancer
# Each tuple: (regex_pattern, label_to_store)
# Labels: lowercase, hyphenated — consistent for SQL WHERE filtering
CANCER_TYPE_PATTERNS = [
    # Lung
    (r"\bNSCLC\b|non.small.cell lung",           "nsclc"),
    (r"\bSCLC\b|small.cell lung",                "sclc"),
    (r"\blung cancer\b|lung carcinoma",           "lung-cancer"),
    # Breast — subtypes before generic
    (r"\bHER2.positive\b|HER2\+|HER-2 positive", "her2-positive-breast"),
    (r"\btriple.negative\b|\bTNBC\b",             "triple-negative-breast"),
    (r"\bhormone.receptor.positive\b|\bHR\+",     "hr-positive-breast"),
    (r"\bbreast cancer\b|breast carcinoma",        "breast-cancer"),
    # Blood cancers
    (r"\bdiffuse large B.cell\b|\bDLBCL\b",       "dlbcl"),
    (r"\bfollicular lymphoma\b",                   "follicular-lymphoma"),
    (r"\bhodgkin lymphoma\b",                      "hodgkin-lymphoma"),
    (r"\bmultiple myeloma\b",                      "multiple-myeloma"),
    (r"\bchronic myeloid\b|\bCML\b",               "cml"),
    (r"\bacute myeloid\b|\bAML\b",                 "aml"),
    (r"\bacute lymphoblastic\b|\bALL\b",           "all"),
    (r"\blymphoma\b",                              "lymphoma"),
    (r"\bleukemia\b",                              "leukemia"),
    (r"\bmyelodysplastic\b|\bMDS\b",               "mds"),
    # GI cancers
    (r"\bcolorectal\b|\bCRC\b|\bcolon cancer",    "colorectal-cancer"),
    (r"\bgastric cancer\b|\bstomach cancer",       "gastric-cancer"),
    (r"\bgastroesophageal\b|\bGEJ\b",              "gastroesophageal-cancer"),
    (r"\bpancreatic cancer\b|pancreatic duct",     "pancreatic-cancer"),
    (r"\bhepatocellular\b|\bHCC\b",                "hepatocellular-carcinoma"),
    (r"\bcholangiocarcinoma\b|bile duct cancer",   "cholangiocarcinoma"),
    # GU cancers
    (r"\brenal cell\b|\bRCC\b",                   "renal-cell-carcinoma"),
    (r"\bbladder cancer\b|urothelial carcinoma",   "bladder-cancer"),
    (r"\bprostate cancer\b|castration.resistant",  "prostate-cancer"),
    (r"\bendometrial cancer\b",                    "endometrial-cancer"),
    (r"\bcervical cancer\b",                       "cervical-cancer"),
    (r"\bovarian cancer\b",                        "ovarian-cancer"),
    # Other solid tumors
    (r"\buveal melanoma\b",                        "uveal-melanoma"),
    (r"\bmelanoma\b",                              "melanoma"),
    (r"\bhead and neck\b|\bHNSCC\b",               "head-and-neck-cancer"),
    (r"\bglioblastoma\b|\bGBM\b",                  "glioblastoma"),
    (r"\bthyroid cancer\b",                        "thyroid-cancer"),
    (r"\bsarcoma\b",                               "sarcoma"),
    (r"\bmesothelioma\b",                          "mesothelioma"),
    (r"\bMerkel cell\b",                           "merkel-cell-carcinoma"),
    # Non-cancer — specific before general
    (r"\btype 2 diabetes\b|\bT2DM\b|\btype II diabetes",  "type-2-diabetes"),
    (r"\btype 1 diabetes\b|\bT1DM\b|\btype I diabetes",   "type-1-diabetes"),
]

# ── Study type keywords ───────────────────────────────────────────────
# Ordered specific -> general (first match wins)
# Tuple: (keyword, study_type_label, evidence_level)
# evidence_level: 1=strongest (RCT/meta), 2=moderate, 3=weakest (case)
STUDY_TYPE_KEYWORDS = [
    ("meta-analysis",          "meta-analysis",        1),
    ("systematic review",      "systematic-review",    1),
    ("network meta-analysis",  "meta-analysis",        1),
    ("pooled analysis",        "pooled-analysis",      1),
    ("randomized controlled",  "RCT",                  1),
    ("randomised controlled",  "RCT",                  1),
    ("phase iii",              "phase-3-trial",        1),
    ("phase 3",                "phase-3-trial",        1),
    ("phase ii",               "phase-2-trial",        1),
    ("phase 2",                "phase-2-trial",        1),
    ("real-world",             "real-world-study",     2),
    ("real world",             "real-world-study",     2),
    ("retrospective cohort",   "retrospective-cohort", 2),
    ("cohort study",           "cohort-study",         2),
    ("clinical trial",         "clinical-trial",       2),
    ("observational",          "observational",        2),
    ("case-control",           "case-control",         2),
    ("case series",            "case-series",          3),
    ("case report",            "case-report",          3),
    ("review",                 "review",               2),
]

# evidence_level lookup — used after LLM returns a study_type string
EVIDENCE_LEVEL_MAP = {
    "RCT": 1, "meta-analysis": 1, "systematic-review": 1,
    "phase-3-trial": 1, "phase-2-trial": 1, "pooled-analysis": 1,
    "real-world-study": 2, "cohort-study": 2, "retrospective-cohort": 2,
    "clinical-trial": 2, "observational": 2, "review": 2,
    "case-control": 2, "case-report": 3, "case-series": 3,
}

# Valid study type labels — used to validate LLM output
VALID_STUDY_TYPES = set(EVIDENCE_LEVEL_MAP.keys())

# ── Endpoint patterns ─────────────────────────────────────────────────
# Standard oncology outcome abbreviations
# Stored comma-separated: "OS,PFS,ORR"
ENDPOINT_PATTERNS = [
    (r"\boverall survival\b|\bOS\b",             "OS"),
    (r"\bprogression.free survival\b|\bPFS\b",   "PFS"),
    (r"\bobjective response rate\b|\bORR\b",     "ORR"),
    (r"\bdisease.free survival\b|\bDFS\b",       "DFS"),
    (r"\bevent.free survival\b|\bEFS\b",         "EFS"),
    (r"\bcomplete response\b|\bCR\b",            "CR"),
    (r"\bpartial response\b|\bPR\b",             "PR"),
    (r"\bduration of response\b|\bDoR\b",        "DoR"),
    (r"\btime to progression\b|\bTTP\b",         "TTP"),
    (r"\badverse event\b|\bsafety\b",            "AE"),
]

VALID_ENDPOINTS = {"OS","PFS","ORR","DFS","EFS","CR","PR","DoR","TTP","AE"}

# ── PD-L1 status patterns ─────────────────────────────────────────────
# Fixed biomarker vocabulary — regex is sufficient, no LLM needed
PDL1_PATTERNS = [
    (r"PD.L1.{0,20}<\s*1%|TPS\s*<\s*1",         "TPS<1%"),
    (r"PD.L1.{0,20}1.49%|TPS\s*1.{0,5}49",      "TPS1-49%"),
    (r"PD.L1.{0,20}>=?\s*50%|TPS\s*>=?\s*50",   "TPS>=50%"),
    (r"PD.L1.{0,10}positive|PD.L1.{0,10}high",   "PD-L1-positive"),
    (r"PD.L1.{0,10}negative|PD.L1.{0,10}low",    "PD-L1-negative"),
]

# ── Funding patterns ──────────────────────────────────────────────────
# 3 stable categories — regex is sufficient, no LLM needed
FUNDING_PATTERNS = [
    (r"merck|msd|bristol.myers|astrazeneca|roche|pfizer|"
     r"novartis|sanofi|eli lilly|bayer|industry.funded",   "industry"),
    (r"NIH|national institutes|national cancer|"
     r"NCI|NCATS|government.funded",                        "NIH"),
    (r"non.profit|foundation|charity|academic.funded",      "non-profit"),
]

# ── Sample size patterns ──────────────────────────────────────────────
# Numeric patterns — very reliable, no LLM needed
SAMPLE_SIZE_PATTERNS = [
    r"\bn\s*=\s*(\d+)",           # n = 680  or  n=680
    r"\bN\s*=\s*(\d+)",           # N = 305
    r"(\d+)\s+patients",          # 680 patients
    r"(\d+)\s+participants",      # 305 participants
    r"(\d+)\s+subjects",          # 120 subjects
    r"enrolled\s+(\d+)",          # enrolled 450
    r"recruited\s+(\d+)",         # recruited 680
    r"analysed\s+(\d+)",          # analysed 1966
    r"analyzed\s+(\d+)",          # analyzed 339
    r"included\s+(\d+)\s+pat",    # included 228 patients
    r"randomized\s+(\d+)",        # randomized 1204
    r"randomised\s+(\d+)",        # randomised 1204
]

# ── Combo drug skip words ─────────────────────────────────────────────
# Words that look like drug partners but aren't
COMBO_SKIP_WORDS = {
    "the","a","an","placebo","chemotherapy","therapy",
    "treatment","best","standard","care","patients","plus"
}

# ── Section header pattern (chunking) ─────────────────────────────────
# Sourced from NLM's 5 official metacategories + real PubMed data
# re.IGNORECASE handles "Background:" AND "BACKGROUND:"
# Colon required — filters mid-sentence word matches
# Space after colon — filters URL artifacts ("Clinicaltrials: gov")
# Excludes: "Simple Summary" (MDPI lay summary), "Tweetable Abstract"
SECTION_HEADERS = (
    r"Background|Introduction|Context|Rationale|Overview|"
    r"Objective|Objectives|Aim|Aims|Purpose|Research Question|"
    r"Methods|Methodology|Materials and Methods|Study Design|"
    r"Patients and Methods|Design|Setting|Participants|"
    r"Interventions|Search Strategy|Selection Criteria|"
    r"Eligibility Criteria|"
    r"Results|Findings|Outcomes|Main Results|Main Outcome Measures|"
    r"Conclusions|Conclusion|Discussion|Summary|Interpretation|"
    r"Implications|Clinical Relevance|"
    r"What This Study Adds|What Is Known|What Is New"
)
SECTION_PATTERN = re.compile(
    rf"\n(?=({SECTION_HEADERS}):\s)",
    re.IGNORECASE
)


# ─────────────────────────────────────────────────────────────────────
# 3. HYBRID METADATA EXTRACTION
# ─────────────────────────────────────────────────────────────────────

def _regex_extract_cancer_type(combined: str) -> str | None:
    """Regex pass for cancer type. Returns label or None if no match."""
    for pattern, label in CANCER_TYPE_PATTERNS:
        if re.search(pattern, combined, re.IGNORECASE):
            return label
    return None


def _regex_extract_study_type(combined_lower: str) -> tuple[str, int] | tuple[None, None]:
    """Regex pass for study type. Returns (label, level) or (None, None)."""
    for keyword, stype, level in STUDY_TYPE_KEYWORDS:
        if keyword in combined_lower:
            return stype, level
    return None, None


def _regex_extract_endpoints(combined: str) -> str | None:
    """Regex pass for endpoints. Returns comma-separated string or None."""
    found = [ep for pattern, ep in ENDPOINT_PATTERNS
             if re.search(pattern, combined, re.IGNORECASE)]
    return ",".join(found) if found else None


def _regex_extract_drug_combination(combined_lower: str) -> str | None:
    """Regex pass for drug combination. Returns partner drug or None."""
    combo_patterns = [
        r"combined? with\s+([\w\-]+)",
        r"in combination with\s+([\w\-]+)",
        r"plus\s+([\w\-]+)",
        r"\+\s*([\w\-]+)",
    ]
    for cp in combo_patterns:
        m = re.search(cp, combined_lower)
        if m:
            partner = m.group(1).strip().rstrip(".,;")
            if partner not in COMBO_SKIP_WORDS and len(partner) > 2:
                return partner
    return None


def extract_metadata_llm(title: str, abstract: str,
                          missing_fields: list[str]) -> dict:
    """
    Single LLM call that extracts ALL missing metadata fields at once.

    Why one combined call instead of separate calls per field:
    - 1 HTTP request instead of N (N=up to 4)
    - LLM reads the abstract once, not N times
    - 4x cheaper and 4x faster
    - response_format=json_object guarantees valid JSON output

    Why gpt-4o-mini:
    - Simple structured extraction — no complex reasoning needed
    - 95% accuracy of gpt-4o on this specific task type
    - 15x cheaper than gpt-4o
    - ~800ms latency

    Why max_tokens=150:
    - JSON with 4 short fields never exceeds this
    - Prevents runaway responses

    Why temperature=0:
    - Deterministic output — same abstract = same result every time
    - No creativity needed for field extraction

    Returns dict with extracted values for requested fields.
    Returns safe defaults on any failure — never crashes pipeline.
    """

    # Dynamic prompt — only ask for fields we actually need
    # Saves tokens and keeps the prompt focused
    field_instructions = {
        "cancer_type": (
            '"cancer_type": primary disease studied. '
            'Lowercase, hyphenated. Examples: "nsclc", '
            '"triple-negative-breast-cancer", "cholangiocarcinoma", '
            '"type-2-diabetes", "multiple-myeloma". '
            'Use "oncology" if no specific disease found.'
        ),
        "study_type": (
            '"study_type": research design. Must be one of: '
            'RCT, meta-analysis, systematic-review, phase-3-trial, '
            'phase-2-trial, real-world-study, retrospective-cohort, '
            'cohort-study, case-report, case-series, review, '
            'observational, clinical-trial, pooled-analysis'
        ),
        "endpoints": (
            '"endpoints": comma-separated outcome measures. '
            'Use only: OS, PFS, ORR, DFS, EFS, CR, PR, DoR, TTP, AE. '
            'Example: "OS,PFS,ORR". Empty string if none found.'
        ),
        "drug_combination": (
            '"drug_combination": partner drug(s) if combination therapy. '
            'Example: "carboplatin", "bevacizumab+pemetrexed". '
            'Empty string if monotherapy or unclear.'
        ),
    }

    needed_instructions = "\n".join(
        f"  {field_instructions[f]}"
        for f in missing_fields
        if f in field_instructions
    )

    prompt = f"""You are a medical literature metadata extractor.
Extract the following fields from this abstract.
Return ONLY a valid JSON object. No explanation. No markdown. No backticks.

Fields to extract:
{needed_instructions}

Title: {title}
Abstract: {abstract[:1500]}

JSON:"""

    # Safe defaults — returned if API fails
    defaults = {
        "cancer_type":      "oncology",
        "study_type":       "observational",
        "endpoints":        "",
        "drug_combination": "",
    }

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0,
            response_format={"type": "json_object"},  # guarantees valid JSON
        )

        raw    = response.choices[0].message.content.strip()
        result = json.loads(raw)

        sanitized = {}

        # ── cancer_type ───────────────────────────────────────────────
        if "cancer_type" in missing_fields:
            ct = str(result.get("cancer_type", "")).lower()
            ct = re.sub(r'[^a-z0-9\-]', '', ct)[:60]
            # Reject if empty, too long, or looks like a sentence
            if ct and ct.count('-') <= 6:
                sanitized["cancer_type"] = ct
            else:
                sanitized["cancer_type"] = defaults["cancer_type"]

        # ── study_type ────────────────────────────────────────────────
        if "study_type" in missing_fields:
            st = str(result.get("study_type", "")).strip()
            st = re.sub(r'[^a-zA-Z0-9\-]', '', st)
            # Only accept values from our controlled vocabulary
            if st in VALID_STUDY_TYPES:
                sanitized["study_type"] = st
            else:
                sanitized["study_type"] = defaults["study_type"]

        # ── endpoints ─────────────────────────────────────────────────
        if "endpoints" in missing_fields:
            ep_raw = str(result.get("endpoints", ""))
            # Keep only valid abbreviations from controlled list
            found = [e.strip() for e in ep_raw.split(",")
                     if e.strip() in VALID_ENDPOINTS]
            sanitized["endpoints"] = ",".join(found)

        # ── drug_combination ──────────────────────────────────────────
        if "drug_combination" in missing_fields:
            dc = str(result.get("drug_combination", "")).lower()
            dc = re.sub(r'[^a-z0-9\-\+]', '', dc)[:100]
            sanitized["drug_combination"] = dc

        return sanitized

    except Exception as e:
        print(f"    LLM extraction error: {type(e).__name__}: {e}")
        # Return safe defaults for all requested fields — never crash
        return {f: defaults.get(f, "") for f in missing_fields}


def extract_metadata_hybrid(title: str, abstract: str) -> dict:
    """
    Master hybrid extractor — runs for EVERY abstract during parsing.

    For each of the 4 hybrid fields:
      Step 1: Regex scan (instant, free)
              → if match found: store result, field is done
              → if no match:    add to missing_fields list

    After all 4 regex passes:
      Step 2: If any fields are missing → ONE combined LLM call
              extracts all missing fields simultaneously
              (never N separate calls — always 1 maximum)

      Step 3: Merge regex results + LLM results
              Safe defaults guaranteed — never returns None for any field

    Returns:
      dict with keys: cancer_type, study_type, evidence_level,
                      endpoints, drug_combination, extraction_method
      extraction_method: "regex" | "hybrid" — for monitoring stats
    """
    combined      = title + " " + abstract
    combined_lower = combined.lower()

    results = {}
    missing = []   # fields regex could not fill

    # ── Step 1a: cancer_type ──────────────────────────────────────────
    ct = _regex_extract_cancer_type(combined)
    if ct:
        results["cancer_type"] = ct
    else:
        missing.append("cancer_type")

    # ── Step 1b: study_type ───────────────────────────────────────────
    st, level = _regex_extract_study_type(combined_lower)
    if st:
        results["study_type"]     = st
        results["evidence_level"] = level
    else:
        missing.append("study_type")
        results["evidence_level"] = 2  # placeholder — LLM will refine

    # ── Step 1c: endpoints ────────────────────────────────────────────
    ep = _regex_extract_endpoints(combined)
    if ep:
        results["endpoints"] = ep
    else:
        missing.append("endpoints")

    # ── Step 1d: drug_combination ─────────────────────────────────────
    dc = _regex_extract_drug_combination(combined_lower)
    if dc:
        results["drug_combination"] = dc
    else:
        missing.append("drug_combination")

    # ── Step 2: ONE combined LLM call for ALL missing fields ──────────
    extraction_method = "regex"

    if missing:
        # Single API call — extracts all missing fields at once
        llm_results = extract_metadata_llm(title, abstract, missing)

        # Merge LLM results into our results dict
        results.update(llm_results)
        extraction_method = "hybrid"

        # Derive evidence_level from LLM-returned study_type
        # (only needed if study_type was in missing_fields)
        if "study_type" in missing and "study_type" in results:
            results["evidence_level"] = EVIDENCE_LEVEL_MAP.get(
                results["study_type"], 2
            )

    # ── Step 3: Guarantee all fields are present ──────────────────────
    # Fill any gaps that somehow slipped through both paths
    results.setdefault("cancer_type",      "oncology")
    results.setdefault("study_type",       "observational")
    results.setdefault("evidence_level",   2)
    results.setdefault("endpoints",        "")
    results.setdefault("drug_combination", "")
    results["extraction_method"] = extraction_method

    return results


# ─────────────────────────────────────────────────────────────────────
# 4. DATABASE CONNECTION
# ─────────────────────────────────────────────────────────────────────

def get_db_connection():
    """
    Opens Postgres connection and registers HALFVEC type support.
    register_vector() is mandatory — without it psycopg2 cannot
    serialize numpy float32 arrays into the HALFVEC column format.
    """
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", 5432),
        dbname=os.getenv("DB_NAME", "pharma_rag"),
        user=os.getenv("DB_USER", "vineet"),
        password=os.getenv("DB_PASSWORD", "devpassword")
    )
    register_vector(conn)
    return conn


# ─────────────────────────────────────────────────────────────────────
# 5. PUBMED FETCHING
# ─────────────────────────────────────────────────────────────────────

def fetch_pubmed_ids(drug: str, max_results: int) -> list[str]:
    """
    Step 1: esearch — returns PubMed IDs only (lightweight).
    Two-step design: IDs first, content second.
    If content fetch fails midway, IDs are safe to retry.
    [Title/Abstract] scopes to relevant fields only.
    sort=relevance puts most cited articles first.
    """
    print(f"\n  Searching PubMed: '{drug}[Title/Abstract]'")
    handle = Entrez.esearch(
        db="pubmed",
        term=f"{drug}[Title/Abstract]",
        retmax=max_results,
        sort="relevance"
    )
    record = Entrez.read(handle)
    handle.close()
    pmids = record["IdList"]
    print(f"  Total on PubMed: {record['Count']} | Fetching: {len(pmids)}")
    return pmids


def fetch_abstracts_batch(pmids: list[str]) -> list[dict]:
    """
    Step 2: efetch — full content for a list of PMIDs.
    Batches of 200: stays under NCBI's 3 req/sec rate limit.
    sleep(0.4) = 2.5 req/sec — safely under the limit.
    Graceful batch-level error handling — one bad batch skips,
    the others still complete.
    """
    abstracts  = []
    batch_size = 200

    for i in range(0, len(pmids), batch_size):
        batch = pmids[i:i + batch_size]
        try:
            handle = Entrez.efetch(
                db="pubmed",
                id=",".join(batch),
                rettype="abstract",
                retmode="xml"
            )
            records = Entrez.read(handle)
            handle.close()
            for article in records["PubmedArticle"]:
                parsed = parse_pubmed_article(article)
                if parsed:
                    abstracts.append(parsed)
            time.sleep(0.4)
        except Exception as e:
            print(f"  Batch {i//batch_size} error: {e}. Skipping.")
            continue

    return abstracts


# ─────────────────────────────────────────────────────────────────────
# 6. ARTICLE PARSING
# ─────────────────────────────────────────────────────────────────────

def parse_pubmed_article(article: dict) -> dict | None:
    """
    Extracts all metadata from one raw PubMed XML record.

    Returns None if no abstract — nothing to embed.

    Pure regex fields (stable, closed vocabularies):
      year, journal, authors, doi    — direct XML extraction
      clinical_trial_id              — DataBankList + NCT regex
      sample_size                    — numeric regex, very reliable
      pdl1_status                    — fixed 5-value vocabulary
      funding                        — 3 stable categories

    Hybrid fields (regex first, LLM fallback if needed):
      cancer_type, study_type, endpoints, drug_combination
      → all handled in one call to extract_metadata_hybrid()

    Schema alignment: every key in the returned dict maps directly
    to a column in the documents table defined in setup_db.py
    """
    try:
        medline = article["MedlineCitation"]
        art     = medline["Article"]

        # ── PMID ──────────────────────────────────────────────────────
        pmid = str(medline["PMID"])

        # ── TITLE ─────────────────────────────────────────────────────
        # re.sub strips HTML tags some journals embed in titles
        title = str(art.get("ArticleTitle", ""))
        title = re.sub(r"<[^>]+>", "", title)

        # ── ABSTRACT ──────────────────────────────────────────────────
        abstract_data = art.get("Abstract", {}).get("AbstractText", [])
        if not abstract_data:
            return None  # no abstract — skip this article

        if isinstance(abstract_data, list):
            # Structured abstract — normalize labels to Title Case
            # "BACKGROUND" -> "Background" so SECTION_PATTERN works
            # regardless of how the journal stored the label
            parts = []
            for section in abstract_data:
                label = section.attributes.get("Label", "")
                text  = str(section)
                if label:
                    label = label.title()
                    parts.append(f"{label}:\n{text}")
                else:
                    parts.append(text)
            abstract = "\n".join(parts)
        else:
            abstract = str(abstract_data)

        # Title prepended — every chunk knows its article context
        full_content = f"{title}\n\n{abstract}"

        # ── YEAR ──────────────────────────────────────────────────────
        pub_date = (art.get("Journal", {})
                       .get("JournalIssue", {})
                       .get("PubDate", {}))
        year = int(str(pub_date.get("Year", 0))) if "Year" in pub_date else 0

        # ── JOURNAL ───────────────────────────────────────────────────
        journal = str(art.get("Journal", {}).get("Title", "Unknown Journal"))

        # ── AUTHORS ───────────────────────────────────────────────────
        # First 3 only — "Smith J, Jones A, Lee B"
        authors_list = art.get("AuthorList", [])
        author_names = []
        for author in authors_list[:3]:
            last = str(author.get("LastName", ""))
            init = str(author.get("Initials", ""))
            if last:
                author_names.append(f"{last} {init}".strip())
        authors = ", ".join(author_names) if author_names else "Unknown"

        # ── DOI ───────────────────────────────────────────────────────
        doi = ""
        for loc in art.get("ELocationID", []):
            if loc.attributes.get("EIdType") == "doi":
                doi = str(loc)
                break

        # ── CLINICAL TRIAL ID ─────────────────────────────────────────
        # DataBankList first (official structured field)
        # Fallback: NCT\d{8} regex in title + abstract
        # Purpose: deduplication — same trial in multiple papers
        clinical_trial_id = ""
        for db in art.get("DataBankList", []):
            if "ClinicalTrials" in str(db.get("DataBankName", "")):
                acc_list = db.get("AccessionNumberList", [])
                if acc_list:
                    clinical_trial_id = str(acc_list[0])
                    break
        if not clinical_trial_id:
            nct_matches = re.findall(r"NCT\d{8}", full_content)
            if nct_matches:
                clinical_trial_id = nct_matches[0]

        # ── SAMPLE SIZE (pure regex) ───────────────────────────────────
        # Numeric patterns are highly reliable — no LLM needed
        # Sanity range: 5 to 500,000 (realistic clinical trial range)
        sample_size = None
        for pattern in SAMPLE_SIZE_PATTERNS:
            match = re.search(pattern, abstract, re.IGNORECASE)
            if match:
                candidate = int(match.group(1))
                if 5 <= candidate <= 500_000:
                    sample_size = candidate
                    break

        # ── PD-L1 STATUS (pure regex) ──────────────────────────────────
        # Fixed 5-value vocabulary — regex is sufficient
        pdl1_status = ""
        for pattern, status in PDL1_PATTERNS:
            if re.search(pattern, full_content, re.IGNORECASE):
                pdl1_status = status
                break

        # ── FUNDING (pure regex) ───────────────────────────────────────
        # 3 stable categories — regex is sufficient
        combined_lower = (title + " " + abstract).lower()
        funding = "unknown"
        for pattern, funder in FUNDING_PATTERNS:
            if re.search(pattern, combined_lower):
                funding = funder
                break

        # ── HYBRID EXTRACTION — all 4 fields in one function call ──────
        # extract_metadata_hybrid() runs regex first for each field
        # then fires ONE combined LLM call for any that failed regex
        # Returns: cancer_type, study_type, evidence_level,
        #          endpoints, drug_combination, extraction_method
        hybrid = extract_metadata_hybrid(title, abstract)

        return {
            # Core content
            "pmid":               pmid,
            "content":            full_content,
            # Publication identity (pure XML/regex)
            "source":             pmid,
            "doi":                doi,
            "clinical_trial_id":  clinical_trial_id,
            # Drug context (hybrid)
            "drug_combination":   hybrid["drug_combination"],
            # Disease context (hybrid)
            "cancer_type":        hybrid["cancer_type"],
            "indication":         hybrid["cancer_type"],  # mirrors cancer_type
            # Study characteristics
            "study_type":         hybrid["study_type"],         # hybrid
            "evidence_level":     hybrid["evidence_level"],     # derived from study_type
            "sample_size":        sample_size,                  # pure regex
            "endpoints":          hybrid["endpoints"],          # hybrid
            "pdl1_status":        pdl1_status,                  # pure regex
            "funding":            funding,                       # pure regex
            # Publication info (pure XML)
            "year":               year,
            "journal":            journal,
            "authors":            authors,
            # Internal monitoring (not stored in DB)
            "extraction_method":  hybrid["extraction_method"],
        }

    except Exception:
        return None  # skip malformed records — never crash pipeline


# ─────────────────────────────────────────────────────────────────────
# 7. STRUCTURE-AWARE CHUNKING
# ─────────────────────────────────────────────────────────────────────

def is_url_artifact(text: str) -> bool:
    """
    Detects URL fragments that create false section headers.
    Real PubMed example: "Clinicaltrials: gov" from a broken URL.
    Detection: content after header starts with URL fragment word.
    """
    stripped   = text.strip()
    first_word = stripped.split()[0].lower().rstrip(").,") if stripped else ""
    url_frags  = {"gov", "org", "com", "net", "edu"}
    return (first_word in url_frags or
            stripped.lower().startswith(("http", "www", "ftp")))


def chunk_abstract(text: str) -> list[dict]:
    """
    Structure-aware chunking — 3 cases from real PubMed data analysis.

    Case 1 (<= 600 tokens): single chunk — keeps context intact
    Case 2 (section headers found): split on boundaries
      - SECTION_PATTERN: re.IGNORECASE + colon + space after colon
      - 40+ label variants covering all NLM metacategories
      - URL artifacts filtered by is_url_artifact()
      - Label preserved: "Methods:\ncontent text..."
    Case 3 (long, no headers): RecursiveCharacterTextSplitter
      - chunk_size=500, chunk_overlap=50
      - Split priority: paragraphs > lines > sentences > words
    """
    token_count = len(ENCODER.encode(text))

    # Case 1: short — keep whole
    if token_count <= 600:
        return [{"content": text, "chunk_index": 0, "total_chunks": 1}]

    # Case 2: section headers found
    # Capturing group in SECTION_PATTERN means split() includes
    # the matched header names: ["intro", "Methods", "text...", ...]
    sections = SECTION_PATTERN.split(text)

    if len(sections) > 1:
        chunks = []
        if sections[0].strip():
            chunks.append(sections[0].strip())

        i = 1
        while i < len(sections) - 1:
            header  = sections[i].strip()
            content = sections[i + 1].strip()
            if not is_url_artifact(content):
                chunks.append(f"{header}:\n{content}")
            i += 2

        valid = [c for c in chunks if c.strip()]
        if valid:
            return [
                {"content": c, "chunk_index": idx, "total_chunks": len(valid)}
                for idx, c in enumerate(valid)
            ]

    # Case 3: no headers — recursive splitter
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="text-embedding-3-large",
        chunk_size=500,
        chunk_overlap=50
    )
    raw = splitter.split_text(text)
    return [
        {"content": c, "chunk_index": idx, "total_chunks": len(raw)}
        for idx, c in enumerate(raw)
    ]


# ─────────────────────────────────────────────────────────────────────
# 8. EMBEDDING
# ─────────────────────────────────────────────────────────────────────

def embed_chunks(texts: list[str]) -> list[list[float]]:
    """
    Embeds texts using text-embedding-3-large (3072 dims).
    Batches of 100: stays under OpenAI's rate limits.
    sleep(0.1) = 600 req/min — under the 3000 RPM tier 1 limit.
    Zero-vector fallback: keeps pipeline running on API errors.
    Zero vectors have near-zero cosine similarity — never retrieved.
    """
    embeddings = []
    for i in range(0, len(texts), 100):
        batch = texts[i:i + 100]
        try:
            resp = client.embeddings.create(
                input=batch, model=EMBEDDING_MODEL
            )
            embeddings.extend([item.embedding for item in resp.data])
            time.sleep(0.1)
        except Exception as e:
            print(f"  Embedding error batch {i//100}: {e}")
            embeddings.extend([[0.0] * 3072] * len(batch))
    return embeddings


# ─────────────────────────────────────────────────────────────────────
# 9. DATABASE STORAGE
# ─────────────────────────────────────────────────────────────────────

def store_chunks(conn, chunks_data: list[dict]):
    """
    Bulk INSERT via executemany() — one DB round-trip for all rows.

    Column order matches setup_db.py CREATE TABLE exactly:
      content, embedding, source, doi, clinical_trial_id,
      chunk_index, total_chunks, drug, drug_combination,
      cancer_type, indication, study_type, sample_size,
      endpoints, pdl1_status, funding, evidence_level,
      year, journal, authors

    ts_content NOT included — trigger fills it automatically.
    np.float32 -> register_vector() -> Postgres HALFVEC.
    None sample_size -> SQL NULL (not zero — means "not found").
    ON CONFLICT DO NOTHING — safe to re-run without duplicates.
    """
    cur  = conn.cursor()
    rows = [
        (
            item["content"],                           # content
            np.array(item["embedding"], dtype=np.float32),  # embedding
            item["source"],                            # source (PMID)
            item.get("doi", ""),                       # doi
            item.get("clinical_trial_id", ""),         # clinical_trial_id
            item["chunk_index"],                       # chunk_index
            item["total_chunks"],                      # total_chunks
            item["drug"],                              # drug
            item.get("drug_combination", ""),          # drug_combination
            item.get("cancer_type", "oncology"),       # cancer_type
            item.get("indication", "oncology"),        # indication
            item.get("study_type", "observational"),   # study_type
            item.get("sample_size"),                   # sample_size (None=NULL)
            item.get("endpoints", ""),                 # endpoints
            item.get("pdl1_status", ""),               # pdl1_status
            item.get("funding", "unknown"),            # funding
            item.get("evidence_level", 2),             # evidence_level
            item.get("year", 0),                       # year
            item.get("journal", ""),                   # journal
            item.get("authors", ""),                   # authors
        )
        for item in chunks_data
    ]

    cur.executemany("""
        INSERT INTO documents (
            content, embedding,
            source, doi, clinical_trial_id,
            chunk_index, total_chunks,
            drug, drug_combination,
            cancer_type, indication,
            study_type, sample_size,
            endpoints, pdl1_status, funding,
            evidence_level,
            year, journal, authors
        ) VALUES (
            %s, %s,
            %s, %s, %s,
            %s, %s,
            %s, %s,
            %s, %s,
            %s, %s,
            %s, %s, %s,
            %s,
            %s, %s, %s
        )
        ON CONFLICT DO NOTHING
    """, rows)

    conn.commit()
    cur.close()


# ─────────────────────────────────────────────────────────────────────
# 10. MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────

def ingest_drug(drug: str, conn):
    """
    Complete pipeline for one drug:
    Fetch PMIDs -> Fetch abstracts -> Parse + hybrid extract
    -> Chunk -> Embed -> Store

    Prints detailed monitoring stats at each stage.
    """
    print(f"\n{'='*65}")
    print(f"  Processing: {drug.upper()}")
    print(f"{'='*65}")

    # Step 1: Get PMIDs
    pmids = fetch_pubmed_ids(drug, ABSTRACTS_PER_DRUG)
    if not pmids:
        print(f"  No articles found for {drug}. Skipping.")
        return

    # Step 2: Fetch and parse all abstracts
    print(f"  Fetching {len(pmids)} abstracts from PubMed...")
    abstracts = fetch_abstracts_batch(pmids)
    n         = len(abstracts)
    print(f"  Retrieved {n} abstracts with content")

    # Metadata quality stats
    with_doi   = sum(1 for a in abstracts if a.get("doi"))
    with_nct   = sum(1 for a in abstracts if a.get("clinical_trial_id"))
    with_size  = sum(1 for a in abstracts if a.get("sample_size"))
    with_ep    = sum(1 for a in abstracts if a.get("endpoints"))
    with_combo = sum(1 for a in abstracts if a.get("drug_combination"))
    n_hybrid   = sum(1 for a in abstracts
                     if a.get("extraction_method") == "hybrid")
    n_regex    = n - n_hybrid

    print(f"  Pure regex fields:")
    print(f"    DOI         : {with_doi}/{n}")
    print(f"    NCT ID      : {with_nct}/{n}")
    print(f"    Sample size : {with_size}/{n}")
    print(f"    PD-L1 status: {sum(1 for a in abstracts if a.get('pdl1_status'))}/{n}")
    print(f"  Hybrid fields (cancer_type + study_type + endpoints + drug_combination):")
    print(f"    Regex only  : {n_regex}/{n} ({n_regex/n*100:.0f}%) — $0.00")
    llm_cost = n_hybrid * 0.0002
    print(f"    LLM used    : {n_hybrid}/{n} ({n_hybrid/n*100:.0f}%) — ~${llm_cost:.3f}")
    print(f"    With endpoints   : {with_ep}/{n}")
    print(f"    With combination : {with_combo}/{n}")

    # Step 3: Chunk with case tracking
    print(f"  Chunking (structure-aware)...")
    all_chunks  = []
    case_counts = {1: 0, 2: 0, 3: 0}

    for abstract in abstracts:
        tc = len(ENCODER.encode(abstract["content"]))
        if tc <= 600:
            case = 1
        else:
            secs = SECTION_PATTERN.split(abstract["content"])
            case = 2 if len(secs) > 1 else 3
        case_counts[case] += 1

        for chunk in chunk_abstract(abstract["content"]):
            all_chunks.append({
                # All metadata from the parsed abstract
                "drug":              drug,
                "source":            abstract["source"],
                "doi":               abstract.get("doi", ""),
                "clinical_trial_id": abstract.get("clinical_trial_id", ""),
                "year":              abstract.get("year", 0),
                "journal":           abstract.get("journal", ""),
                "authors":           abstract.get("authors", ""),
                "study_type":        abstract.get("study_type", "observational"),
                "evidence_level":    abstract.get("evidence_level", 2),
                "sample_size":       abstract.get("sample_size"),
                "cancer_type":       abstract.get("cancer_type", "oncology"),
                "indication":        abstract.get("indication", "oncology"),
                "drug_combination":  abstract.get("drug_combination", ""),
                "endpoints":         abstract.get("endpoints", ""),
                "pdl1_status":       abstract.get("pdl1_status", ""),
                "funding":           abstract.get("funding", "unknown"),
                # Chunk-level fields
                "content":           chunk["content"],
                "chunk_index":       chunk["chunk_index"],
                "total_chunks":      chunk["total_chunks"],
            })

    total = len(all_chunks)
    avg   = total / n if n else 0
    print(f"  {total} chunks from {n} abstracts ({avg:.1f} avg/abstract)")
    print(f"  Cases — Short:{case_counts[1]} | "
          f"Section-split:{case_counts[2]} | Recursive:{case_counts[3]}")

    # Step 4: Embed
    print(f"  Embedding {total} chunks with {EMBEDDING_MODEL}...")
    texts      = [c["content"] for c in all_chunks]
    embeddings = []

    for i in tqdm(range(0, len(texts), 100), desc=f"  {drug}"):
        embeddings.extend(embed_chunks(texts[i:i + 100]))

    for chunk, emb in zip(all_chunks, embeddings):
        chunk["embedding"] = emb

    # Step 5: Store
    print(f"  Storing {total} chunks in Postgres...")
    store_chunks(conn, all_chunks)
    print(f"  {drug.upper()} complete")


def main():
    """
    Entry point — sequential ingestion for all 5 drugs.
    Sequential: NCBI rate limits + safer to monitor + easier to debug.
    Estimated runtime: 30-45 minutes for 10,000 abstracts.
    """
    print("PubMed Ingestion Pipeline — Full Hybrid Extraction")
    print(f"  Drugs:           {', '.join(DRUGS)}")
    print(f"  Per drug:        {ABSTRACTS_PER_DRUG} abstracts")
    print(f"  Embedding model: {EMBEDDING_MODEL} (3072 dims -> HALFVEC)")
    print(f"  Hybrid fields:   cancer_type, study_type, endpoints, drug_combination")
    print(f"  Regex only:      sample_size, pdl1_status, funding, clinical_trial_id")
    print(f"  Est. runtime:    30-45 minutes total")

    conn = get_db_connection()

    try:
        for drug in DRUGS:
            ingest_drug(drug, conn)

        # Final summary table
        cur = conn.cursor()
        cur.execute("""
            SELECT
                drug,
                COUNT(*)                          AS chunks,
                COUNT(DISTINCT source)            AS articles,
                AVG(sample_size)::int             AS avg_n,
                COUNT(*) FILTER (WHERE clinical_trial_id != '') AS nct,
                COUNT(*) FILTER (WHERE endpoints  != '')        AS ep,
                COUNT(*) FILTER (WHERE cancer_type != 'oncology') AS specific,
                COUNT(*) FILTER (WHERE study_type  = 'RCT')    AS rcts
            FROM documents
            GROUP BY drug
            ORDER BY drug;
        """)
        rows = cur.fetchall()
        cur.close()

        print(f"\n{'='*80}")
        print("  INGESTION COMPLETE")
        print(f"  {'Drug':<20} {'Chunks':>7} {'Articles':>9} "
              f"{'Avg N':>7} {'NCT':>5} {'EP':>5} {'Cancer%':>8} {'RCTs':>6}")
        print(f"  {'-'*70}")
        total_chunks = 0
        for drug, chunks, articles, avg_n, nct, ep, specific, rcts in rows:
            total_chunks += chunks
            cancer_pct    = f"{specific/chunks*100:.0f}%" if chunks else "N/A"
            print(f"  {drug:<20} {chunks:>7,} {articles:>9,} "
                  f"{str(avg_n) if avg_n else 'N/A':>7} "
                  f"{nct:>5} {ep:>5} {cancer_pct:>8} {rcts:>6}")
        print(f"  {'-'*70}")
        print(f"  {'TOTAL':<20} {total_chunks:>7,}")
        print(f"{'='*80}")
        print("  Ready for hybrid retrieval")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
