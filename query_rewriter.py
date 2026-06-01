"""
query_rewriter.py — Intelligent Query Rewriting via LangChain
==============================================================
Python 3.13 optimisations applied:
  - str | None  instead of Optional[str]     — cleaner union syntax
  - list[str]   instead of List[str]         — built-in generics
  - tuple[...]  instead of Tuple[...]        — built-in generics
  - Literal[*DB_VOCABULARY["drugs"]]         — no sys.version_info guard needed
  - No typing imports for Optional/List/Dict/Tuple — all built-in

Why LangChain instead of raw OpenAI client:
  - with_structured_output(RewriterOutput) passes Pydantic schema to model
    at generation time — model constrained to exact field names and types
  - ChatPromptTemplate makes prompts versioned, typed, and LangSmith-traced
  - ainvoke() is async — consistent with the rest of the async stack
  - Every call auto-traced in LangSmith alongside retrieval and agent traces

Why with_structured_output() over json_object mode:
  json_object  — guarantees valid JSON only. Model can return any keys.
                 Wrong field names → Pydantic validation fails → fallback.
  structured   — full schema sent to model at generation time.
                 Model constrained to exact field names and value types.
                 .invoke() returns typed Pydantic instance directly.

Three-layer protection:
  Layer 1: Literal types  → model constrained at generation (field values)
  Layer 2: field_validators → validate + coerce post-generation (safety net)
  Layer 3: Exception / refusal → _safe_fallback() — never crashes pipeline

Architecture position:
  User query → query_rewriter.py → retriever.py → grader.py → agents.py
"""

import re
import asyncio
from dataclasses import dataclass
from typing import Literal                        # still needed for Literal itself

from pydantic import BaseModel, Field, field_validator, model_validator
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()


# ─────────────────────────────────────────────────────────────────────
# 1. DATABASE VOCABULARY
# ─────────────────────────────────────────────────────────────────────
# Exact values stored in the database.
# Three jobs:
#   a) Builds Literal type constraints → model constrained at generation
#   b) Feeds _normalize_aliases() → pre-processing before LLM call
#   c) Injected into system prompt → model understands domain vocabulary
# Update when new drugs / cancer types are ingested.

DB_VOCABULARY: dict[str, list[str] | dict[str, str] | dict[int, str]] = {
    "drugs": [
        "pembrolizumab", "nivolumab", "osimertinib",
        "trastuzumab", "metformin",
    ],

    # Trade names / abbreviations → canonical generic name
    # Resolved in _normalize_aliases() BEFORE the LLM call
    "drug_aliases": {
        "keytruda":   "pembrolizumab",
        "pembro":     "pembrolizumab",
        "mk-3475":    "pembrolizumab",
        "opdivo":     "nivolumab",
        "nivo":       "nivolumab",
        "tagrassi":   "osimertinib",   # common misspelling
        "tagrisso":   "osimertinib",
        "osi":        "osimertinib",
        "azd9291":    "osimertinib",
        "herceptin":  "trastuzumab",
        "glucophage": "metformin",
    },

    "cancer_types": [
        "nsclc", "sclc", "lung-cancer",
        "breast-cancer", "her2-positive-breast",
        "triple-negative-breast", "hr-positive-breast",
        "melanoma", "uveal-melanoma",
        "renal-cell-carcinoma", "bladder-cancer",
        "colorectal-cancer", "gastric-cancer",
        "gastroesophageal-cancer", "hepatocellular-carcinoma",
        "cholangiocarcinoma", "pancreatic-cancer",
        "prostate-cancer", "endometrial-cancer",
        "cervical-cancer", "ovarian-cancer",
        "head-and-neck-cancer", "glioblastoma",
        "thyroid-cancer", "sarcoma", "mesothelioma",
        "merkel-cell-carcinoma", "multiple-myeloma",
        "dlbcl", "follicular-lymphoma", "hodgkin-lymphoma",
        "lymphoma", "leukemia", "aml", "cml", "all", "mds",
        "type-2-diabetes", "type-1-diabetes",
    ],

    "study_types": [
        "RCT", "meta-analysis", "systematic-review",
        "phase-3-trial", "phase-2-trial", "pooled-analysis",
        "real-world-study", "retrospective-cohort",
        "cohort-study", "clinical-trial", "observational",
        "case-control", "case-report", "case-series", "review",
    ],

    "endpoints": [
        "OS", "PFS", "ORR", "DFS", "EFS",
        "CR", "PR", "DoR", "TTP", "AE",
    ],

    "pdl1_values": [
        "TPS<1%", "TPS1-49%", "TPS>=50%",
        "PD-L1-positive", "PD-L1-negative",
    ],

    "evidence_levels": {
        1: "RCT, meta-analysis, phase-3-trial (strongest evidence)",
        2: "cohort, real-world, observational (moderate evidence)",
        3: "case-report, case-series (weakest evidence)",
    },
}


# ─────────────────────────────────────────────────────────────────────
# 2. LITERAL TYPES — schema-level vocabulary constraints
# ─────────────────────────────────────────────────────────────────────
# Python 3.13 — Literal[*tuple(...)] is fully supported.
# No sys.version_info guard needed anymore.
# The model CANNOT return a value outside these lists.
# "pembro", "Pembrolizumab", "invented-drug" are impossible outputs.
#
# str | None replaces Optional[str] — Python 3.10+ union syntax,
# fully idiomatic in 3.13. Cleaner, no Optional import needed.

DrugLiteral       = Literal[*DB_VOCABULARY["drugs"]] | None          # type: ignore[misc]
CancerTypeLiteral = Literal[*DB_VOCABULARY["cancer_types"]] | None   # type: ignore[misc]
StudyTypeLiteral  = Literal[*DB_VOCABULARY["study_types"]] | None    # type: ignore[misc]
EndpointLiteral   = Literal[*DB_VOCABULARY["endpoints"]] | None      # type: ignore[misc]
PDL1Literal       = Literal[*DB_VOCABULARY["pdl1_values"]] | None    # type: ignore[misc]
FundingLiteral    = Literal["industry", "NIH", "non-profit"] | None
ConfidenceLiteral = Literal["high", "medium", "low"]


class SearchStrategy(str):
    """
    Search strategy chosen by the LLM based on query type.
    Mapped to (dense_weight, bm25_weight) in retriever.py.

    Why an Enum-style class with Literal on the field (not Python Enum):
    Pydantic's with_structured_output serialises Literal fields cleanly
    into JSON schema enum constraints. Python Enum classes require extra
    Pydantic config to serialize the same way.

    Five discrete strategies — not a free float:
    Free floats (0.743, 0.21) imply false precision. The LLM has no
    empirical basis for fine-grained floats. Discrete strategies with
    clear semantic meaning give the LLM a real, validatable choice.

    Weight mapping (dense_weight, bm25_weight):
      dense_only   → (1.0, 0.0)  pure semantic, no keyword noise
      dense_heavy  → (0.7, 0.3)  semantic-leaning, some keyword support
      equal        → (0.5, 0.5)  balanced — default for most queries
      bm25_heavy   → (0.3, 0.7)  keyword-leaning, some semantic support
      bm25_only    → (0.0, 1.0)  pure keyword, no semantic noise
    """
    pass


SearchStrategyLiteral = Literal[
    "dense_only",    # (1.0, 0.0) mechanism/concept queries — pure semantic
    "dense_heavy",   # (0.7, 0.3) general clinical — semantic leaning
    "equal",         # (0.5, 0.5) mixed queries — balanced default
    "bm25_heavy",    # (0.3, 0.7) known drug/cancer + keyword signals
    "bm25_only",     # (0.0, 1.0) exact trial IDs, NCT numbers, DOIs
]

# Maps each strategy to (dense_weight, bm25_weight)
STRATEGY_WEIGHTS: dict[str, tuple[float, float]] = {
    "dense_only":  (1.0, 0.0),
    "dense_heavy": (0.7, 0.3),
    "equal":       (0.5, 0.5),
    "bm25_heavy":  (0.3, 0.7),
    "bm25_only":   (0.0, 1.0),
}


# ─────────────────────────────────────────────────────────────────────
# 3. METADATA FILTER DATACLASS
# ─────────────────────────────────────────────────────────────────────
# Internal container — consumed by retriever.py → build_filter_clause()
# Stays as dataclass: not from external JSON, no Pydantic overhead.
# Produced by FiltersOutput.to_metadata_filter() after validation.
#
# Python 3.13: str | None replaces Optional[str] throughout.

@dataclass
class MetadataFilter:
    """
    Typed SQL filter container. All fields None by default (None = skip).
    None means "don't add this field to the SQL WHERE clause."
    Passed directly into retriever.retrieve() as the filters argument.
    """
    drug:               str | None = None
    cancer_type:        str | None = None
    study_type:         str | None = None
    year_min:           int | None = None
    year_max:           int | None = None
    evidence_level_max: int | None = None
    endpoints:          str | None = None
    pdl1_status:        str | None = None
    funding:            str | None = None


# ─────────────────────────────────────────────────────────────────────
# 4. PYDANTIC MODELS — schema passed to model via with_structured_output
# ─────────────────────────────────────────────────────────────────────
# Two purposes:
#   a) JSON schema passed to model at generation time
#      → model constrained to exact field names + Literal value sets
#   b) field_validators run as safety net after generation
#      → coerce, normalize, reject unknowns → None (safe default)

class FiltersOutput(BaseModel):
    """
    Filter values extracted from the query.
    Literal types constrain model at generation.
    field_validators catch remaining edge cases post-generation.
    Field descriptions read by model as extraction instructions.
    """

    drug: DrugLiteral = Field(
        default=None,
        description=(
            "Generic drug name from valid list only. "
            "Map trade names: keytruda→pembrolizumab, opdivo→nivolumab, "
            "tagrisso→osimertinib, herceptin→trastuzumab. "
            "If comparing two drugs → null. "
            "If negated ('not pembrolizumab') → null. "
            "If drug not in valid list → null."
        )
    )

    cancer_type: CancerTypeLiteral = Field(
        default=None,
        description=(
            "Cancer type from valid list, lowercase hyphenated. "
            "If negated ('not NSCLC', 'other than NSCLC') → null. "
            "If misspelled but context is clearly NSCLC → 'nsclc'. "
            "If ambiguous or comparing multiple cancer types → null."
        )
    )

    study_type: StudyTypeLiteral = Field(
        default=None,
        description=(
            "Research design from valid list. "
            "Set only if explicitly mentioned or strongly implied. "
            "If comparing study types → null."
        )
    )

    year_min: int | None = Field(
        default=None,
        description=(
            "Minimum publication year as integer 1990-2026. "
            "Set to 2020 if 'recent', 'latest', 'current'. "
            "Set to 2018 if 'new data' or 'updated'. "
            "Null if no recency signal."
        )
    )

    year_max: int | None = Field(
        default=None,
        description=(
            "Maximum publication year as integer 1990-2026. "
            "Rarely needed — only if query limits to older studies."
        )
    )

    evidence_level_max: int | None = Field(
        default=None,
        description=(
            "Max evidence level: 1=RCT/meta only, 2=include cohort, "
            "3=all including case reports. "
            "Set to 1 if 'RCT only' or 'randomized trials only'. "
            "Set to 2 if 'excluding case reports'. "
            "Null if no evidence constraint mentioned."
        )
    )

    endpoints: EndpointLiteral = Field(
        default=None,
        description=(
            "Primary outcome from valid abbreviations. "
            "OS=overall survival, PFS=progression-free, "
            "ORR=response rate, AE=adverse events/safety. "
            "Set if query focuses on one specific outcome. "
            "Null if multiple outcomes or unspecified."
        )
    )

    pdl1_status: PDL1Literal = Field(
        default=None,
        description=(
            "PD-L1 biomarker status from valid list. "
            "TPS>=50% for high/positive, TPS<1% for negative/low. "
            "Set only for checkpoint inhibitor queries with clear signal."
        )
    )

    funding: FundingLiteral = Field(
        default=None,
        description=(
            "Funding source: industry, NIH, or non-profit. "
            "Set only if explicitly mentioned. Null in almost all cases."
        )
    )

    # ── Validators (Layer 2 safety net) ──────────────────────────────
    # On 3.13: Literal constraints handle most cases at generation time.
    # Validators catch remaining edge cases: wrong case, aliases,
    # type coercion (model returns "2020" string instead of 2020 int).

    @field_validator("drug", mode="before")
    @classmethod
    def validate_drug(cls, v: object) -> str | None:
        """Normalizes case, resolves aliases, rejects unknowns."""
        if v is None:
            return None
        v_lower = str(v).lower().strip()
        if v_lower in DB_VOCABULARY["drugs"]:
            return v_lower
        # Alias resolution — trade names Literal might not catch
        if v_lower in DB_VOCABULARY["drug_aliases"]:
            return DB_VOCABULARY["drug_aliases"][v_lower]   # type: ignore[return-value]
        return None

    @field_validator("cancer_type", mode="before")
    @classmethod
    def validate_cancer_type(cls, v: object) -> str | None:
        """Rejects values not in DB_VOCABULARY cancer_types."""
        if v is None:
            return None
        v_lower = str(v).lower().strip()
        return v_lower if v_lower in DB_VOCABULARY["cancer_types"] else None

    @field_validator("study_type", mode="before")
    @classmethod
    def validate_study_type(cls, v: object) -> str | None:
        """Case-insensitive match against DB_VOCABULARY study_types."""
        if v is None:
            return None
        v_str = str(v).strip()
        if v_str in DB_VOCABULARY["study_types"]:
            return v_str
        # Try case-insensitive fallback
        return next(
            (valid for valid in DB_VOCABULARY["study_types"]
             if v_str.lower() == valid.lower()),
            None
        )

    @field_validator("endpoints", mode="before")
    @classmethod
    def validate_endpoints(cls, v: object) -> str | None:
        """Must be exact abbreviation from DB_VOCABULARY endpoints."""
        if v is None:
            return None
        v_str = str(v).strip()
        return v_str if v_str in DB_VOCABULARY["endpoints"] else None

    @field_validator("pdl1_status", mode="before")
    @classmethod
    def validate_pdl1(cls, v: object) -> str | None:
        """Must be exact value from DB_VOCABULARY pdl1_values."""
        if v is None:
            return None
        v_str = str(v).strip()
        return v_str if v_str in DB_VOCABULARY["pdl1_values"] else None

    @field_validator("funding", mode="before")
    @classmethod
    def validate_funding(cls, v: object) -> str | None:
        """Three valid values: industry, NIH, non-profit."""
        if v is None:
            return None
        v_lower = str(v).lower().strip()
        if v_lower == "nih":
            return "NIH"
        return v_lower if v_lower in {"industry", "non-profit"} else None

    @field_validator("year_min", "year_max", mode="before")
    @classmethod
    def validate_year(cls, v: object) -> int | None:
        """Coerce strings to int, reject out-of-range 1990-2026."""
        if v is None:
            return None
        try:
            year = int(v)  # type: ignore[arg-type]
            return year if 1990 <= year <= 2026 else None
        except (ValueError, TypeError):
            return None

    @field_validator("evidence_level_max", mode="before")
    @classmethod
    def validate_evidence_level(cls, v: object) -> int | None:
        """Must be 1, 2, or 3 exactly."""
        if v is None:
            return None
        try:
            level = int(v)  # type: ignore[arg-type]
            return level if level in (1, 2, 3) else None
        except (ValueError, TypeError):
            return None

    @model_validator(mode="after")
    def validate_year_range(self) -> "FiltersOutput":
        """Swap silently if year_min > year_max — LLM may reverse them."""
        if (self.year_min is not None
                and self.year_max is not None
                and self.year_min > self.year_max):
            self.year_min, self.year_max = self.year_max, self.year_min
        return self

    def to_metadata_filter(self) -> MetadataFilter:
        """Converts validated Pydantic model → MetadataFilter dataclass."""
        return MetadataFilter(
            drug               = self.drug,
            cancer_type        = self.cancer_type,
            study_type         = self.study_type,
            year_min           = self.year_min,
            year_max           = self.year_max,
            evidence_level_max = self.evidence_level_max,
            endpoints          = self.endpoints,
            pdl1_status        = self.pdl1_status,
            funding            = self.funding,
        )


class RewriterOutput(BaseModel):
    """
    Complete structured output. Schema passed to model via
    llm.with_structured_output(RewriterOutput).
    Model constrained to produce exactly these fields and types.
    """

    rewritten_query: str = Field(
        ...,
        min_length=3,
        description=(
            "Normalized, spell-corrected, vocabulary-aligned query. "
            "Expand abbreviations: pembro→pembrolizumab, OS→overall survival. "
            "Keep clinical terminology. "
            "Example: 'Pembro OS NSC' → 'pembrolizumab overall survival NSCLC'."
        )
    )

    filters: FiltersOutput = Field(
        default_factory=FiltersOutput,
        description="Validated database filter values extracted from the query."
    )

    sub_questions: list[str] = Field(       # list[str] not List[str]
        default_factory=list,
        description=(
            "Decomposed retrieval sub-questions for complex/comparison queries. "
            "Empty list [] if query is already focused. "
            "2-3 sub-questions for comparisons — one per arm. Max 3."
        )
    )

    reasoning: str = Field(
        default="",
        description=(
            "One sentence explaining the key rewriting decision. "
            "Example: 'pembro mapped via alias, NSC corrected to nsclc'."
        )
    )

    confidence: ConfidenceLiteral = Field(
        default="medium",
        description=(
            "high = drug AND cancer_type both clearly identified. "
            "medium = one field ambiguous. "
            "low = query vague or heavily negated/comparative."
        )
    )

    search_strategy: SearchStrategyLiteral = Field(
        default="equal",
        description=(
            "How to weight dense (semantic) vs BM25 (keyword) search in RRF fusion. "
            "Choose based on query type:\n"
            "  dense_only  (1.0, 0.0): Pure concept/mechanism queries. "
            "'How does pembrolizumab work?' — semantics matter, keywords add noise.\n"
            "  dense_heavy (0.7, 0.3): General clinical efficacy/safety questions. "
            "'What is the OS benefit of pembrolizumab in NSCLC?'\n"
            "  equal       (0.5, 0.5): Mixed queries with both concept and "
            "keyword signals. Most queries land here. Default.\n"
            "  bm25_heavy  (0.3, 0.7): Query mentions specific drug name, "
            "cancer type, or known biomarker that must appear in the text.\n"
            "  bm25_only   (0.0, 1.0): Exact identifier lookup. "
            "NCT numbers, trial names (KEYNOTE-189, FLAURA), DOIs, PMIDs. "
            "Semantic search adds noise here — the string must match exactly."
        )
    )

    @field_validator("rewritten_query", mode="before")
    @classmethod
    def clean_query(cls, v: object) -> str:
        if not v or not str(v).strip():
            raise ValueError("rewritten_query cannot be empty")
        return str(v).strip()

    @field_validator("sub_questions", mode="before")
    @classmethod
    def clean_sub_questions(cls, v: object) -> list[str]:
        if not isinstance(v, list):
            return []
        return [str(q).strip() for q in v if q and str(q).strip()][:3]

    @field_validator("search_strategy", mode="before")
    @classmethod
    def validate_search_strategy(cls, v: object) -> str:
        """
        Safety net — if LLM returns unexpected value, default to equal.
        Literal constraint should prevent this, but validator is Layer 2.
        """
        valid = {"dense_only", "dense_heavy", "equal", "bm25_heavy", "bm25_only"}
        if str(v) in valid:
            return str(v)
        return "equal"

    def get_metadata_filter(self) -> MetadataFilter:
        return self.filters.to_metadata_filter()

    def get_rrf_weights(self) -> tuple[float, float]:
        """
        Convenience method — returns (dense_weight, bm25_weight) tuple
        ready to pass into retriever.rrf_fusion().
        Called by retriever.py after rewrite_query() returns.
        """
        return STRATEGY_WEIGHTS.get(self.search_strategy, (0.5, 0.5))


# ─────────────────────────────────────────────────────────────────────
# 5. LANGCHAIN LLM + STRUCTURED OUTPUT CHAIN
# ─────────────────────────────────────────────────────────────────────

# ChatOpenAI — LangChain's OpenAI chat wrapper
# temperature=0: deterministic extraction — same query = same output
# max_tokens=400: direct constructor arg (not model_kwargs — avoids warning)
_llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    max_tokens=400,
)

# with_structured_output(RewriterOutput):
# - Serializes RewriterOutput JSON schema
# - Sends it to the model as response_format at generation time
# - Model constrained to exact field names and Literal value sets
# - Returns RewriterOutput instance directly — no json.loads() needed
# - Every call auto-traced in LangSmith
_structured_llm = _llm.with_structured_output(RewriterOutput)


# ─────────────────────────────────────────────────────────────────────
# 6. PROMPT TEMPLATE
# ─────────────────────────────────────────────────────────────────────
# ChatPromptTemplate — typed, versioned, LangSmith-traced prompt.
# Variables validated at template creation time.
# Python 3.13: f-strings support nested quotes and multi-line
# expressions natively — no escaping gymnastics needed.

_SYSTEM_PROMPT = """You are a clinical literature query rewriter for a pharma RAG system.

Your job: transform the raw query into structured output with:
1. A normalized, spell-corrected, expanded rewritten query
2. Database filter values to constrain retrieval
3. Sub-questions for complex or comparative queries

DATABASE CONTEXT:
Valid drugs: {drugs}
Valid study types (sample): {study_types}
Valid endpoints: {endpoints}
Valid PD-L1 values: {pdl1_values}

CRITICAL RULES:
1. NEGATIONS → null — "not NSCLC", "other than NSCLC" → cancer_type: null
2. COMPARISONS → null for compared field + sub_questions per arm
   "pembro vs nivo" → drug: null, sub_questions: ["pembro survival...", "nivo survival..."]
3. MISSPELLINGS → correct if confident from context, else null
4. ALIASES already normalized in the query shown to you
5. CONFIDENCE: high=drug+cancer both clear, medium=one ambiguous, low=vague
6. SAFETY: when uncertain → null (broad search is safe, wrong filter is bad)
7. SEARCH STRATEGY: choose based on what the query is fundamentally asking
   dense_only  → pure concept/mechanism: "how does pembrolizumab work"
   dense_heavy → general clinical question: "OS benefit of pembrolizumab NSCLC"
   equal       → mixed signals or unsure — DEFAULT for most queries
   bm25_heavy  → specific known terms must appear: drug + cancer + endpoint named
   bm25_only   → exact identifier: NCT number, trial name, PMID, DOI"""

_USER_PROMPT = """Original query: {original_query}
Normalized query (aliases pre-resolved): {normalized_query}

Rewrite and extract filters."""

_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM_PROMPT),
    ("human",  _USER_PROMPT),
])

# The complete LangChain chain: prompt | structured LLM
# | is the pipe operator — output of left feeds input of right
# Entire chain is one traceable unit in LangSmith
_REWRITER_CHAIN = _PROMPT | _structured_llm


# ─────────────────────────────────────────────────────────────────────
# 7. ALIAS PRE-PROCESSING
# ─────────────────────────────────────────────────────────────────────

def _normalize_aliases(query: str) -> str:
    """
    Replaces drug trade names / abbreviations with canonical names
    BEFORE the LLM call.

    Literal constraints prevent wrong output values but don't help the
    model recognise "pembro" as a drug to extract. Pre-processing puts
    "pembrolizumab" in the query text the model reads — extraction
    becomes trivially obvious regardless of schema constraints.

    \b word boundary prevents partial replacements ("opdivot" → safe).
    re.escape() handles special characters in alias names.
    """
    normalized = query
    for alias, canonical in DB_VOCABULARY["drug_aliases"].items():
        pattern    = r'\b' + re.escape(alias) + r'\b'
        normalized = re.sub(pattern, canonical, normalized, flags=re.IGNORECASE)
    return normalized


# ─────────────────────────────────────────────────────────────────────
# 8. SAFE FALLBACK
# ─────────────────────────────────────────────────────────────────────

def _safe_fallback(
    original_query: str,
    reason: str = "",
) -> tuple["RewriterOutput", MetadataFilter]:
    """
    Returns original query + empty filters on any failure.
    Full corpus search — always safe, never wrong results.
    tuple[X, Y] — Python 3.9+ built-in generic, no Tuple import.
    """
    output = RewriterOutput(
        rewritten_query = original_query,
        filters         = FiltersOutput(),
        sub_questions   = [],
        reasoning       = reason or "Fallback — rewriter failed",
        confidence      = "low",
    )
    return output, MetadataFilter()


# ─────────────────────────────────────────────────────────────────────
# 9. CORE REWRITER
# ─────────────────────────────────────────────────────────────────────

async def rewrite_query(
    query: str,
) -> tuple[RewriterOutput, MetadataFilter]:
    """
    Main entry point. One LangChain chain invocation that:
      - Spell-corrects and normalizes query text
      - Extracts DB-valid metadata filters
      - Handles negations, comparisons, abbreviations, misspellings
      - Decomposes complex queries into sub-questions

    Three-layer protection:
      Layer 1: Literal types     → model constrained at generation
      Layer 2: field_validators  → safety net post-generation
      Layer 3: Exception handler → _safe_fallback(), never crashes

    Returns:
      tuple[RewriterOutput, MetadataFilter]
      MetadataFilter ready for retriever.retrieve().
      Never raises.
    """
    normalized = _normalize_aliases(query)

    try:
        output: RewriterOutput = await _REWRITER_CHAIN.ainvoke({
            "original_query":   query,
            "normalized_query": normalized,
            "drugs":            ", ".join(DB_VOCABULARY["drugs"]),   # type: ignore[arg-type]
            "study_types":      ", ".join(DB_VOCABULARY["study_types"]),  # type: ignore[arg-type]
            "endpoints":        ", ".join(DB_VOCABULARY["endpoints"]),    # type: ignore[arg-type]
            "pdl1_values":      ", ".join(DB_VOCABULARY["pdl1_values"]),  # type: ignore[arg-type]
        })

        if output is None:
            return _safe_fallback(query, "Chain returned None")

        # Ensure rewritten_query is non-empty
        if not output.rewritten_query.strip():
            output.rewritten_query = normalized

        output.sub_questions = output.sub_questions[:3]

        return output, output.get_metadata_filter()

    except Exception as e:
        print(f"  QueryRewriter: {type(e).__name__}: {e}")
        return _safe_fallback(query, f"{type(e).__name__}: {e}")


async def rewrite_query_with_subquestions(
    query: str,
) -> tuple[RewriterOutput, MetadataFilter, list[str], tuple[float, float]]:
    """
    Convenience wrapper for retriever.py multi-pass retrieval.
    Returns (RewriterOutput, MetadataFilter, sub_questions, rrf_weights).

    rrf_weights: (dense_weight, bm25_weight) from output.get_rrf_weights()
    Retriever passes these directly into rrf_fusion() — no classify_query()
    needed since the LLM already decided the strategy.
    """
    output, filters = await rewrite_query(query)
    return output, filters, output.sub_questions or [], output.get_rrf_weights()


# ─────────────────────────────────────────────────────────────────────
# 10. QUICK TEST
# ─────────────────────────────────────────────────────────────────────

async def _test() -> None:
    """Quick smoke test — python query_rewriter.py"""
    test_cases = [
        "What is the overall survival benefit of pembrolizumab in NSCLC?",
        "Pembro survival benefit in NSC patients PD-L1 high",
        "pembrolizumab in cancer types other than NSCLC",
        "pembrolizumab vs nivolumab NSCLC survival",
        "Does Keytruda improve OS vs Opdivo in lung cancer?",
        "best cancer treatment",
        "KEYNOTE-189 primary endpoint overall survival results",
        "latest real-world trastuzumab HER2 breast cancer 2023",
        "RCT only osimertinib NSCLC progression free survival",
    ]

    print("QueryRewriter smoke test (Python 3.13 + LangChain)\n" + "=" * 60)

    for query in test_cases:
        print(f"\nInput     : {query}")
        output, filters = await rewrite_query(query)
        print(f"Rewritten : {output.rewritten_query}")
        print(f"Drug      : {filters.drug}")
        print(f"Cancer    : {filters.cancer_type}")
        print(f"Study     : {filters.study_type}")
        print(f"Endpoints : {filters.endpoints}")
        print(f"PD-L1     : {filters.pdl1_status}")
        print(f"Ev. level : {filters.evidence_level_max}")
        print(f"Year min  : {filters.year_min}")
        print(f"Sub-Qs    : {output.sub_questions}")
        print(f"Confidence: {output.confidence}")
        print(f"Strategy  : {output.search_strategy}  -> weights {output.get_rrf_weights()}")
        print(f"Reasoning : {output.reasoning}")


if __name__ == "__main__":
    asyncio.run(_test())