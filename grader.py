"""
grader.py — Self-Correcting Retrieval Evaluation Loop
======================================================
Python 3.13 | LangChain throughout

Position in pipeline:
  retriever.py → grader.py → agents.py

What it does:
  Receives 5 retrieved chunks from retriever.py.
  Evaluates each chunk on two dimensions:
    1. Relevancy   — does this chunk answer the query?
    2. Faithfulness — is the content grounded in strong clinical evidence?
  Decides whether the retrieved evidence is good enough to pass to agents,
  or whether to re-query with a refined search term and filters.
  Loops up to MAX_RETRIES times before returning the best available evidence.

Why this component exists — the "agentic" part of Agentic RAG:
  Vanilla RAG retrieves once and passes everything to the LLM regardless
  of quality. The LLM then generates answers grounded in whatever it got,
  even if the retrieval was poor. This produces confident-sounding but
  unfaithful answers.

  Agentic RAG evaluates retrieval quality BEFORE generation and self-corrects
  when quality is insufficient. This is the difference between a system that
  "tries its best with bad data" and one that "keeps searching until it finds
  good data or admits it cannot."

Two evaluation dimensions:
  Relevancy:
    Does this chunk answer the user's actual question?
    A chunk about pembrolizumab safety when the query asks about efficacy
    scores low on relevancy even if it is a high-quality paper.
    Scored 0.0-1.0 by LLM-as-grader.

  Faithfulness:
    Is the content grounded in strong clinical evidence?
    evidence_level=1 (RCT/meta-analysis) + large sample_size = high faithfulness.
    evidence_level=3 (case report, n=1) = low faithfulness for efficacy claims.
    Computed from structured metadata — no LLM needed for this dimension.
    Adjusted by filter_level: "relaxed"/"no_filter" → stricter threshold.

Re-query strategy (LLM-driven):
  Low relevancy  → LLM reads failure reasons + writes a better query
  Low faithfulness → rule-based: tighten evidence_level_max SQL filter
  Both low       → LLM tries one combined fix, then gives up

  Why LLM-driven query refinement instead of rule-based:
    Rule-based: f"{original_query} {drug} {cancer}" → clunky, redundant
    LLM-driven: reads actual failure reasons per chunk → writes targeted
    natural language query that addresses the specific gap identified.
    Filter changes stay rule-based — SQL parameters are not language.

LangChain usage:
  ChatOpenAI + with_structured_output(GradingResult) — relevancy grading.
  ChatOpenAI + with_structured_output(RefineQueryOutput) — query refinement.
  Same pattern as query_rewriter.py throughout.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field, field_validator
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

from retriever import RetrievedChunk, retrieve, MetadataFilter

load_dotenv()

# ─────────────────────────────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────────────────────────────

MAX_RETRIES              = 3      # max re-query attempts before giving up
MIN_CHUNKS_REQUIRED      = 3      # minimum acceptable chunks after grading
RELEVANCY_THRESHOLD      = 0.5    # below this → chunk rejected
FAITHFULNESS_THRESHOLD   = 0.4    # below this → chunk rejected (base)
RELEVANCY_WEIGHT         = 0.6    # combined score weight for relevancy
FAITHFULNESS_WEIGHT      = 0.4    # combined score weight for faithfulness

# Faithfulness threshold adjustment by filter_level
# When retrieval had to relax filters, evidence is less targeted → stricter
FAITHFULNESS_BY_LEVEL: dict[str, float] = {
    "full":      0.4,   # full filters applied — trust retrieved evidence
    "relaxed":   0.5,   # edge filters dropped — slightly stricter
    "minimal":   0.55,  # only drug + cancer → stricter
    "drug_only": 0.6,   # only drug filtered → strict
    "no_filter": 0.65,  # full corpus — very strict faithfulness required
}

# Evidence level → base faithfulness contribution
# RCT/meta = strong evidence → high faithfulness ceiling
# Case report → weak evidence → low faithfulness ceiling
EVIDENCE_FAITHFULNESS: dict[int, float] = {
    1: 1.0,   # RCT, meta-analysis, phase-3-trial
    2: 0.65,  # cohort, real-world, observational
    3: 0.25,  # case report, case series
}


# ─────────────────────────────────────────────────────────────────────
# 2. DATA CLASSES
# ─────────────────────────────────────────────────────────────────────

@dataclass
class GradedChunk:
    """
    A RetrievedChunk with grading scores attached.
    Passed to agents.py — agents use these scores to weight evidence.

    relevancy_score:
      0.0 = completely off-topic
      0.5 = tangentially related
      1.0 = directly answers the query

    faithfulness_score:
      0.0 = case report, n=1, no clinical trial
      0.5 = real-world study, moderate evidence
      1.0 = large RCT or meta-analysis

    combined_score:
      (relevancy × 0.6) + (faithfulness × 0.4)
      Relevancy weighted more — a faithful but irrelevant
      chunk is useless for answering the query.
    """
    chunk:              RetrievedChunk
    relevancy_score:    float
    faithfulness_score: float
    combined_score:     float
    relevancy_reason:   str   # one sentence from LLM explaining relevancy score
    accepted:           bool  # True if both thresholds passed


@dataclass
class GraderOutput:
    """
    Complete output from one grader.grade() call.
    Passed to agents.py as the evidence package.

    verdict:
      "accepted"     → ≥ MIN_CHUNKS_REQUIRED chunks passed both thresholds
      "partial"      → some chunks passed, fewer than MIN_CHUNKS_REQUIRED
      "insufficient" → no chunks passed — evidence not found in knowledge base

    accepted_chunks:
      Chunks that passed both relevancy and faithfulness thresholds.
      Sorted by combined_score descending.

    all_graded_chunks:
      All 5 chunks with scores — even rejected ones.
      Useful for debugging and for agents to see why chunks were rejected.

    attempts:
      How many retrieval + grading cycles ran before returning.
      attempts=1 means first retrieval was good enough.
      attempts=3 means two re-queries were needed.

    refined_query:
      The query used in the final successful retrieval attempt.
      May differ from original if re-query rewrote it.

    failure_reason:
      Populated only when verdict="insufficient".
      Explains why no good evidence was found.
    """
    accepted_chunks:    list[GradedChunk]
    all_graded_chunks:  list[GradedChunk]
    verdict:            Literal["accepted", "partial", "insufficient"]
    attempts:           int
    original_query:     str
    refined_query:      str
    failure_reason:     str = ""


# ─────────────────────────────────────────────────────────────────────
# 3. PYDANTIC MODEL FOR LLM-AS-GRADER OUTPUT
# ─────────────────────────────────────────────────────────────────────

class GradingResult(BaseModel):
    """
    Structured output from the LLM relevancy grader.
    Schema passed to ChatOpenAI.with_structured_output().

    Why LLM grades relevancy but NOT faithfulness:
      Relevancy is semantic — "does this chunk answer the query?" —
      requires understanding the query intent and the chunk content.
      Only an LLM can judge this reliably.

      Faithfulness is structural — evidence_level, sample_size, study_type
      are already stored as metadata in RetrievedChunk.
      A deterministic formula is more reliable than LLM judgment here,
      and cheaper (no API call per chunk).

    Why we grade relevancy per chunk (not all at once):
      Grading all 5 chunks in one prompt risks the LLM averaging its
      judgment. Separate prompts per chunk produce more calibrated scores.
      We batch them in asyncio.gather() so it is still one concurrent call.
    """

    relevancy_score: float = Field(
        ...,
        description=(
            "How well does this chunk answer the query? "
            "0.0 = completely irrelevant, wrong drug or cancer type. "
            "0.3 = tangentially related, mentions topic but does not answer. "
            "0.5 = partially answers — relevant topic but missing key details. "
            "0.7 = mostly answers the query with useful clinical information. "
            "1.0 = directly and completely answers the query. "
            "Be precise — do not default to 0.5."
        )
    )

    relevancy_reason: str = Field(
        ...,
        min_length=10,
        description=(
            "One sentence explaining your relevancy score. "
            "State specifically what the chunk does or does not contain "
            "relative to the query. "
            "Example: 'Chunk reports OS benefit of pembrolizumab in TPS>=50% "
            "NSCLC which directly matches the query.' "
            "Example: 'Chunk discusses nivolumab safety, not pembrolizumab "
            "efficacy as asked.'"
        )
    )

    @field_validator("relevancy_score", mode="before")
    @classmethod
    def clamp_relevancy(cls, v: object) -> float:
        """Clamp to [0.0, 1.0] — LLM might return 1.2 or -0.1."""
        try:
            return max(0.0, min(1.0, float(v)))  # type: ignore[arg-type]
        except (ValueError, TypeError):
            return 0.5   # neutral default on parse failure

    @field_validator("relevancy_reason", mode="before")
    @classmethod
    def clean_reason(cls, v: object) -> str:
        if not v or not str(v).strip():
            return "No reason provided."
        return str(v).strip()


# ─────────────────────────────────────────────────────────────────────
# 4. LANGCHAIN GRADER CHAIN
# ─────────────────────────────────────────────────────────────────────

_grader_llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    max_tokens=150,    # relevancy_score + one sentence reason
)

_structured_grader = _grader_llm.with_structured_output(GradingResult)

_GRADER_SYSTEM = """You are a clinical evidence relevancy grader for a pharma RAG system.
Your job: assess how well a retrieved text chunk answers a clinical query.

Score 0.0-1.0 based on:
- Does the chunk address the correct drug? (pembrolizumab ≠ nivolumab)
- Does the chunk address the correct cancer type? (NSCLC ≠ breast cancer)
- Does the chunk address the correct clinical question? (efficacy ≠ safety)
- Does the chunk contain the specific outcome asked about? (OS ≠ PFS)

Be strict. A chunk about the right drug but wrong endpoint scores 0.4-0.5 at most.
A chunk about the wrong drug scores 0.0-0.2 regardless of quality."""

_GRADER_HUMAN = """Query: {query}

Chunk to evaluate:
{chunk_content}

Chunk metadata:
  Drug: {drug}
  Cancer type: {cancer_type}
  Study type: {study_type}
  Endpoints: {endpoints}
  Year: {year}
  Journal: {journal}

Score this chunk's relevancy to the query."""

_GRADER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _GRADER_SYSTEM),
    ("human",  _GRADER_HUMAN),
])

_GRADER_CHAIN = _GRADER_PROMPT | _structured_grader


# ─────────────────────────────────────────────────────────────────────
# 5. PYDANTIC MODEL + CHAIN FOR LLM-DRIVEN QUERY REFINEMENT
# ─────────────────────────────────────────────────────────────────────

class RefineQueryOutput(BaseModel):
    """
    Structured output from the LLM query refiner.
    Schema passed to ChatOpenAI.with_structured_output().

    The refiner reads:
      - The original query that failed
      - The actual failure reasons from graded chunks
      - Which pattern failed (relevancy / faithfulness / both)

    It returns a refined query string and a boolean flag indicating
    whether the faithfulness filter should also be tightened.

    Why structured output for query refinement:
      Free-form text generation could return anything — a question,
      an explanation, a multi-sentence paragraph. The structured schema
      constrains output to exactly the fields we need, validated by
      Pydantic, with no post-processing required.

    Why the refiner does NOT decide filter changes:
      Filters are SQL parameters — evidence_level_max, year_min.
      These are deterministic decisions based on which pattern failed.
      An LLM adding noise to SQL parameter decisions would be wrong.
      Language generation (refined_query) needs LLM intelligence.
      Parameter tightening (filters) needs deterministic rules.
      Correct tool for each job.
    """

    refined_query: str = Field(
        ...,
        min_length=10,
        description=(
            "A specific, well-formed clinical search query that addresses "
            "the identified gaps. "
            "Rules: "
            "Use full generic drug names (pembrolizumab, not pembro). "
            "Include the specific clinical outcome (overall survival, not OS). "
            "Include cancer type and biomarker if identifiable from context. "
            "Do NOT repeat terms that are already in the original query unnecessarily. "
            "Write a natural, retrievable clinical question. "
            "Example: 'pembrolizumab overall survival benefit in first-line NSCLC "
            "patients with PD-L1 TPS ≥50%' "
            "NOT: 'pembrolizumab treatment pembrolizumab nsclc OS'"
        )
    )

    tighten_evidence_filter: bool = Field(
        default=False,
        description=(
            "Set True if faithfulness was also low — signals that the "
            "filter should exclude case reports (evidence_level_max=2). "
            "Set False if only relevancy was the problem."
        )
    )

    reasoning: str = Field(
        default="",
        description=(
            "One sentence explaining what was wrong with the original "
            "retrieval and what the refined query addresses differently."
        )
    )

    @field_validator("refined_query", mode="before")
    @classmethod
    def clean_refined_query(cls, v: object) -> str:
        if not v or not str(v).strip():
            raise ValueError("refined_query cannot be empty")
        return str(v).strip()


# LLM for query refinement — same model as grader, different chain
# max_tokens=200: refined query + boolean + one sentence reasoning
_refiner_llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    max_tokens=200,
)

_structured_refiner = _refiner_llm.with_structured_output(RefineQueryOutput)

_REFINER_SYSTEM = """You are a clinical search query refinement specialist.
A retrieval attempt failed because the retrieved chunks were not relevant enough.
Your job: write a better search query that will find the specific clinical evidence needed.

You will be shown:
1. The original query that failed
2. The failure reasons for each retrieved chunk
3. The failure pattern (relevancy / faithfulness / both)

Write a refined query that:
- Addresses the specific gap identified in the failure reasons
- Uses precise clinical terminology (full drug names, outcome names)
- Is specific enough to retrieve targeted evidence
- Is natural enough to produce a good embedding and BM25 match"""

_REFINER_HUMAN = """Original query: {original_query}

Failure pattern: {failure_pattern}

Retrieved chunk failure reasons (top 3):
{failure_reasons}

Average relevancy score:    {avg_relevancy:.2f} (threshold: 0.5)
Average faithfulness score: {avg_faithfulness:.2f} (threshold: 0.4)

Write a refined query that would find better clinical evidence."""

_REFINER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _REFINER_SYSTEM),
    ("human",  _REFINER_HUMAN),
])

_REFINER_CHAIN = _REFINER_PROMPT | _structured_refiner


# ─────────────────────────────────────────────────────────────────────
# 6. FAITHFULNESS COMPUTATION (deterministic — no LLM)
# ─────────────────────────────────────────────────────────────────────

def compute_faithfulness(
    chunk: RetrievedChunk,
    filter_level: str,
) -> float:
    """
    Computes faithfulness score deterministically from chunk metadata.
    No LLM call — metadata already tells us the evidence quality.

    Formula:
      base = EVIDENCE_FAITHFULNESS[evidence_level]   (1.0 / 0.65 / 0.25)
      sample_size_boost: large trials boost score slightly
      faithfulness = min(1.0, base + boost)

    Why deterministic not LLM-judged:
      evidence_level, sample_size, study_type are structured metadata
      extracted during ingestion. A deterministic formula produces
      consistent, explainable scores. An LLM reading these same values
      would produce noisier and more expensive results.

    Why filter_level affects threshold (not the score):
      The score reflects the chunk's inherent evidence quality.
      The threshold at which we accept/reject shifts based on how
      permissive the retrieval was — this happens in grade_chunks(),
      not here.

    Sample size boost:
      n ≥ 1000 → +0.1   (large trial — more reliable)
      n ≥ 300  → +0.05  (medium trial)
      n < 300  → +0.0   (small study — no boost)
    """
    base = EVIDENCE_FAITHFULNESS.get(chunk.evidence_level, 0.4)

    # Sample size boost — larger trials are more reliable
    boost = 0.0
    if chunk.sample_size:
        if chunk.sample_size >= 1000:
            boost = 0.1
        elif chunk.sample_size >= 300:
            boost = 0.05

    return min(1.0, base + boost)


# ─────────────────────────────────────────────────────────────────────
# 7. RELEVANCY GRADING (LLM-as-grader)
# ─────────────────────────────────────────────────────────────────────

async def grade_relevancy_single(
    query: str,
    chunk: RetrievedChunk,
) -> GradingResult:
    """
    Grades one chunk's relevancy via LLM. Called concurrently for all chunks.

    Returns GradingResult(relevancy_score, relevancy_reason).
    Falls back to score=0.3, reason="Grading failed" on any error —
    conservative fallback ensures failed grades don't pass thresholds.
    """
    try:
        result: GradingResult = await _GRADER_CHAIN.ainvoke({
            "query":        query,
            "chunk_content": chunk.content[:800],  # first 800 chars
            "drug":         chunk.drug or "unknown",
            "cancer_type":  chunk.cancer_type or "unknown",
            "study_type":   chunk.study_type or "unknown",
            "endpoints":    chunk.endpoints or "not reported",
            "year":         chunk.year or "unknown",
            "journal":      chunk.journal or "unknown",
        })
        return result
    except Exception as e:
        print(f"  Grader: relevancy scoring failed for chunk {chunk.id}: {e}")
        return GradingResult(
            relevancy_score  = 0.3,
            relevancy_reason = f"Grading failed: {type(e).__name__}",
        )


async def grade_all_relevancy(
    query: str,
    chunks: list[RetrievedChunk],
) -> list[GradingResult]:
    """
    Grades all chunks concurrently via asyncio.gather().
    5 chunks → 5 LLM calls fired simultaneously.
    Total time ≈ time of slowest single call (~800ms), not 5 × 800ms.
    """
    tasks = [grade_relevancy_single(query, chunk) for chunk in chunks]
    return await asyncio.gather(*tasks)


# ─────────────────────────────────────────────────────────────────────
# 8. CHUNK GRADING — combine relevancy + faithfulness
# ─────────────────────────────────────────────────────────────────────

async def grade_chunks(
    query:        str,
    chunks:       list[RetrievedChunk],
    filter_level: str,
) -> list[GradedChunk]:
    """
    Full grading pass for all chunks.

    Steps:
      1. Grade relevancy for all chunks concurrently (LLM)
      2. Compute faithfulness for each chunk (deterministic)
      3. Combine into combined_score
      4. Apply thresholds to set accepted=True/False

    faithfulness_threshold adjusted by filter_level:
      "full"      → 0.4  (standard threshold)
      "relaxed"   → 0.5  (slightly stricter — evidence less targeted)
      "no_filter" → 0.65 (strict — full corpus retrieval is noisy)

    Returns list[GradedChunk] sorted by combined_score descending.
    """
    # Step 1: Grade all relevancy scores concurrently
    grading_results = await grade_all_relevancy(query, chunks)

    # Step 2+3+4: Compute faithfulness + combine + threshold
    faith_threshold = FAITHFULNESS_BY_LEVEL.get(filter_level, FAITHFULNESS_THRESHOLD)
    graded: list[GradedChunk] = []

    for chunk, result in zip(chunks, grading_results):
        faithfulness = compute_faithfulness(chunk, filter_level)

        combined = (
            result.relevancy_score * RELEVANCY_WEIGHT +
            faithfulness           * FAITHFULNESS_WEIGHT
        )

        accepted = (
            result.relevancy_score >= RELEVANCY_THRESHOLD and
            faithfulness           >= faith_threshold
        )

        graded.append(GradedChunk(
            chunk              = chunk,
            relevancy_score    = result.relevancy_score,
            faithfulness_score = faithfulness,
            combined_score     = combined,
            relevancy_reason   = result.relevancy_reason,
            accepted           = accepted,
        ))

    # Sort by combined_score descending — best evidence first
    graded.sort(key=lambda g: g.combined_score, reverse=True)
    return graded


# ─────────────────────────────────────────────────────────────────────
# 9. RE-QUERY STRATEGY — LLM-driven query refinement
# ─────────────────────────────────────────────────────────────────────

def _build_refined_filters(
    filters:         MetadataFilter,
    tighten_evidence: bool,
) -> MetadataFilter:
    """
    Builds tightened MetadataFilter when faithfulness was low.
    Rule-based — SQL parameters are deterministic decisions, not language.

    tighten_evidence=True:
      Sets evidence_level_max=2 (excludes case reports, level 3).
      Sets year_min to at least 2018 (recent evidence only).
      Keeps drug, cancer_type, endpoints, pdl1_status unchanged.

    tighten_evidence=False:
      Returns filters unchanged — only the query text was the problem.
    """
    if not tighten_evidence:
        return filters

    return MetadataFilter(
        drug               = filters.drug,
        cancer_type        = filters.cancer_type,
        year_min           = max(filters.year_min or 2015, 2018),
        year_max           = filters.year_max,
        evidence_level_max = 2,          # exclude level 3 (case reports)
        endpoints          = filters.endpoints,
        pdl1_status        = filters.pdl1_status,
        funding            = filters.funding,
    )


async def decide_requery(
    graded_chunks:  list[GradedChunk],
    original_query: str,
    filters:        MetadataFilter,
    attempt:        int,
) -> tuple[str, MetadataFilter] | None:
    """
    LLM-driven re-query decision.
    Returns (refined_query, refined_filters) or None if should give up.

    Two-part decision:
      Part A — Query refinement (LLM):
        Reads actual failure reasons from graded chunks.
        Writes a natural, targeted query that addresses the specific gap.
        Produces better queries than mechanical string concatenation.
        Example old: "pembrolizumab treatment pembrolizumab nsclc OS"
        Example new: "pembrolizumab overall survival in first-line NSCLC"

      Part B — Filter tightening (rule-based):
        If faithfulness was low → tighten evidence_level_max and year_min.
        SQL parameters are deterministic decisions — LLM adds no value here.
        The RefineQueryOutput.tighten_evidence_filter boolean signals this.

    Three failure patterns:
      Relevancy low, faithfulness OK  → LLM rewrites query, filters unchanged
      Faithfulness low, relevancy OK  → query unchanged, filters tightened
      Both low                        → LLM rewrites AND filters tightened
                                        On second occurrence → return None (give up)

    Returns None when:
      attempt >= MAX_RETRIES (hard stop)
      Both low AND attempt >= 2 (data gap — no re-query will help)
      LLM refinement fails and fallback also fails (degrade gracefully)
    """
    if attempt >= MAX_RETRIES:
        return None

    # Compute average scores
    avg_relevancy    = sum(g.relevancy_score    for g in graded_chunks) / len(graded_chunks)
    avg_faithfulness = sum(g.faithfulness_score for g in graded_chunks) / len(graded_chunks)

    relevancy_low    = avg_relevancy    < RELEVANCY_THRESHOLD
    faithfulness_low = avg_faithfulness < FAITHFULNESS_THRESHOLD

    # If both low and this is not the first attempt → data gap, give up
    if relevancy_low and faithfulness_low and attempt >= 2:
        return None

    # Describe the failure pattern for the LLM refiner
    if relevancy_low and faithfulness_low:
        failure_pattern = (
            "Both relevancy AND faithfulness are low. "
            "Chunks are off-topic AND from weak evidence. "
            "Write a much more specific query targeting strong clinical trials."
        )
    elif relevancy_low:
        failure_pattern = (
            "Relevancy is low — chunks are not answering the query. "
            "The retrieved papers are about the right drug/area but miss "
            "the specific clinical question being asked. "
            "Write a more targeted query."
        )
    else:
        failure_pattern = (
            "Faithfulness is low — chunks are relevant but from weak evidence "
            "(case reports, small observational studies). "
            "The query is fine — we need to find stronger clinical trial evidence. "
            "Keep the same clinical question but make it more specific to trials."
        )

    # Build failure reasons from top 3 graded chunks (worst first for context)
    # Sort by relevancy_score ascending to show the biggest failures
    worst_chunks = sorted(graded_chunks, key=lambda g: g.relevancy_score)[:3]
    failure_reasons = "\n".join([
        f"  Chunk {i+1} (relevancy={g.relevancy_score:.2f}): {g.relevancy_reason}"
        for i, g in enumerate(worst_chunks)
    ])

    # LLM generates the refined query
    try:
        refine_output: RefineQueryOutput = await _REFINER_CHAIN.ainvoke({
            "original_query":   original_query,
            "failure_pattern":  failure_pattern,
            "failure_reasons":  failure_reasons,
            "avg_relevancy":    avg_relevancy,
            "avg_faithfulness": avg_faithfulness,
        })

        refined_query = refine_output.refined_query

        # Use LLM's signal + our pattern analysis to decide filter tightening
        # LLM can signal tighten_evidence_filter=True when it detects weak evidence
        # We also enforce it when faithfulness_low is True (our own analysis)
        should_tighten = refine_output.tighten_evidence_filter or faithfulness_low
        refined_filters = _build_refined_filters(filters, should_tighten)

        print(f"  Refiner: '{refined_query[:70]}'")
        print(f"  Refiner reasoning: {refine_output.reasoning}")
        print(f"  Filter tightened: {should_tighten}")

        return refined_query, refined_filters

    except Exception as e:
        print(f"  Refiner LLM failed: {type(e).__name__}: {e}. Using fallback.")

        # Fallback: simple rule-based refinement
        # Better than nothing if LLM is unavailable
        if relevancy_low:
            # Append the most informative metadata from best chunk
            best = graded_chunks[0].chunk
            terms = [t for t in [best.drug, best.cancer_type] if t]
            refined_query = f"{original_query} {' '.join(terms)}".strip()
        else:
            refined_query = original_query

        refined_filters = _build_refined_filters(filters, faithfulness_low)
        return refined_query, refined_filters


# ─────────────────────────────────────────────────────────────────────
# 10. VERDICT COMPUTATION
# ─────────────────────────────────────────────────────────────────────

def compute_verdict(
    accepted_chunks: list[GradedChunk],
) -> Literal["accepted", "partial", "insufficient"]:
    """
    Determines the overall verdict based on accepted chunk count.

    accepted     → ≥ MIN_CHUNKS_REQUIRED chunks passed both thresholds
                   Enough evidence to generate a reliable answer.

    partial      → 1 or 2 chunks accepted (below MIN_CHUNKS_REQUIRED)
                   Some evidence available. Agents should generate a
                   cautious answer and acknowledge limited evidence.

    insufficient → 0 chunks accepted
                   No reliable evidence found. Agents should not
                   generate a clinical answer — return a "not found"
                   response to the user.
    """
    n = len(accepted_chunks)
    if n >= MIN_CHUNKS_REQUIRED:
        return "accepted"
    if n > 0:
        return "partial"
    return "insufficient"


# ─────────────────────────────────────────────────────────────────────
# 11. MASTER GRADE FUNCTION — the self-correcting loop
# ─────────────────────────────────────────────────────────────────────

async def grade(
    query:           str,
    retrieval_result: dict,
    filters:         MetadataFilter | None = None,
) -> GraderOutput:
    """
    Master grading function — the self-correcting loop.

    Receives retrieval_result from retriever.retrieve() and
    iteratively grades, re-queries, and re-grades until either:
      a) Enough high-quality chunks are found (verdict = "accepted")
      b) MAX_RETRIES attempts are exhausted (verdict = "partial"/"insufficient")

    Flow per attempt:
      1. Grade all chunks (LLM relevancy + deterministic faithfulness)
      2. Count accepted chunks
      3. If enough → return GraderOutput (accepted)
      4. If not enough → decide_requery() determines what to change
      5. Call retriever.retrieve() with refined query/filters
      6. Repeat from step 1

    Arguments:
      query:            original user query
      retrieval_result: dict from retriever.retrieve()
      filters:          MetadataFilter used by retriever (for re-query)

    Returns GraderOutput with:
      accepted_chunks, verdict, attempts, refined_query, failure_reason
    """
    original_query   = query
    current_query    = retrieval_result.get("rewritten_query", query)
    current_filters  = filters or retrieval_result.get("filters_applied") or MetadataFilter()
    current_result   = retrieval_result
    best_graded:  list[GradedChunk] = []
    best_accepted: list[GradedChunk] = []

    for attempt in range(1, MAX_RETRIES + 1):
        chunks       = current_result.get("chunks", [])
        filter_level = current_result.get("filter_level", "full")

        if not chunks:
            print(f"  Grader attempt {attempt}: no chunks returned by retriever")
            break

        print(f"  Grader attempt {attempt}: grading {len(chunks)} chunks "
              f"(filter_level={filter_level})")

        # Grade all chunks
        graded = await grade_chunks(current_query, chunks, filter_level)
        accepted = [g for g in graded if g.accepted]

        # Print per-chunk summary for visibility
        for i, g in enumerate(graded, 1):
            status = "✓" if g.accepted else "✗"
            print(f"    Chunk {i} {status}  "
                  f"relevancy={g.relevancy_score:.2f}  "
                  f"faithfulness={g.faithfulness_score:.2f}  "
                  f"combined={g.combined_score:.2f}  "
                  f"| {g.relevancy_reason[:60]}")

        # Track the best result across attempts
        if len(accepted) > len(best_accepted):
            best_accepted = accepted
            best_graded   = graded

        print(f"  Grader attempt {attempt}: "
              f"{len(accepted)}/{len(chunks)} chunks accepted")

        # Enough evidence — return immediately
        if len(accepted) >= MIN_CHUNKS_REQUIRED:
            return GraderOutput(
                accepted_chunks   = accepted,
                all_graded_chunks = graded,
                verdict           = "accepted",
                attempts          = attempt,
                original_query    = original_query,
                refined_query     = current_query,
            )

        # Not enough — decide whether to re-query (async — LLM reads failure reasons)
        requery = await decide_requery(graded, current_query, current_filters, attempt)

        if requery is None:
            print(f"  Grader: no re-query strategy available — stopping at attempt {attempt}")
            break

        refined_query, refined_filters = requery
        print(f"  Grader: re-querying with '{refined_query[:60]}...'")

        # Re-retrieve with refined query and filters
        current_result  = await retrieve(
            refined_query,
            filters      = refined_filters,
            use_rewriter = False,   # already refined — skip rewriter
        )
        current_query   = refined_query
        current_filters = refined_filters

    # Loop ended — return best we found across all attempts
    verdict      = compute_verdict(best_accepted)
    failure_reason = ""

    if verdict == "insufficient":
        failure_reason = (
            f"After {MAX_RETRIES} retrieval attempts, no chunks passed "
            f"relevancy threshold ({RELEVANCY_THRESHOLD}) and faithfulness "
            f"threshold. The knowledge base may not contain strong clinical "
            f"evidence for this query."
        )
    elif verdict == "partial":
        failure_reason = (
            f"Only {len(best_accepted)} chunk(s) passed grading thresholds "
            f"(minimum required: {MIN_CHUNKS_REQUIRED}). "
            f"Answer generated from limited evidence — treat with caution."
        )

    return GraderOutput(
        accepted_chunks   = best_accepted,
        all_graded_chunks = best_graded,
        verdict           = verdict,
        attempts          = MAX_RETRIES,
        original_query    = original_query,
        refined_query     = current_query,
        failure_reason    = failure_reason,
    )


# ─────────────────────────────────────────────────────────────────────
# 12. CONVENIENCE WRAPPER — retrieve + grade in one call
# ─────────────────────────────────────────────────────────────────────

async def retrieve_and_grade(
    query:   str,
    filters: MetadataFilter | None = None,
) -> GraderOutput:
    """
    Convenience function for agents.py — retrieves and grades in one call.

    Agents call this instead of calling retrieve() and grade() separately.
    Ensures query_embedding is passed from retrieval result to grader
    (avoids a second embedding call during re-query).

    Usage in agents.py:
      grader_output = await retrieve_and_grade(
          query="pembrolizumab OS NSCLC TPS>=50%",
          filters=MetadataFilter(drug="pembrolizumab", cancer_type="nsclc"),
      )
      if grader_output.verdict in ("accepted", "partial"):
          evidence = grader_output.accepted_chunks
    """
    result = await retrieve(query, filters=filters)
    return await grade(query, result, filters)


# ─────────────────────────────────────────────────────────────────────
# 13. QUICK TEST
# ─────────────────────────────────────────────────────────────────────

async def _test() -> None:
    """Quick smoke test — python grader.py"""
    print("Grader smoke test\n" + "=" * 55)

    test_cases: list[tuple[str, MetadataFilter | None]] = [
        # Standard case — should accept on first attempt
        (
            "What is the overall survival benefit of pembrolizumab in NSCLC?",
            MetadataFilter(drug="pembrolizumab", cancer_type="nsclc",
                           evidence_level_max=2, year_min=2018),
        ),
        # Vague query — may need re-query
        (
            "checkpoint inhibitor outcomes",
            None,
        ),
        # Comparison query — tests sub-question handling
        (
            "pembrolizumab vs nivolumab NSCLC survival comparison",
            None,
        ),
    ]

    for query, filters in test_cases:
        print(f"\nQuery: {query}")
        output = await retrieve_and_grade(query, filters)

        print(f"\nVerdict      : {output.verdict}")
        print(f"Attempts     : {output.attempts}")
        print(f"Accepted     : {len(output.accepted_chunks)}/{len(output.all_graded_chunks)}")
        print(f"Orig query   : {output.original_query}")
        print(f"Refined query: {output.refined_query}")
        rewritten = output.refined_query != output.original_query
        print(f"Query changed: {rewritten}")
        if output.failure_reason:
            print(f"Failure      : {output.failure_reason}")

        print(f"\nAccepted chunks:")
        for i, g in enumerate(output.accepted_chunks, 1):
            print(f"  {i}. [{g.chunk.study_type}] "
                  f"[{g.chunk.cancer_type}] "
                  f"n={g.chunk.sample_size or 'N/A'}  "
                  f"relevancy={g.relevancy_score:.2f}  "
                  f"faithful={g.faithfulness_score:.2f}  "
                  f"combined={g.combined_score:.2f}")
            print(f"     {g.relevancy_reason}")
        print()


if __name__ == "__main__":
    asyncio.run(_test())