# grader.py — Documentation

## Position in Pipeline

```
retriever.py (5 RetrievedChunks)
        ↓
grader.py   ← YOU ARE HERE
        ↓
agents.py (GraderOutput with accepted_chunks + verdict)
```

---

## Architecture Flow

```
grade(query, retrieval_result, filters)
        │
        ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Self-correcting loop — up to MAX_RETRIES (3) attempts              │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  Attempt N                                                   │   │
│  │                                                              │   │
│  │  grade_chunks(query, chunks, filter_level)                   │   │
│  │    │                                                         │   │
│  │    ├── grade_all_relevancy()    asyncio.gather(5 LLM calls)  │   │
│  │    │   GradingResult per chunk: relevancy_score + reason     │   │
│  │    │   LLM judges semantic fit — only LLM can do this        │   │
│  │    │                                                         │   │
│  │    └── compute_faithfulness()   deterministic, no LLM        │   │
│  │        EVIDENCE_FAITHFULNESS[evidence_level] + sample boost  │   │
│  │        formula is consistent and cheap                       │   │
│  │                                                              │   │
│  │  For each chunk:                                             │   │
│  │    combined = relevancy×0.6 + faithfulness×0.4              │   │
│  │    accepted = relevancy≥0.5 AND faithfulness≥threshold       │   │
│  │    (AND not average — both must independently pass)          │   │
│  │                                                              │   │
│  │  accepted_count ≥ MIN_CHUNKS_REQUIRED (3)?                   │   │
│  │    YES → return GraderOutput(verdict="accepted")             │   │
│  │    NO  → decide_requery()                                    │   │
│  │                                                              │   │
│  │  decide_requery() — async, LLM reads failure reasons        │   │
│  │    ┌─────────────────────────────────────────────────────┐  │   │
│  │    │  avg_relevancy < 0.5?  → Pattern 1: relevancy low   │  │   │
│  │    │  avg_faithfulness < 0.4? → Pattern 2: faith low     │  │   │
│  │    │  Both low AND attempt≥2? → Pattern 3: give up       │  │   │
│  │    │                                                      │  │   │
│  │    │  LLM refiner reads:                                  │  │   │
│  │    │    original_query + failure_pattern                  │  │   │
│  │    │    failure_reasons (actual per-chunk explanations)   │  │   │
│  │    │    avg scores                                        │  │   │
│  │    │  → RefineQueryOutput(refined_query, tighten_filter)  │  │   │
│  │    │                                                      │  │   │
│  │    │  _build_refined_filters() → tightened MetadataFilter │  │   │
│  │    │  (rule-based: SQL params are not a language problem)  │  │   │
│  │    └─────────────────────────────────────────────────────┘  │   │
│  │                                                              │   │
│  │  retrieve(refined_query, refined_filters, use_rewriter=False)│   │
│  │  → new retrieval result → loop back to top                  │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  Loop ends when:                                                     │
│    accepted ≥ 3 → verdict="accepted"                                │
│    attempts exhausted → verdict="partial" or "insufficient"          │
└──────────────────────────────┬──────────────────────────────────────┘
                               ▼
                    GraderOutput
                    accepted_chunks    (sorted by combined_score)
                    all_graded_chunks  (including rejected — for debugging)
                    verdict            "accepted" / "partial" / "insufficient"
                    attempts           how many cycles ran
                    refined_query      final query used
                    failure_reason     explains why evidence was limited
```

---

## Step-by-Step Process

### Step 1 — Configuration

```python
MAX_RETRIES            = 3      # maximum re-query attempts
MIN_CHUNKS_REQUIRED    = 3      # need at least 3 good chunks
RELEVANCY_THRESHOLD    = 0.5    # below this → chunk rejected
FAITHFULNESS_THRESHOLD = 0.4    # below this → chunk rejected (base)
RELEVANCY_WEIGHT       = 0.6    # relevancy matters more than faithfulness
FAITHFULNESS_WEIGHT    = 0.4

FAITHFULNESS_BY_LEVEL = {
    "full":      0.4,   # standard threshold
    "relaxed":   0.5,   # evidence less targeted → slightly stricter
    "minimal":   0.55,
    "drug_only": 0.6,
    "no_filter": 0.65,  # full corpus → strict faithfulness required
}

EVIDENCE_FAITHFULNESS = {
    1: 1.0,   # RCT, meta-analysis
    2: 0.65,  # cohort, real-world
    3: 0.25,  # case report, case series
}
```

`FAITHFULNESS_BY_LEVEL` — when retrieval relaxed its filters, retrieved chunks are less targeted. The faithfulness threshold tightens to compensate — less precise retrieval requires stronger inherent evidence quality. `EVIDENCE_FAITHFULNESS` — the clinical evidence hierarchy encoded as numbers. This hierarchy is universal and fixed — a formula is more reliable than LLM judgment here.

---

### Step 2 — Data classes

**`GradedChunk`** — a `RetrievedChunk` with three scores added:
- `relevancy_score` — LLM judgment: 0.0 (off-topic) → 1.0 (directly answers)
- `faithfulness_score` — deterministic formula: 0.0 (case report n=1) → 1.0 (large RCT)
- `combined_score` — `relevancy×0.6 + faithfulness×0.4`
- `accepted` — True if both thresholds independently passed

**`GraderOutput`** — the complete package passed to agents:
- `accepted_chunks` — sorted by combined_score, used for LLM generation
- `all_graded_chunks` — all 5, including rejected ones
- `verdict` — `"accepted"` / `"partial"` / `"insufficient"`
- `attempts` — how many retrieval cycles ran (1 = first pass was good enough)
- `failure_reason` — explains why evidence was limited

Both are dataclasses because they are constructed by trusted internal code from already-validated inputs.

---

### Step 3 — `GradingResult` Pydantic model (LLM relevancy output)

```python
class GradingResult(BaseModel):
    relevancy_score:  float   # 0.0 → 1.0, clamped by validator
    relevancy_reason: str     # one sentence explanation
```

Pydantic because it comes from LLM generation. `clamp_relevancy` validator ensures the model cannot return `1.2` or `-0.1`. Schema sent to model via `with_structured_output(GradingResult)` — model constrained to return exactly these two fields.

**Why LLM grades relevancy but not faithfulness:**

| Dimension | Method | Why |
|---|---|---|
| Relevancy | LLM | Semantic judgment — "OS benefit" = "survival improvement." Only LLM understands synonyms and clinical context. |
| Faithfulness | Formula | Structural judgment — `evidence_level=1` is an integer. Arithmetic on integers is more reliable and cheaper than LLM judgment. |

**Why grade per chunk, not all 5 at once:** Grading all chunks in one prompt risks the LLM producing averaged or relative scores ("chunk 1 is better than chunk 2"). Separate prompts produce calibrated absolute scores. `asyncio.gather(5 calls)` runs them concurrently — total time ≈ slowest single call (~800ms), not 5 × 800ms.

---

### Step 4 — LangChain grader chain

```python
_grader_llm     = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=150)
_GRADER_CHAIN   = _GRADER_PROMPT | _structured_grader
```

`max_tokens=150` — output is a float and one sentence. Capping prevents essay responses. Same LangChain pattern as `query_rewriter.py` — `with_structured_output()`, `ChatPromptTemplate`, pipe operator chain.

---

### Step 5 — `RefineQueryOutput` Pydantic model (LLM refiner output)

```python
class RefineQueryOutput(BaseModel):
    refined_query:           str    # natural language improved query
    tighten_evidence_filter: bool   # should we tighten evidence_level_max?
    reasoning:               str    # one sentence explanation
```

**Why separate Pydantic model for refinement:** Free-form text generation would return anything — a question, an explanation, multiple sentences. Structured schema constrains output to exactly the fields needed.

**Why the refiner does NOT decide filter changes:** Filters are SQL parameters — `evidence_level_max`, `year_min`. These are deterministic decisions based on which pattern failed. LLM provides natural language refinement. Rules provide SQL parameter changes. Correct tool for each job.

---

### Step 6 — `compute_faithfulness()` — deterministic

```python
base  = EVIDENCE_FAITHFULNESS.get(chunk.evidence_level, 0.4)

boost = 0.0
if chunk.sample_size:
    if chunk.sample_size >= 1000: boost = 0.1
    elif chunk.sample_size >= 300: boost = 0.05

return min(1.0, base + boost)
```

The clinical evidence hierarchy as code:

```
Level 1 (RCT, meta-analysis): base=1.0 + up to 0.1 = max 1.0
Level 2 (cohort, real-world):  base=0.65 + up to 0.1 = max 0.75
Level 3 (case report):         base=0.25 + up to 0.1 = max 0.35
```

A case report with n=5,000 gets `0.35` — still below the faithfulness threshold for most filter levels. Study design dominates over sample size. A large case series is still not equivalent to an RCT.

---

### Step 7 — Threshold logic

```python
accepted = (
    result.relevancy_score >= RELEVANCY_THRESHOLD    # 0.5
    and
    faithfulness >= faith_threshold                   # 0.4 to 0.65
)
```

**Why `AND` not average:** Both thresholds must independently pass.

```
Chunk A: relevancy=1.0, faithfulness=0.1 (case report n=1)
  Average: (1.0×0.6) + (0.1×0.4) = 0.64 → would PASS if averaged
  AND check: faithfulness 0.1 < 0.4 → REJECTED ✓

Chunk B: relevancy=0.3, faithfulness=1.0 (large RCT, wrong topic)
  Average: (0.3×0.6) + (1.0×0.4) = 0.58 → would PASS if averaged
  AND check: relevancy 0.3 < 0.5 → REJECTED ✓
```

Averaging hides the problem. A perfect paper about the wrong drug should never pass. An irrelevant case report should never pass regardless of how perfect it is.

---

### Step 8 — `decide_requery()` — LLM-driven

**Why LLM-driven instead of rule-based:**

Rule-based (old approach):
```python
refined_query = f"{original_query} {drug} {cancer}"
# → "pembrolizumab treatment pembrolizumab nsclc OS"
# Redundant, clunky, won't embed well
```

LLM-driven (current approach):
```python
# LLM reads failure_reasons: "Chunk covers dosing, not efficacy"
# LLM writes: "pembrolizumab overall survival benefit first-line NSCLC"
# Natural, specific, embeds well
```

**Three failure patterns:**

| Pattern | avg_relevancy | avg_faithfulness | LLM action | Filter action |
|---|---|---|---|---|
| Pattern 1 | LOW | OK | Rewrites query more specifically | Unchanged |
| Pattern 2 | OK | LOW | Query unchanged (topic is fine) | Tightens evidence_level_max=2, year_min=2018 |
| Pattern 3 first | LOW | LOW | Rewrites + signals tighten | Both changes |
| Pattern 3 second | LOW | LOW | Returns None | Stop loop |

**`_build_refined_filters()` — rule-based filter tightening:**
```python
def _build_refined_filters(filters, tighten_evidence: bool) -> MetadataFilter:
    if not tighten_evidence:
        return filters   # query was the problem, not evidence quality
    return MetadataFilter(
        drug               = filters.drug,
        cancer_type        = filters.cancer_type,
        year_min           = max(filters.year_min or 2015, 2018),
        evidence_level_max = 2,    # exclude case reports
        ...
    )
```

**The hybrid approach:** Query text refinement is a language problem → LLM. SQL parameter tightening is a deterministic decision → rules. Same principle as relevancy (LLM) vs faithfulness (formula).

**`use_rewriter=False` in the re-query call:**
```python
current_result = await retrieve(
    refined_query,
    filters      = refined_filters,
    use_rewriter = False,   # do not re-normalize the already-refined query
)
```

The refined query is the product of deliberate reasoning. Running it through `query_rewriter.py` again would re-normalize it, potentially losing the specific terms the LLM just carefully added.

---

### Step 9 — Verdict computation

```python
def compute_verdict(accepted_chunks):
    n = len(accepted_chunks)
    if n >= MIN_CHUNKS_REQUIRED:  return "accepted"
    if n > 0:                     return "partial"
    return "insufficient"
```

| Verdict | Meaning | Agent behaviour |
|---|---|---|
| `accepted` | ≥ 3 chunks passed both thresholds | Full answer with citations |
| `partial` | 1–2 chunks passed | Hedged answer: "limited evidence suggests..." |
| `insufficient` | 0 chunks passed | "No reliable evidence found in knowledge base" |

The verdict travels through the system: `GraderOutput.verdict` → `_format_verdict_note()` → injected into agent system prompt → agent writes hedged or absent content → synthesizer reflects it in the final report.

---

### Step 10 — `retrieve_and_grade()` convenience wrapper

```python
async def retrieve_and_grade(query, filters=None) -> GraderOutput:
    result = await retrieve(query, filters=filters)
    return await grade(query, result, filters)
```

Single function call for agents. Agents call this — they do not need to know retrieval and grading are separate steps.

---

## Data contracts

| Input | Source | Purpose |
|---|---|---|
| `retrieval_result` | `retriever.retrieve()` | chunks + filter_level |
| `filters` | `retriever` / `agents` | for re-query filter changes |
| `query` | original user query | relevancy grading context |

| Output | Type | Consumer |
|---|---|---|
| `accepted_chunks` | `list[GradedChunk]` | agent prompt (evidence) |
| `verdict` | `Literal[...]` | agent prompt (warning injection) |
| `failure_reason` | `str` | agent `verdict_note` field |
| `attempts` | `int` | synthesizer evidence_summary |

---

## Key decisions

**Why `best_accepted` tracked across attempts:** If attempt 2 is worse than attempt 1 (the re-query made things worse), we still return the best result from any attempt. Not the last attempt.

**Why faithfulness is deterministic:** The evidence level hierarchy (RCT > cohort > case report) is universal, fixed, and objective. An LLM producing noisier scores on the same metadata values would be strictly worse than arithmetic on those same values.

