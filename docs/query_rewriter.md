# query_rewriter.py — Documentation

## Position in Pipeline

```
User query (raw, messy, abbreviated)
        ↓
query_rewriter.py   ← YOU ARE HERE
        ↓
retriever.py → grader.py → agents.py
```

---

## Architecture Flow

```
Raw query: "Pembro OS in NSC pts PD-L1 high"
        │
        ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Step 1: _normalize_aliases()          pure Python, zero API cost   │
│                                                                      │
│  "Pembro" → "pembrolizumab"   "Keytruda" → "pembrolizumab"          │
│  Whole-word match only (\b) — "opdivot" stays "opdivot"             │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ normalized query
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Step 2: _REWRITER_CHAIN.ainvoke()     one LangChain chain call     │
│                                                                      │
│  _PROMPT (ChatPromptTemplate)                                        │
│    fills: original_query, normalized_query, drugs, endpoints, etc.  │
│    ↓                                                                 │
│  _structured_llm (ChatOpenAI.with_structured_output(RewriterOutput))│
│    schema sent to model at generation time                           │
│    model constrained to exact field names + Literal value sets       │
│    ↓                                                                 │
│  RewriterOutput instance (typed, validated)                          │
│    rewritten_query  = "pembrolizumab overall survival NSCLC"         │
│    filters.drug     = "pembrolizumab"                                │
│    filters.cancer   = "nsclc"                                        │
│    filters.endpoints= "OS"                                           │
│    filters.pdl1     = "TPS>=50%"                                     │
│    search_strategy  = "dense_heavy"                                  │
│    sub_questions    = []                                             │
│    confidence       = "high"                                         │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
              ┌────────────────┴────────────────┐
              ▼                                 ▼
┌─────────────────────────┐       ┌─────────────────────────────────┐
│  Step 3: Three-Layer    │       │  Step 4: get_rrf_weights()      │
│  Protection             │       │                                  │
│                         │       │  "dense_heavy" → (0.7, 0.3)     │
│  Layer 1: Literal types │       │  Maps strategy → weight tuple   │
│  model cannot return    │       │  passed to retriever RRF fusion  │
│  "pembro" or invented   │       │                                  │
│  values                 │       └─────────────────────────────────┘
│                         │
│  Layer 2: field_validators
│  normalize case, resolve│
│  aliases, reject unknowns│
│  → None (safe default)  │
│                         │
│  Layer 3: try/except    │
│  _safe_fallback() →     │
│  original query + empty │
│  filters, never crashes │
└─────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Output: (RewriterOutput, MetadataFilter, sub_questions, rrf_weights)│
│  Passed to retriever.py                                             │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Step-by-Step Process

### Step 1 — `DB_VOCABULARY` (the reference book)

The single source of truth for every valid value in the database. Serves three jobs:

**Job 1 — Builds Literal type constraints.** These are sent to the LLM as the JSON schema. The model physically cannot return `"pembro"` because it is not in `Literal["pembrolizumab", "nivolumab", ...]`.

**Job 2 — Feeds `_normalize_aliases()`.** Trade names resolved to generic names before the LLM reads the query: `keytruda→pembrolizumab`, `opdivo→nivolumab`, `tagrisso→osimertinib`.

**Job 3 — Injected into the system prompt.** The LLM reads the vocabulary list as context for extraction decisions.

```python
DB_VOCABULARY = {
    "drugs":       ["pembrolizumab", "nivolumab", "osimertinib", "trastuzumab", "metformin"],
    "drug_aliases": {"pembro": "pembrolizumab", "keytruda": "pembrolizumab", ...},
    "cancer_types": ["nsclc", "sclc", "breast-cancer", "melanoma", ...],
    "endpoints":    ["OS", "PFS", "ORR", "DFS", "AE", ...],
    "pdl1_values":  ["TPS<1%", "TPS1-49%", "TPS>=50%", ...],
}
```

---

### Step 2 — Literal types (Layer 1 constraint)

```python
DrugLiteral = Literal["pembrolizumab", "nivolumab", "osimertinib",
                       "trastuzumab", "metformin"] | None
```

`Literal[*DB_VOCABULARY["drugs"]]` — the `*` unpacks the list into separate Literal values. Without `*`, `Literal[("a","b")]` = one tuple value, not two string values. On Python 3.13, `*` unpacking in Literal is natively supported — no version guard needed.

`str | None` — Python 3.13 union syntax replacing `Optional[str]`. Reads naturally as "a valid drug string or nothing."

---

### Step 3 — `MetadataFilter` dataclass

```python
@dataclass
class MetadataFilter:
    drug:               str | None = None
    cancer_type:        str | None = None
    evidence_level_max: int | None = None
    year_min:           int | None = None
    # ...all None by default
```

Internal container — not from LLM JSON, not from API requests. Built by `FiltersOutput.to_metadata_filter()` after Pydantic validation has already run. `None` = "don't add this field to SQL WHERE." Dataclass (not Pydantic) because all values are already validated.

---

### Step 4 — `FiltersOutput` Pydantic model

The form the LLM must fill out. Every field is constrained by a Literal type AND has a `Field(description=...)` the LLM reads as instructions.

```python
class FiltersOutput(BaseModel):
    drug: DrugLiteral = Field(
        default=None,
        description=(
            "If comparing two drugs → null. "
            "If negated ('not pembrolizumab') → null. "
        )
    )
```

The description handles four edge cases the Literal alone cannot:

| Situation | LLM instruction | Result |
|---|---|---|
| "pembro vs nivo" | "If comparing → null" | drug = null |
| "not NSCLC" | "If negated → null" | cancer_type = null |
| "NSC" (misspelling) | "correct if confident" | cancer_type = "nsclc" |
| Unknown drug | "If not in valid list → null" | drug = null |

**field_validators (Layer 2)** run after generation as a safety net:

- `validate_drug()` — normalizes case, resolves remaining aliases, rejects unknowns → None
- `validate_year()` — coerces `"2020"` string → `2020` int, rejects out-of-range
- `validate_year_range()` — model_validator, swaps year_min/year_max if reversed
- Each validator returns None on failure — never a wrong SQL filter

---

### Step 5 — `SearchStrategyLiteral` and `STRATEGY_WEIGHTS`

```python
SearchStrategyLiteral = Literal[
    "dense_only",   # (1.0, 0.0) — pure concept/mechanism
    "dense_heavy",  # (0.7, 0.3) — general clinical
    "equal",        # (0.5, 0.5) — balanced default
    "bm25_heavy",   # (0.3, 0.7) — specific known terms
    "bm25_only",    # (0.0, 1.0) — exact NCT/trial IDs
]

STRATEGY_WEIGHTS = {
    "dense_only":  (1.0, 0.0),
    "bm25_only":   (0.0, 1.0),
    ...
}
```

The LLM picks a search strategy as part of `RewriterOutput`. Five discrete options — not a free float. Free floats imply false precision (the LLM has no basis for `0.743`). Named strategies are validatable and explainable. The weight mapping is deterministic — `STRATEGY_WEIGHTS` converts the strategy string to `(dense_weight, bm25_weight)` for the retriever's RRF fusion.

---

### Step 6 — `RewriterOutput` Pydantic model

The complete structured output schema sent to the model via `with_structured_output()`:

```python
class RewriterOutput(BaseModel):
    rewritten_query: str             # normalized, spell-corrected
    filters:         FiltersOutput   # the form above
    sub_questions:   list[str]       # for comparison queries
    reasoning:       str             # one sentence explanation
    confidence:      Literal["high", "medium", "low"]
    search_strategy: SearchStrategyLiteral
```

`min_length=3` on `rewritten_query` prevents empty string returns. `sub_questions` capped at 3 by validator. `search_strategy` validated against the valid set as Layer 2. `get_rrf_weights()` convenience method returns `(dense_weight, bm25_weight)` ready for the retriever.

---

### Step 7 — LangChain chain

```python
_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=400)
_structured_llm = _llm.with_structured_output(RewriterOutput)

_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM_PROMPT),   # instructions + vocabulary
    ("human",  _USER_PROMPT),     # original + normalized query
])

_REWRITER_CHAIN = _PROMPT | _structured_llm
```

`with_structured_output(RewriterOutput)` serializes the full JSON schema and sends it to the model at generation time. `.ainvoke()` returns a typed `RewriterOutput` instance directly — no `json.loads()`, no `model_validate()`. `|` is the LangChain pipe operator — left side output feeds right side input. The chain is one traceable unit in LangSmith.

`max_tokens=400` as direct constructor argument — not inside `model_kwargs`. Newer LangChain warns when known parameters are nested inside `model_kwargs`.

---

### Step 8 — `_normalize_aliases()`

```python
def _normalize_aliases(query: str) -> str:
    for alias, canonical in DB_VOCABULARY["drug_aliases"].items():
        pattern    = r'\b' + re.escape(alias) + r'\b'
        normalized = re.sub(pattern, canonical, normalized, flags=re.IGNORECASE)
    return normalized
```

Runs **before** the LLM call — zero API cost. `\b` word boundary prevents partial replacements. `re.escape()` handles hyphens and special characters in alias names like `mk-3475`. Alias normalization is pre-processing, not post-validation — it makes the extraction step obvious rather than ambiguous.

---

### Step 9 — `rewrite_query()` and `rewrite_query_with_subquestions()`

`rewrite_query()` — the main entry point. Normalizes aliases, invokes the chain, validates the output, returns `(RewriterOutput, MetadataFilter)`. Wrapped in try/except — `_safe_fallback()` returns the original query with empty filters on any failure. Full corpus search is always safe; wrong filters are not.

`rewrite_query_with_subquestions()` — convenience wrapper for `retriever.py`. Returns the 4-tuple `(RewriterOutput, MetadataFilter, sub_questions, rrf_weights)`. The retriever unpacks all four in one call.

---

## Data contracts

| Output | Type | Consumer |
|---|---|---|
| `rewritten_query` | `str` | `retriever.embed_query()` |
| `MetadataFilter` | `@dataclass` | `retriever.build_filter_clause()` |
| `sub_questions` | `list[str]` | `retriever` multi-pass loop |
| `rrf_weights` | `tuple[float, float]` | `retriever.rrf_fusion()` |

---

## Key decisions

**Why `with_structured_output()` over `json_object` mode:** `json_object` guarantees valid JSON only. The model can return `{"query": "..."}` when we expect `{"rewritten_query": "..."}` — valid JSON, wrong schema. `with_structured_output()` sends the full Pydantic schema as `response_format` — wrong field names are impossible.

**Why pre-process aliases before LLM call:** Literal constraints prevent wrong output values. They do not help the model recognize that "pembro" in the input is a drug signal. Pre-processing puts "pembrolizumab" in the text the model reads — recognition becomes obvious.

**Why dataclass for `MetadataFilter`, Pydantic for `FiltersOutput`:** `FiltersOutput` comes from LLM output (untrusted) — Pydantic validates it. `MetadataFilter` is constructed by our trusted code from already-validated fields — dataclass carries the clean data to the SQL builder.

