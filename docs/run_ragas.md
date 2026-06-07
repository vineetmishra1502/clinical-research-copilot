# eval/run_ragas.py — RAGAS Evaluation Script

End-to-end evaluation of the Clinical Research Copilot pipeline using [RAGAS](https://docs.ragas.io/) metrics. Runs 42 clinical queries against the live API, scores the results across four metrics, and saves a full per-query breakdown to `eval/results.json`.

---

## How it works

```
golden_dataset.jsonl (42 queries + ground truths)
         │
         ▼
[1] Health check      → verifies API is running at localhost:8000
         │
         ▼
[2] Load dataset      → filters by --difficulty or --sample if provided
         │
         ▼
[3] Collect outputs   → POST /research for each query
         │              stores full_report + exec_summary + contexts + metadata
         │
         ▼
[4] RAGAS evaluation  → Option A: two separate evaluate() calls
         │
         ├── Eval 1: full report   → Context Precision + Faithfulness
         └── Eval 2: exec summary  → Context Recall    + Answer Correctness
         │
         ▼
[5] Results           → console table + eval/results.json
```

---

## Option A — Why two separate evaluation calls?

Each RAGAS metric is designed to measure a specific property. Using one text for all four metrics produces a systematic mismatch because the full report (400–600 words) and ground truth (3–5 sentences) are structurally different lengths.

| Metric | Text passed to RAGAS | Why |
|---|---|---|
| Context Precision | Full report | Measures whether retrieved chunks were relevant — not affected by answer length |
| Faithfulness | Full report | Measures whether the LLM stayed grounded in chunk text — needs the full detailed report, not a condensed summary which draws inferences |
| Context Recall | Executive summary | Measures whether key GT facts appear in contexts — length-matched to GT avoids penalising thoroughness |
| Answer Correctness | Executive summary | Measures factual + semantic match against GT — length mismatch between a 600-word report and a 3-sentence GT artificially deflates this score |

**The faithfulness insight:** The full report writes claims very close to chunk language ("KEYNOTE-024 demonstrated significantly improved OS"). The executive summary generalises ("pembrolizumab is the preferred treatment") — a clinically correct inference that RAGAS faithfulness flags as ungrounded because no chunk states "preferred treatment" explicitly. Evaluating faithfulness on the full report gives the honest grounding signal.

---

## Golden dataset

`eval/golden_dataset.jsonl` — 42 hand-crafted clinical Q&A pairs.

| Property | Value |
|---|---|
| Total queries | 42 |
| Drugs covered | pembrolizumab, nivolumab, osimertinib, trastuzumab, metformin |
| Difficulty levels | easy (13), medium (20), hard (9) |
| Query types | efficacy (26), safety (7), mechanism (4), comparison (3), biomarker (2) |
| Ground truth style | Synthesis-level directional claims — matches what the pipeline retrieves, not what a specific trial paper says |

### Ground truth design principles

Ground truths went through four iterations. The final v4 ground truths follow three rules:

**No negative claims.** "Pembrolizumab and nivolumab have not been compared in a head-to-head trial" cannot be verified by any retrieved chunk — PubMed papers don't contain sentences asserting what studies were never done. Replaced with positive descriptions of each drug's evidence base.

**No over-specific sub-claims.** Claims like "HR 0.63" or "median OS 30.0 months" or "neoadjuvant osimertinib is also being evaluated" require retrieving a specific trial paper's results section. The pipeline retrieves meta-analyses and synthesis evidence — those exact numbers rarely appear. Ground truths describe the directional finding ("pembrolizumab significantly improves OS versus chemotherapy") not the specific trial number.

**Scope-matched to the executive summary.** Ground truths are 3–5 sentences covering the main clinical finding. This matches the structure of the executive summary which is evaluated for recall and correctness.

---

## Evaluation results (final run — 42 queries)

| Metric | Score | Text evaluated | What it measures |
|---|---|---|---|
| Context Precision | 0.920 | Full report | 92% of retrieved chunks were relevant to the query |
| Faithfulness | 0.886 | Full report | LLM stays grounded in retrieved evidence across the detailed report |
| Context Recall | 0.686 | Exec summary | 69% of key ground truth facts present in retrieved contexts |
| Answer Correctness | 0.615 | Exec summary | Factual + semantic match between concise answer and reference |

### By difficulty

| Difficulty | n | Precision | Recall | Faithfulness | Correctness |
|---|---|---|---|---|---|
| Easy | 13 | 0.981 | 0.667 | 0.857 | 0.661 |
| Medium | 20 | 0.945 | 0.679 | 0.897 | 0.601 |
| Hard | 9 | 0.776 | 0.728 | 0.905 | 0.581 |

### By query type

| Type | n | Precision | Recall | Faithfulness | Correctness |
|---|---|---|---|---|---|
| Efficacy | 26 | 0.969 | 0.724 | 0.904 | 0.608 |
| Safety | 7 | 0.917 | 0.559 | 0.786 | 0.667 |
| Mechanism | 4 | 0.703 | 0.771 | 1.000 | 0.542 |
| Comparison | 3 | 0.742 | 0.600 | 0.944 | 0.600 |
| Biomarker | 2 | 1.000 | 0.583 | 0.696 | 0.690 |

---

## Prerequisites

The API must be running before executing the script. RAGAS uses OpenAI as the judge LLM — `OPENAI_API_KEY` must be set in `.env`.

```bash
# Terminal 1 — start the API
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2 — run evaluation
cd E:\vineet\Python\clinical-research-copilot
.venv\Scripts\activate
python eval/run_ragas.py
```

---

## Usage

```bash
# Full 42-query evaluation (~25 min)
python eval/run_ragas.py

# Quick 5-sample test run (~3 min)
python eval/run_ragas.py --sample 5

# Only easy queries
python eval/run_ragas.py --difficulty easy

# Only hard queries
python eval/run_ragas.py --difficulty hard

# Against a remote API
python eval/run_ragas.py --api-url http://192.168.1.50:8000
```

---

## Configuration

All constants are at the top of the file:

| Constant | Default | Purpose |
|---|---|---|
| `API_BASE_URL` | `http://localhost:8000` | API server address — overridden by `--api-url` |
| `DATASET_PATH` | `eval/golden_dataset.jsonl` | Golden dataset location |
| `RESULTS_PATH` | `eval/results.json` | Output file location |
| `REQUEST_TIMEOUT` | `180` | Seconds per `/research` call — pipeline takes ~30s normally |
| `RETRY_ATTEMPTS` | `2` | Retry a failed API call once before skipping |
| `RAGAS_LLM_MODEL` | `gpt-4o-mini` | Judge LLM — fast and accurate enough for evaluation |
| `RAGAS_EMBEDDING_MODEL` | `text-embedding-3-small` | Embeddings for AnswerCorrectness semantic similarity |

---

## Output files

### `eval/results.json`

Full per-sample results with all four metric scores, pipeline metadata, and evaluation strategy notes.

```json
{
  "eval_timestamp": "2026-06-07T14:53:11.831518",
  "n_samples": 42,
  "ragas_model": "gpt-4o-mini",
  "eval_strategy": "option_a_split",
  "eval_note": {
    "context_precision":  "evaluated on full report",
    "faithfulness":       "evaluated on full report",
    "context_recall":     "evaluated on executive summary",
    "answer_correctness": "evaluated on executive summary"
  },
  "overall_scores": {
    "context_precision":  0.920,
    "faithfulness":       0.886,
    "context_recall":     0.686,
    "answer_correctness": 0.615
  },
  "per_sample": [
    {
      "id":                 "pembro_nsclc_os_001",
      "query":              "What is the overall survival benefit...",
      "difficulty":         "easy",
      "query_type":         "efficacy",
      "verdict":            "accepted",
      "n_agents":           1,
      "n_contexts":         5,
      "elapsed_s":          51.55,
      "context_precision":  1.0,
      "faithfulness":       0.8,
      "context_recall":     0.0,
      "answer_correctness": 0.9115
    }
  ]
}
```

---

## Dependencies

```
ragas==0.4.3        # vibrantlabsai fork — required for this version
langchain-openai    # ChatOpenAI + OpenAIEmbeddings
python-dotenv
requests
```

### Known issue — VertexAI import error

`ragas==0.4.3` hard-imports `ChatVertexAI` from `langchain_community` which was removed in `langchain-community>=0.3`. If you see `ModuleNotFoundError: No module named 'langchain_community.chat_models.vertexai'`, apply this fix to `.venv/Lib/site-packages/ragas/llms/base.py`:

```python
# Replace the hard imports at lines 12–13:
try:
    from langchain_community.chat_models.vertexai import ChatVertexAI
except ImportError:
    try:
        from langchain_google_vertexai import ChatVertexAI
    except ImportError:
        ChatVertexAI = None

try:
    from langchain_community.llms import VertexAI
except ImportError:
    try:
        from langchain_google_vertexai import VertexAI
    except ImportError:
        VertexAI = None

# Replace the MULTIPLE_COMPLETION_SUPPORTED list to filter None values:
MULTIPLE_COMPLETION_SUPPORTED = [
    t for t in [OpenAI, ChatOpenAI, AzureOpenAI, AzureChatOpenAI, ChatVertexAI, VertexAI]
    if t is not None
]
```

---

## Code structure

| Function | Purpose |
|---|---|
| `load_dataset()` | Loads and optionally filters `golden_dataset.jsonl` |
| `check_api_health()` | Verifies the API is running before evaluation starts |
| `call_research_api()` | POST `/research` with retry logic |
| `_extract_executive_summary()` | Parses the `# Executive Summary` section from the full report |
| `collect_pipeline_outputs()` | Calls the pipeline for all queries, builds two `SingleTurnSample` lists |
| `run_ragas_evaluation()` | Runs two `evaluate()` calls — one per metric group |
| `print_results()` | Prints formatted score table with per-difficulty and per-type breakdown |
| `save_results()` | Merges both result objects and writes `eval/results.json` |
| `main()` | Entry point — argument parsing and orchestration |

---

## Known limitations

**`trast_tnbc_023` (trastuzumab in TNBC)** — always returns `verdict=insufficient`. The knowledge base has no trastuzumab TNBC evidence because trastuzumab is not indicated in that setting. The pipeline correctly identifies insufficient evidence. All four RAGAS scores are zero for this query.

**`trast_gastric_021` (trastuzumab in gastric cancer)** — slow (~90s) and returns only 1 context. The knowledge base has 35 trastuzumab gastric chunks vs 1,400+ breast cancer chunks. The grader runs multiple retry cycles and accepts only 1 chunk. This is a knowledge base coverage gap, not a pipeline failure.

**Comparison queries** (`pembro_nivo_nsclc_007`, `osi_vs_pembro_018`) — score lowest on precision and recall. The retriever is not optimised for cross-drug evidence and comparison ground truths require claims about both drugs simultaneously.
