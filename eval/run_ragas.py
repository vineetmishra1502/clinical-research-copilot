"""
eval/run_ragas.py — RAGAS Evaluation Script
=============================================
Evaluates the Clinical Research Copilot end-to-end using RAGAS metrics.

How it works:
  For each of the 42 queries in golden_dataset.jsonl:
    1. POST /research → gets the synthesized report + the exact chunks
       each agent actually used (retrieved_contexts from api/main.py)
    2. Assembles SingleTurnSample with:
         user_input         = the query
         retrieved_contexts = chunks the LLM actually read
         response           = the synthesized markdown report
         reference          = hand-written ground truth from golden dataset
    3. After all 42 queries, calls ragas.evaluate() with 4 metrics

Metrics scored:
  context_precision  — of retrieved chunks, how many were relevant?
  context_recall     — did retrieved chunks contain key facts from ground truth?
  faithfulness       — does the report stick to retrieved chunks (no hallucination)?
  answer_correctness — how factually close is the report to the ground truth?

Output:
  Console:              scores per metric + per query_type breakdown
  eval/results.json:    full scores + per-sample details for README

Usage:
  # Start your API first:
  uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

  # Then in a separate terminal:
  cd E:\\vineet\\Python\\agentic-rag-pharma
  .venv\\Scripts\\activate
  python eval/run_ragas.py

  # Optional: run only easy queries to test quickly
  python eval/run_ragas.py --difficulty easy

  # Optional: run a small sample first
  python eval/run_ragas.py --sample 5

Dependencies (add to requirements.txt):
  ragas>=0.2.0
  langchain-openai
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from datetime import datetime

import requests
from dotenv import load_dotenv

# ── Load env so OPENAI_API_KEY is available for RAGAS judge ──────────
PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / ".env")


# ── RAGAS imports ─────────────────────────────────────────────────────
from ragas import evaluate, EvaluationDataset
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import (
    LLMContextPrecisionWithReference,   # context_precision
    LLMContextRecall,                   # context_recall
    Faithfulness,                       # faithfulness
    AnswerCorrectness,                  # answer_correctness
)
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


# ─────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────

API_BASE_URL    = "http://localhost:8000"
DATASET_PATH    = PROJECT_ROOT / "eval" / "golden_dataset.jsonl"
RESULTS_PATH    = PROJECT_ROOT / "eval" / "results.json"
REQUEST_TIMEOUT = 180          # seconds per /research call — pipeline can take ~30s
RETRY_ATTEMPTS  = 2            # retry a failed API call once before skipping
RETRY_DELAY     = 5            # seconds between retries

# RAGAS uses gpt-4o-mini as the LLM judge — cheap, fast, accurate enough
# AnswerCorrectness also needs embeddings to compute semantic similarity
RAGAS_LLM_MODEL        = "gpt-4o-mini"
RAGAS_EMBEDDING_MODEL  = "text-embedding-3-small"   # cheaper than large for judging



# ─────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────

def load_dataset(path: Path, difficulty: str | None = None, sample: int | None = None) -> list[dict]:
    """Load golden_dataset.jsonl, optionally filtered by difficulty or sample size."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if difficulty:
        records = [r for r in records if r["difficulty"] == difficulty]
        print(f"  Filtered to {difficulty} queries: {len(records)} records")

    if sample and sample < len(records):
        records = records[:sample]
        print(f"  Sampled: {len(records)} records")

    return records


def call_research_api(query: str) -> dict | None:
    """
    POST /research and return the JSON response.

    The response now includes `retrieved_contexts` — the accepted chunk
    content strings from all specialist agents. This is the critical
    addition to api/main.py that makes RAGAS evaluation correct.

    Returns None on failure (after retries).
    """
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            resp = requests.post(
                f"{API_BASE_URL}/research",
                json={"query": query},
                timeout=REQUEST_TIMEOUT,
            )
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 504:
                print(f"    Timeout on attempt {attempt}: {query[:50]}...")
            else:
                print(f"    API error {resp.status_code} on attempt {attempt}: {resp.text[:100]}")
        except requests.Timeout:
            print(f"    Request timeout on attempt {attempt} (>{REQUEST_TIMEOUT}s)")
        except requests.ConnectionError:
            print(f"    Connection error — is the API server running at {API_BASE_URL}?")
            sys.exit(1)
        except Exception as e:
            print(f"    Unexpected error on attempt {attempt}: {type(e).__name__}: {e}")

        if attempt < RETRY_ATTEMPTS:
            print(f"    Retrying in {RETRY_DELAY}s...")
            time.sleep(RETRY_DELAY)

    return None


def check_api_health() -> bool:
    """Verify the API is running and healthy before starting evaluation."""
    try:
        resp = requests.get(f"{API_BASE_URL}/health", timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            status = data.get("status", "unknown")
            print(f"  API status: {status}")
            print(f"  Database:   {data.get('database', 'unknown')}")
            keys = data.get("api_keys", {})
            for k, v in keys.items():
                print(f"  {k:12s}: {'OK' if v else 'MISSING'}")
            return status in ("healthy", "degraded")
        else:
            print(f"  Health check failed: HTTP {resp.status_code}")
            return False
    except requests.ConnectionError:
        print(f"  Cannot connect to {API_BASE_URL}")
        print("  Start with: uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload")
        return False


# ─────────────────────────────────────────────────────────────────────
# MAIN EVALUATION LOOP
# ─────────────────────────────────────────────────────────────────────

def _extract_executive_summary(report: str) -> str:
    """
    Extracts the Executive Summary section from the synthesized report.

    Every report generated by the pipeline starts with:
        # Executive Summary
        <3-4 sentences>
        # Next Section...

    We evaluate only this summary against the ground truth because:
    - Ground truths are 3-5 sentences describing the clinical answer
    - Full reports are 400-600 words covering multiple sections
    - RAGAS answer_correctness penalises length divergence even when
      facts are correct — matching scope gives a fairer measurement

    Falls back to the first 500 chars of the full report if no
    executive summary header is found.
    """
    lines = report.strip().split("\n")
    summary_lines = []
    in_summary = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if in_summary and summary_lines:
                continue
            continue
        # Detect start of executive summary section
        if "executive summary" in stripped.lower() and stripped.startswith("#"):
            in_summary = True
            continue
        # Detect start of next section (stop collecting)
        if in_summary and stripped.startswith("#"):
            break
        if in_summary and stripped:
            summary_lines.append(stripped)

    if summary_lines:
        return " ".join(summary_lines)

    # Fallback: first 500 chars of full report
    return report[:500]



def collect_pipeline_outputs(records: list[dict]) -> tuple[list[SingleTurnSample], list[SingleTurnSample], list[dict]]:
    """
    Call POST /research for each query in the golden dataset and build
    TWO lists of SingleTurnSample objects — one per metric group.

    Option A — right text for each metric:
      samples_full    uses full_report   → Context Precision + Faithfulness
      samples_summary uses exec_summary  → Context Recall + Answer Correctness

    Why this split:
      Precision/Faithfulness measure retrieval grounding — the full report
      stays verbatim-close to chunk language, giving accurate grounding signal.

      Recall/Correctness measure answer quality — the executive summary
      (3-4 sentences) is length-matched to the ground truth (3-5 sentences),
      giving a fair factual accuracy signal without length-mismatch penalty.

    Also returns a list of metadata dicts for the per-query breakdown report.
    """
    samples_full:    list[SingleTurnSample] = []   # full report  → precision + faithfulness
    samples_summary: list[SingleTurnSample] = []   # exec summary → recall + correctness
    meta:            list[dict]             = []
    skipped = 0

    total = len(records)
    print(f"\n  Running {total} queries against {API_BASE_URL}/research")
    print(f"  Estimated time: {total * 25 // 60} min {total * 25 % 60} sec\n")

    for i, record in enumerate(records, 1):
        query        = record["query"]
        ground_truth = record["ground_truth"]
        query_id     = record["id"]
        difficulty   = record["difficulty"]
        query_type   = record["query_type"]

        print(f"  [{i:2d}/{total}] {query_id}")
        print(f"         Query: {query[:70]}...")

        start    = time.perf_counter()
        response = call_research_api(query)
        elapsed  = time.perf_counter() - start

        if response is None:
            print(f"         SKIPPED — API call failed after {RETRY_ATTEMPTS} attempts\n")
            skipped += 1
            continue

        # The key data we need from the response
        full_report = response.get("report", "")
        contexts    = response.get("retrieved_contexts", [])
        verdict     = response.get("overall_verdict", "unknown")
        n_agents    = len(response.get("sections", []))

        if not full_report:
            print(f"         SKIPPED — empty report in response\n")
            skipped += 1
            continue

        # Extract executive summary for RAGAS evaluation.
        # Reason: full reports are 400-600 words; ground truths are 3-5
        # sentences. RAGAS answer_correctness penalises length mismatch
        # even when facts are correct. The executive summary (3-4 sentences)
        # is structurally comparable to the ground truth, giving a fairer
        # factual accuracy measure.
        report = _extract_executive_summary(full_report)

        if not contexts:
            # This means retrieved_contexts wasn't in the response — likely
            # running against the old api/main.py without the modification.
            print(f"         WARNING — no retrieved_contexts in response.")
            print(f"         Check that api/main.py includes the retrieved_contexts field.")
            print(f"         Continuing with empty contexts (faithfulness will be 0).\n")

        print(f"         Verdict: {verdict} | Agents: {n_agents} | Contexts: {len(contexts)} | {elapsed:.1f}s\n")

        # Build TWO RAGAS samples — Option A: right text for each metric
        # Full report → grounding metrics (precision + faithfulness)
        sample_full = SingleTurnSample(
            user_input         = query,
            retrieved_contexts = contexts,
            response           = full_report,
            reference          = ground_truth,
        )
        samples_full.append(sample_full)

        # Executive summary → quality metrics (recall + correctness)
        sample_summary = SingleTurnSample(
            user_input         = query,
            retrieved_contexts = contexts,
            response           = report,          # report = exec_summary
            reference          = ground_truth,
        )
        samples_summary.append(sample_summary)

        # Store metadata for post-eval breakdown
        meta.append({
            "id":           query_id,
            "query":        query,
            "difficulty":   difficulty,
            "query_type":   query_type,
            "verdict":      verdict,
            "n_agents":     n_agents,
            "n_contexts":   len(contexts),
            "elapsed_s":    round(elapsed, 2),
        })

    print(f"\n  Collected {len(samples_full)} samples ({skipped} skipped)")
    return samples_full, samples_summary, meta


# ─────────────────────────────────────────────────────────────────────
# RAGAS EVALUATION
# ─────────────────────────────────────────────────────────────────────

def run_ragas_evaluation(
    samples_full:    list[SingleTurnSample],
    samples_summary: list[SingleTurnSample],
) -> tuple:
    """
    Option A: two separate evaluate() calls, each with the right text.

    Call 1 — grounding metrics on full report:
      LLMContextPrecisionWithReference  uses: retrieved_contexts, reference
      Faithfulness                      uses: response (full_report), retrieved_contexts

    Call 2 — quality metrics on executive summary:
      LLMContextRecall                  uses: retrieved_contexts, reference
      AnswerCorrectness                 uses: response (exec_summary), reference

    Returns (result_grounding, result_quality) — two RAGAS result objects.
    Both are merged in save_results() into a single per-sample output.
    """
    print("\n  Setting up RAGAS judge LLM and embeddings...")

    judge_llm = LangchainLLMWrapper(
        ChatOpenAI(
            model       = RAGAS_LLM_MODEL,
            temperature = 0,
        )
    )

    judge_embeddings_lc = OpenAIEmbeddings(model=RAGAS_EMBEDDING_MODEL)
    from ragas.embeddings import LangchainEmbeddingsWrapper
    judge_embeddings = LangchainEmbeddingsWrapper(judge_embeddings_lc)

    n = len(samples_full)

    # ── Eval 1: grounding metrics on full report ──────────────────────
    print(f"\n  [Eval 1/2] Precision + Faithfulness on full report ({n} samples)...")
    print(f"  Judge model: {RAGAS_LLM_MODEL}")
    print(f"  ~{n * 3} LLM judge calls...\n")

    start = time.perf_counter()
    result_grounding = evaluate(
        dataset          = EvaluationDataset(samples=samples_full),
        metrics          = [
            LLMContextPrecisionWithReference(llm=judge_llm),
            Faithfulness(llm=judge_llm),
        ],
        show_progress    = True,
        raise_exceptions = False,
    )
    print(f"\n  Eval 1 complete in {time.perf_counter()-start:.1f}s")

    # ── Eval 2: quality metrics on executive summary ──────────────────
    print(f"\n  [Eval 2/2] Recall + Answer Correctness on executive summary ({n} samples)...")
    print(f"  ~{n * 3} LLM judge calls...\n")

    start = time.perf_counter()
    result_quality = evaluate(
        dataset          = EvaluationDataset(samples=samples_summary),
        metrics          = [
            LLMContextRecall(llm=judge_llm),
            AnswerCorrectness(llm=judge_llm, embeddings=judge_embeddings),
        ],
        show_progress    = True,
        raise_exceptions = False,
    )
    print(f"\n  Eval 2 complete in {time.perf_counter()-start:.1f}s")

    return result_grounding, result_quality


# ─────────────────────────────────────────────────────────────────────
# RESULTS REPORTING
# ─────────────────────────────────────────────────────────────────────

def print_results(result_grounding, result_quality, meta: list[dict]) -> None:
    """Print a formatted results table to console — Option A split."""
    import pandas as pd

    df_g     = result_grounding.to_pandas()
    df_q     = result_quality.to_pandas()
    scores_g = df_g.mean(numeric_only=True).to_dict()
    scores_q = df_q.mean(numeric_only=True).to_dict()

    prec  = scores_g.get("llm_context_precision_with_reference", float("nan"))
    faith = scores_g.get("faithfulness",                          float("nan"))
    rec   = scores_q.get("context_recall",                        float("nan"))
    corr  = scores_q.get("answer_correctness",                    float("nan"))

    print("\n" + "=" * 70)
    print("  RAGAS EVALUATION RESULTS — Clinical Research Copilot  (Option A)")
    print("=" * 70)
    print(f"\n  Overall scores ({len(meta)} queries evaluated):\n")
    print(f"  {'Metric':<24} {'Score':>6}  {'Text evaluated':<18}  Description")
    print(f"  {'-'*24} {'-'*6}  {'-'*18}  {'-'*38}")

    rows = [
        ("Context Precision",  prec,  "full report",  "Of retrieved chunks, how many were relevant?"),
        ("Faithfulness",       faith, "full report",  "LLM stays grounded in retrieved evidence"),
        ("Context Recall",     rec,   "exec summary", "Did contexts contain key ground truth facts?"),
        ("Answer Correctness", corr,  "exec summary", "How close is the answer to ground truth?"),
    ]
    for name, score, text, desc in rows:
        bar = "\u2588" * int(score * 20) + "\u2591" * (20 - int(score * 20))
        print(f"  {name:<24} {score:>6.3f}  [{bar}]")
        print(f"  {'':24} {'':>6}  text={text:<16}  {desc}")
        print()

    # Merge for breakdown tables
    n = min(len(df_g), len(df_q), len(meta))
    merged = pd.DataFrame({
        "llm_context_precision_with_reference": df_g["llm_context_precision_with_reference"].values[:n],
        "faithfulness":        df_g["faithfulness"].values[:n],
        "context_recall":      df_q["context_recall"].values[:n],
        "answer_correctness":  df_q["answer_correctness"].values[:n],
        "difficulty":          [m["difficulty"]  for m in meta[:n]],
        "query_type":          [m["query_type"]  for m in meta[:n]],
    })

    print("  By difficulty:\n")
    for diff in ["easy", "medium", "hard"]:
        sub = merged[merged["difficulty"] == diff]
        if len(sub) == 0:
            continue
        print(f"  {diff.upper()} ({len(sub)} queries):")
        print(f"    precision={sub['llm_context_precision_with_reference'].mean():.3f}  "
              f"recall={sub['context_recall'].mean():.3f}  "
              f"faithfulness={sub['faithfulness'].mean():.3f}  "
              f"correctness={sub['answer_correctness'].mean():.3f}")

    print("\n  By query type:\n")
    for qt in sorted(merged["query_type"].unique()):
        sub = merged[merged["query_type"] == qt]
        print(f"  {qt.upper()} ({len(sub)} queries):")
        print(f"    precision={sub['llm_context_precision_with_reference'].mean():.3f}  "
              f"recall={sub['context_recall'].mean():.3f}  "
              f"faithfulness={sub['faithfulness'].mean():.3f}  "
              f"correctness={sub['answer_correctness'].mean():.3f}")

    print("\n" + "=" * 70)


def save_results(result_grounding, result_quality, meta: list[dict], output_path: Path) -> None:
    """
    Save full results to eval/results.json — Option A merged output.
    Combines grounding metrics (from full report) and quality metrics
    (from executive summary) into a single per-sample JSON output.
    This file is what you reference in your README performance table.
    """
    import pandas as pd
    df_g   = result_grounding.to_pandas()
    df_q   = result_quality.to_pandas()
    n      = min(len(df_g), len(df_q), len(meta))
    scores_g = df_g.mean(numeric_only=True).to_dict()
    scores_q = df_q.mean(numeric_only=True).to_dict()
    scores = {**scores_g, **scores_q}

    # Per-sample detail — merge grounding + quality rows by position
    per_sample = []
    for i in range(n):
        row_g = df_g.iloc[i]
        row_q = df_q.iloc[i]
        m = meta[i] if i < len(meta) else {}
        per_sample.append({
            "id":                   m.get("id", f"sample_{i}"),
            "query":                m.get("query", ""),
            "difficulty":           m.get("difficulty", ""),
            "query_type":           m.get("query_type", ""),
            "verdict":              m.get("verdict", ""),
            "n_agents":             m.get("n_agents", 0),
            "n_contexts":           m.get("n_contexts", 0),
            "elapsed_s":            m.get("elapsed_s", 0),
            "context_precision":    round(row_g.get("llm_context_precision_with_reference", 0), 4),
            "faithfulness":         round(row_g.get("faithfulness", 0), 4),
            "context_recall":       round(row_q.get("context_recall", 0), 4),
            "answer_correctness":   round(row_q.get("answer_correctness", 0), 4),
        })

    output = {
        "eval_timestamp":   datetime.now().isoformat(),
        "n_samples":        len(meta),
        "ragas_model":      RAGAS_LLM_MODEL,
        "eval_strategy": "option_a_split",
        "eval_note": {
            "context_precision":  "evaluated on full report",
            "faithfulness":       "evaluated on full report",
            "context_recall":     "evaluated on executive summary",
            "answer_correctness": "evaluated on executive summary",
        },
        "overall_scores": {
            "context_precision":  round(scores.get("llm_context_precision_with_reference", 0), 4),
            "faithfulness":       round(scores.get("faithfulness", 0), 4),
            "context_recall":     round(scores.get("context_recall", 0), 4),
            "answer_correctness": round(scores.get("answer_correctness", 0), 4),
        },
        "per_sample": per_sample,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  Results saved to: {output_path}")
    print("\n  README table (copy-paste this):\n")
    print("  | Metric             | Score | Text evaluated   |")
    print("  |--------------------|-------|------------------|")
    text_map = {
        "context_precision":  "full report",
        "faithfulness":       "full report",
        "context_recall":     "exec summary",
        "answer_correctness": "exec summary",
    }
    for metric, score in output["overall_scores"].items():
        name = metric.replace("_", " ").title()
        text = text_map.get(metric, "")
        print(f"  | {name:<18} | {score:.3f} | {text:<16} |")


# ─────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────

def main():
    global API_BASE_URL

    parser = argparse.ArgumentParser(
        description="Run RAGAS evaluation on the Clinical Research Copilot."
    )
    parser.add_argument(
        "--difficulty",
        choices=["easy", "medium", "hard"],
        default=None,
        help="Only evaluate queries of this difficulty level.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Only evaluate the first N queries (useful for a quick test run).",
    )
    parser.add_argument(
        "--api-url",
        default=API_BASE_URL,
        help=f"API base URL (default: {API_BASE_URL})",
    )
    args = parser.parse_args()

    API_BASE_URL = args.api_url

    print("\n" + "=" * 60)
    print("  Clinical Research Copilot — RAGAS Evaluation")
    print("=" * 60)

    # Step 1: verify API is running
    print("\n[1/4] Checking API health...")
    if not check_api_health():
        sys.exit(1)

    # Step 2: load golden dataset
    print(f"\n[2/4] Loading golden dataset from {DATASET_PATH}...")
    if not DATASET_PATH.exists():
        print(f"  ERROR: {DATASET_PATH} not found.")
        print("  Make sure golden_dataset.jsonl is in the eval/ directory.")
        sys.exit(1)

    records = load_dataset(DATASET_PATH, difficulty=args.difficulty, sample=args.sample)
    print(f"  Loaded {len(records)} records")

    # Step 3: call the pipeline for each query
    print(f"\n[3/4] Collecting pipeline outputs...")
    samples_full, samples_summary, meta = collect_pipeline_outputs(records)

    if len(samples_full) == 0:
        print("  ERROR: No samples collected. Check API logs.")
        sys.exit(1)

    # Step 4: run RAGAS — Option A (two separate evaluate() calls)
    print(f"\n[4/4] Running RAGAS evaluation (Option A: split by metric)...")
    result_grounding, result_quality = run_ragas_evaluation(samples_full, samples_summary)

    # Report
    print_results(result_grounding, result_quality, meta)
    save_results(result_grounding, result_quality, meta, RESULTS_PATH)


if __name__ == "__main__":
    main()  