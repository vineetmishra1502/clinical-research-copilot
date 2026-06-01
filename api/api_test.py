"""
api_test.py — Test all Pharma RAG API endpoints
=================================================
Run with: python api_test.py

Requires the API server to be running:
  uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

Tests all 4 endpoints in order (fast to slow):
  1. GET  /health   — instant
  2. GET  /drugs    — instant
  3. POST /search   — ~2-5 seconds
  4. POST /research — ~15-30 seconds
"""

import requests
import json
import time
import sys

BASE_URL = "http://localhost:8000"


def separator(title: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


# -----------------------------------------------------------------
# Test 1: GET /health
# -----------------------------------------------------------------
def test_health():
    separator("TEST 1: GET /health")
    try:
        start = time.perf_counter()
        resp = requests.get(f"{BASE_URL}/health")
        elapsed = time.perf_counter() - start

        print(f"Status code: {resp.status_code}")
        print(f"Time:        {elapsed:.2f}s")
        print()

        data = resp.json()
        print(f"Status:      {data['status']}")
        print(f"Database:    {data['database']}")
        print(f"API keys:")
        for key, present in data["api_keys"].items():
            status = "OK" if present else "MISSING"
            print(f"  {key:12s} {status}")

        return data["status"] == "healthy"

    except requests.ConnectionError:
        print("FAILED: Cannot connect to API server.")
        print("Start with: uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload")
        return False
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")
        return False


# -----------------------------------------------------------------
# Test 2: GET /drugs
# -----------------------------------------------------------------
def test_drugs():
    separator("TEST 2: GET /drugs")
    try:
        start = time.perf_counter()
        resp = requests.get(f"{BASE_URL}/drugs")
        elapsed = time.perf_counter() - start

        print(f"Status code: {resp.status_code}")
        print(f"Time:        {elapsed:.2f}s")
        print()

        data = resp.json()

        print("Drugs in knowledge base:")
        for drug in data["drugs"]:
            aliases = ", ".join(drug["aliases"]) if drug["aliases"] else "none"
            print(f"  {drug['name']:20s} aliases: {aliases}")

        print(f"\nCancer types ({len(data['cancer_types'])}): {', '.join(data['cancer_types'][:8])}, ...")
        print(f"Endpoints    ({len(data['endpoints'])}):    {', '.join(data['endpoints'])}")
        print(f"Study types  ({len(data['study_types'])}):  {', '.join(data['study_types'][:6])}, ...")

        return resp.status_code == 200

    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")
        return False


# -----------------------------------------------------------------
# Test 3a: POST /search — no explicit filters
# -----------------------------------------------------------------
def test_search_basic():
    separator("TEST 3a: POST /search (no filters, rewriter active)")

    query = "pembrolizumab overall survival NSCLC"
    print(f"Query: {query}")
    print()

    try:
        start = time.perf_counter()
        resp = requests.post(
            f"{BASE_URL}/search",
            json={"query": query},
        )
        elapsed = time.perf_counter() - start

        print(f"Status code: {resp.status_code}")
        print(f"Time:        {elapsed:.2f}s")
        print()

        data = resp.json()

        if resp.status_code != 200:
            print(f"Error: {data.get('detail', 'unknown')}")
            return False

        print(f"Rewritten:   {data['rewritten_query']}")
        print(f"Confidence:  {data['confidence']}")
        print(f"Filter lvl:  {data['filter_level']}")
        print(f"Candidates:  {data['n_candidates']} -> {data['n_returned']} returned")
        print(f"\nChunks returned: {len(data['chunks'])}")
        print("-" * 60)

        for i, chunk in enumerate(data["chunks"], 1):
            sample = f"N={chunk['sample_size']}" if chunk['sample_size'] else "N=NR"
            print(f"\n  [{i}] {chunk['drug']} | {chunk['cancer_type']} | "
                  f"{chunk['study_type']} | {sample}")
            print(f"      {chunk['journal']} ({chunk['year']}) | "
                  f"PMID: {chunk['source']} | Level {chunk['evidence_level']}")
            print(f"      Endpoints: {chunk['endpoints']} | "
                  f"Rerank: {chunk['rerank_score']:.4f}")
            print(f"      {chunk['content'][:150]}...")

        return len(data["chunks"]) > 0

    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")
        return False


# -----------------------------------------------------------------
# Test 3b: POST /search — with explicit filters
# -----------------------------------------------------------------
def test_search_filtered():
    separator("TEST 3b: POST /search (with explicit filters)")

    payload = {
        "query": "overall survival benefit",
        "drug": "pembrolizumab",
        "cancer_type": "nsclc",
        "year_min": 2020,
        "endpoints": "OS",
    }
    print(f"Query:   {payload['query']}")
    print(f"Filters: drug={payload['drug']}, cancer={payload['cancer_type']}, "
          f"year>={payload['year_min']}, endpoints={payload['endpoints']}")
    print()

    try:
        start = time.perf_counter()
        resp = requests.post(f"{BASE_URL}/search", json=payload)
        elapsed = time.perf_counter() - start

        print(f"Status code: {resp.status_code}")
        print(f"Time:        {elapsed:.2f}s")
        print()

        data = resp.json()

        if resp.status_code != 200:
            print(f"Error: {data.get('detail', 'unknown')}")
            return False

        print(f"Filter lvl:  {data['filter_level']}")
        print(f"Candidates:  {data['n_candidates']} -> {data['n_returned']} returned")
        print(f"Chunks:      {len(data['chunks'])}")

        if data["chunks"]:
            print("-" * 60)
            for i, chunk in enumerate(data["chunks"][:3], 1):
                print(f"  [{i}] {chunk['drug']} | {chunk['cancer_type']} | "
                      f"{chunk['year']} | Rerank: {chunk['rerank_score']:.4f}")

        return len(data["chunks"]) > 0

    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")
        return False


# -----------------------------------------------------------------
# Test 4: POST /research — full pipeline
# -----------------------------------------------------------------
def test_research():
    separator("TEST 4: POST /research (full pipeline)")

    query = "What is the overall survival benefit of pembrolizumab in NSCLC?"
    print(f"Query: {query}")
    print(f"(This takes 15-30 seconds...)")
    print()

    try:
        start = time.perf_counter()
        resp = requests.post(
            f"{BASE_URL}/research",
            json={"query": query},
            timeout=180,
        )
        elapsed = time.perf_counter() - start

        print(f"Status code: {resp.status_code}")
        print(f"Time:        {elapsed:.2f}s")
        print()

        data = resp.json()

        if resp.status_code != 200:
            print(f"Error: {data.get('detail', 'unknown')}")
            return False

        print(f"Verdict:     {data['overall_verdict']}")
        print(f"Server time: {data['elapsed_seconds']:.2f}s")
        print(f"Sections:    {len(data['sections'])}")
        print(f"Report len:  {len(data['report'])} chars")
        print()

        # Evidence summary
        print(f"Evidence summary:")
        print(f"  {data['evidence_summary']}")
        print()

        # Sections
        print("Sections:")
        print("-" * 60)
        for i, section in enumerate(data["sections"], 1):
            print(f"\n  [{i}] {section['title']}")
            print(f"      Verdict note: {section['verdict_note'] or '(none)'}")
            print(f"      Evidence:     {section['evidence_quality'][:100]}")
            print(f"      Citations:    {len(section['citations'])}")
            for cit in section["citations"][:3]:
                print(f"        - {cit}")
            print(f"      Content preview:")
            content_preview = section["content"][:300].replace("\n", "\n        ")
            print(f"        {content_preview}...")
            print(f"      Limitations:  {section['limitations'][:150]}")

        # Full report preview
        print()
        print("Full report (first 500 chars):")
        print("-" * 60)
        print(data["report"][:500])
        print("...")

        return data["overall_verdict"] in ("accepted", "partial", "insufficient")

    except requests.Timeout:
        print("FAILED: Request timed out after 180 seconds.")
        return False
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")
        return False


# -----------------------------------------------------------------
# Test 5: POST /research — comparison query (multi-agent)
# -----------------------------------------------------------------
def test_research_comparison():
    separator("TEST 5: POST /research (comparison, multi-agent)")

    query = "Compare pembrolizumab vs nivolumab in NSCLC overall survival"
    print(f"Query: {query}")
    print(f"(Supervisor should activate 2+ agents. Takes 20-30 seconds...)")
    print()

    try:
        start = time.perf_counter()
        resp = requests.post(
            f"{BASE_URL}/research",
            json={"query": query},
            timeout=180,
        )
        elapsed = time.perf_counter() - start

        print(f"Status code: {resp.status_code}")
        print(f"Time:        {elapsed:.2f}s")
        print()

        data = resp.json()

        if resp.status_code != 200:
            print(f"Error: {data.get('detail', 'unknown')}")
            return False

        print(f"Verdict:     {data['overall_verdict']}")
        print(f"Server time: {data['elapsed_seconds']:.2f}s")
        print(f"Sections:    {len(data['sections'])}")

        section_titles = [s["title"] for s in data["sections"]]
        print(f"Agents used: {section_titles}")

        total_citations = sum(len(s["citations"]) for s in data["sections"])
        print(f"Citations:   {total_citations} total")

        print(f"\nReport preview (first 400 chars):")
        print("-" * 60)
        print(data["report"][:400])
        print("...")

        return len(data["sections"]) >= 1

    except requests.Timeout:
        print("FAILED: Request timed out after 180 seconds.")
        return False
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")
        return False


# -----------------------------------------------------------------
# Test 6: Error handling — invalid input
# -----------------------------------------------------------------
def test_error_handling():
    separator("TEST 6: Error handling")

    # 6a: Empty query (should fail validation — min_length=3)
    print("6a: Empty query (should return 422)")
    resp = requests.post(f"{BASE_URL}/search", json={"query": ""})
    print(f"  Status: {resp.status_code} (expected: 422)")
    passed_a = resp.status_code == 422

    # 6b: Missing query field
    print("\n6b: Missing query field (should return 422)")
    resp = requests.post(f"{BASE_URL}/search", json={})
    print(f"  Status: {resp.status_code} (expected: 422)")
    passed_b = resp.status_code == 422

    # 6c: Invalid JSON
    print("\n6c: Invalid JSON body (should return 422)")
    resp = requests.post(
        f"{BASE_URL}/search",
        data="not json",
        headers={"Content-Type": "application/json"},
    )
    print(f"  Status: {resp.status_code} (expected: 422)")
    passed_c = resp.status_code == 422

    return passed_a and passed_b and passed_c


# -----------------------------------------------------------------
# Run all tests
# -----------------------------------------------------------------
def main():
    print("\n" + "=" * 70)
    print("  PHARMA RAG API -- Full Test Suite")
    print("  Server: " + BASE_URL)
    print("=" * 70)

    results = {}

    # Test 1 -- if health fails, skip everything
    results["health"] = test_health()
    if not results["health"]:
        print("\nHealth check failed. Is the server running?")
        print("Start with: uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload")
        sys.exit(1)

    # Test 2
    results["drugs"] = test_drugs()

    # Test 3a
    results["search_basic"] = test_search_basic()

    # Test 3b
    results["search_filtered"] = test_search_filtered()

    # Test 4
    results["research"] = test_research()

    # Test 5
    results["research_comparison"] = test_research_comparison()

    # Test 6
    results["error_handling"] = test_error_handling()

    # Summary
    separator("TEST SUMMARY")
    total = len(results)
    passed = sum(1 for v in results.values() if v)

    for name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"  {name:25s} {status}")

    print(f"\n  {passed}/{total} tests passed")

    if passed == total:
        print("\n  All tests passed!")
    else:
        failed = [name for name, v in results.items() if not v]
        print(f"\n  Failed: {', '.join(failed)}")

    print()


if __name__ == "__main__":
    main()