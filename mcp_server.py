"""
mcp_server.py — MCP Server for Pharma RAG Pipeline
====================================================
Python 3.13 | FastMCP

Exposes the Agentic RAG pipeline as MCP tools that Claude Desktop
(or any MCP client) can call during conversation.

Two tools:
  search_literature  — fast raw evidence retrieval (~2s)
  run_full_research  — complete multi-agent pipeline (~25s)

Why two tools (not one, not three):
  One tool would force the full 25-second pipeline for simple lookups.
  Three tools (with get_drug_summary) would duplicate routing logic
  that the supervisor agent already handles. Two tools have a clear,
  unambiguous contract: raw evidence vs synthesized report.

Architecture position:
  Claude Desktop / MCP client
        ↓  MCP protocol (stdio or SSE)
  mcp_server.py
        ↓  calls Python functions directly
  agents.run_research()  OR  retriever.retrieve()
        ↓  (full pipeline beneath)
  query_rewriter → retriever → grader → agents → synthesizer

Transport:
  stdio  — for Claude Desktop (local, same machine)
  SSE    — for remote access (HTTP, deployed server)
  Default: stdio (started with `python mcp_server.py`)

How to connect to Claude Desktop:
  Add to claude_desktop_config.json:
  {
    "mcpServers": {
      "pharma-rag": {
        "command": "python",
        "args": ["path/to/mcp_server.py"],
        "env": {
          "OPENAI_API_KEY": "sk-...",
          "COHERE_API_KEY": "...",
          "DB_HOST": "localhost",
          "DB_PORT": "5432",
          "DB_NAME": "pharma_rag",
          "DB_USER": "vineet",
          "DB_PASSWORD": "devpassword"
        }
      }
    }
  }
"""

import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# Add project root to Python path — mcp_server.py is in mcp/ subfolder
# but retriever.py, agents.py etc. are in the project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP, Context

# Pipeline imports — these are our own modules
from retriever import retrieve, MetadataFilter, RetrievedChunk, close_pool
from agents import run_research, AgentOutput
from query_rewriter import DB_VOCABULARY

load_dotenv()


# ─────────────────────────────────────────────────────────────────────
# 1. MCP SERVER INSTANCE
# ─────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def _lifespan(server):
    """
    Lifespan context manager for FastMCP server.

    Why lifespan instead of @mcp.on_shutdown():
      The installed FastMCP version uses the lifespan pattern (same as
      FastAPI/Starlette). The server runs during the yield. After yield
      (teardown phase), we release the asyncpg connection pool.

    Why close_pool() matters:
      Without it, asyncpg connections (3-10) remain open after the MCP
      server process exits. The OS eventually reclaims them but explicit
      cleanup is cleaner — especially during development when the server
      restarts frequently.
    """
    yield                  # server runs here
    await close_pool()     # teardown on shutdown
    print("Pharma RAG MCP server shut down. DB pool closed.")


mcp = FastMCP(
    name="pharma-rag",
    lifespan=_lifespan,
)


# ─────────────────────────────────────────────────────────────────────
# 2. FORMATTER UTILITIES
# ─────────────────────────────────────────────────────────────────────

def _format_chunk(i: int, chunk: RetrievedChunk) -> str:
    """
    Formats one RetrievedChunk into readable text for MCP response.
    Includes metadata, scores, and content preview.
    """
    sample = f"N={chunk.sample_size}" if chunk.sample_size else "N=NR"
    return (
        f"[{i}] {chunk.drug} | {chunk.cancer_type} | {chunk.study_type} | "
        f"{sample} | {chunk.journal} ({chunk.year})\n"
        f"    PMID: {chunk.source} | Evidence: Level {chunk.evidence_level} | "
        f"Endpoints: {chunk.endpoints or 'NR'}\n"
        f"    Rerank score: {chunk.rerank_score:.3f}\n"
        f"    {chunk.content[:500]}"
    )


def _format_search_results(result: dict) -> str:
    """
    Formats the full retrieve() output dict into a structured text response.
    Called by search_literature tool.
    """
    chunks = result.get("chunks", [])
    if not chunks:
        return (
            f"No results found for query: '{result.get('original_query', '')}'\n"
            f"Filters applied: {result.get('filters_applied', 'none')}\n"
            f"Filter level: {result.get('filter_level', 'unknown')}\n"
            f"Try a broader query or fewer filters."
        )

    header = (
        f"Query: {result.get('original_query', '')}\n"
        f"Rewritten: {result.get('rewritten_query', '')}\n"
        f"Confidence: {result.get('confidence', 'unknown')}\n"
        f"Filters: drug={result['filters_applied'].drug}, "
        f"cancer={result['filters_applied'].cancer_type}, "
        f"endpoints={result['filters_applied'].endpoints}\n"
        f"Filter level: {result.get('filter_level', 'unknown')} | "
        f"Candidates: {result.get('n_candidates', 0)} -> "
        f"Returned: {result.get('n_returned', 0)}\n"
        f"{'-' * 60}\n"
    )

    chunk_texts = [_format_chunk(i, c) for i, c in enumerate(chunks, 1)]

    return header + "\n\n".join(chunk_texts)


def _format_research_output(output: AgentOutput) -> str:
    """
    Formats AgentOutput for MCP response.
    The report is already well-formatted markdown from the synthesizer —
    we add metadata header and return it.
    """
    header = (
        f"Query: {output.query}\n"
        f"Verdict: {output.overall_verdict}\n"
        f"Evidence: {output.evidence_summary}\n"
        f"Sections: {[s.title for s in output.sections]}\n"
        f"{'-' * 60}\n\n"
    )
    return header + output.report


# ─────────────────────────────────────────────────────────────────────
# 3. MCP TOOLS
# ─────────────────────────────────────────────────────────────────────

@mcp.tool()
async def search_literature(
    query: str,
    drug: str | None = None,
    cancer_type: str | None = None,
    year_min: int | None = None,
    study_type: str | None = None,
    endpoints: str | None = None,
) -> str:
    """
    Search the pharma knowledge base for clinical evidence.

    Returns raw retrieved evidence chunks with metadata and relevance scores.
    Use when you want to see source papers, not a synthesized answer.
    Fast: ~2 seconds.

    Args:
        query: Clinical question in natural language.
               Examples: "pembrolizumab OS in NSCLC",
                         "osimertinib EGFR mutation resistance",
                         "KEYNOTE-189 results"
        drug: Optional filter — exact drug name from:
              pembrolizumab, nivolumab, osimertinib, trastuzumab, metformin
        cancer_type: Optional filter — e.g. "nsclc", "breast-cancer", "melanoma"
        year_min: Optional filter — minimum publication year (e.g. 2020)
        study_type: Optional filter — e.g. "RCT", "meta-analysis", "real-world-study"
        endpoints: Optional filter — e.g. "OS", "PFS", "ORR", "AE"

    Returns:
        Formatted evidence chunks with drug, cancer type, study design,
        sample size, journal, year, PMID, evidence level, and content preview.
    """
    # Build MetadataFilter from explicit parameters if provided
    filters = None
    if any([drug, cancer_type, year_min, study_type, endpoints]):
        filters = MetadataFilter(
            drug        = drug,
            cancer_type = cancer_type,
            year_min    = year_min,
            study_type  = study_type,
            endpoints   = endpoints,
        )

    try:
        result = await retrieve(
            query   = query,
            filters = filters,
            # use_rewriter=True when no explicit filters
            # use_rewriter=False when filters provided (skip rewriter)
            use_rewriter = (filters is None),
        )
        return _format_search_results(result)

    except Exception as e:
        return f"Search failed: {type(e).__name__}: {e}"


@mcp.tool()
async def run_full_research(query: str, ctx: Context) -> str:
    """
    Run a comprehensive clinical research analysis.

    Activates specialist agents (efficacy, safety, mechanism, competitor)
    based on the query. Each agent independently retrieves and grades
    evidence, then a synthesizer merges findings into a cited research brief.

    Use for: clinical questions needing a thorough, cited answer.
    Takes: 20-30 seconds.

    Args:
        query: Clinical research question in natural language.
               Examples:
               - "What is the overall survival benefit of pembrolizumab in NSCLC?"
               - "Compare pembrolizumab vs nivolumab in NSCLC survival"
               - "What are the immune-related adverse events of pembrolizumab?"
               - "How does osimertinib work in EGFR-mutant NSCLC?"

    Returns:
        A comprehensive markdown research brief with:
        - Executive summary
        - Evidence sections per specialist agent
        - Inline citations [PMID: XXXXX]
        - Evidence quality assessment
        - Limitations section
        - Overall verdict (accepted/partial/insufficient)
    """
    async def _send_progress(pct: int, msg: str = "") -> None:
        """Send progress notification — silently skip if unsupported."""
        try:
            await ctx.report_progress(pct, 100)
            if msg:
                await ctx.info(msg)
        except Exception:
            pass   # progress reporting not supported — tool still works

    try:
        await _send_progress(0, "Starting research pipeline...")

        # Run the full pipeline as a background task
        # Send progress every 5 seconds to keep the MCP connection alive
        # Without this, MCP Inspector/Claude Desktop times out at ~60s
        task = asyncio.create_task(run_research(query))

        progress = 5
        while not task.done():
            await asyncio.sleep(5)
            if not task.done():
                progress = min(progress + 10, 90)
                await _send_progress(progress, f"Pipeline running... {progress}%")

        output = await task

        # Check for exceptions that occurred during execution
        if task.exception():
            raise task.exception()

        await _send_progress(100, "Research complete.")
        return _format_research_output(output)

    except Exception as e:
        return f"Research failed: {type(e).__name__}: {e}"


# ─────────────────────────────────────────────────────────────────────
# 4. MCP RESOURCES — static reference data
# ─────────────────────────────────────────────────────────────────────

@mcp.resource("pharma://drugs")
async def list_drugs() -> str:
    """
    Lists all drugs available in the knowledge base with their
    trade names and aliases.
    """
    lines = []
    for drug in DB_VOCABULARY["drugs"]:
        aliases = [
            alias for alias, canonical
            in DB_VOCABULARY["drug_aliases"].items()
            if canonical == drug
        ]
        alias_str = f" (aliases: {', '.join(aliases)})" if aliases else ""
        lines.append(f"- {drug}{alias_str}")

    return (
        "Drugs in the knowledge base:\n\n"
        + "\n".join(lines)
        + "\n\nCancer types: "
        + ", ".join(DB_VOCABULARY["cancer_types"][:15])
        + ", ..."
        + "\n\nEndpoints: "
        + ", ".join(DB_VOCABULARY["endpoints"])
        + "\n\nStudy types: "
        + ", ".join(DB_VOCABULARY["study_types"][:8])
        + ", ..."
    )


# ─────────────────────────────────────────────────────────────────────
# 5. ENTRY POINT
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    transport = sys.argv[1] if len(sys.argv) > 1 else "stdio"

    if transport == "sse":
        # SSE transport — for remote access via HTTP
        # Start with: python mcp_server.py sse
        port = int(os.getenv("MCP_PORT", "8765"))
        print(f"Starting Pharma RAG MCP server (SSE) on port {port}")
        mcp.run(transport="sse", port=port)
    else:
        # stdio transport — for Claude Desktop (default)
        # Start with: python mcp_server.py
        print("Starting Pharma RAG MCP server (stdio)")
        mcp.run(transport="stdio")
