"""
test_mcp_protocol.py — Direct MCP JSON-RPC protocol test.
Launches each server as a subprocess and exchanges messages exactly as
the MCP Inspector does, but from the command line.

Usage:
    python test_mcp_protocol.py

Tests:
  1. test_minimal_mcp.py  → hello, slow_hello, slow_hello_20
  2. test_import_mcp.py   → quick_search
  3. mcp_server.py        → search_literature (if above pass)
"""

import asyncio
import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).parent


def _msg(obj: dict) -> bytes:
    return (json.dumps(obj) + "\n").encode()


async def run_mcp_session(
    script: Path,
    tool_calls: list[tuple[str, dict, float]],
) -> list[dict]:
    """
    Starts script as a subprocess, performs MCP initialize handshake,
    then calls each tool.

    tool_calls: list of (tool_name, arguments_dict, timeout_seconds)
    Returns list of result dicts (one per tool call).
    """
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"Server: {script.name}", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)

    proc = await asyncio.create_subprocess_exec(
        sys.executable, str(script),
        stdin  = asyncio.subprocess.PIPE,
        stdout = asyncio.subprocess.PIPE,
        stderr = asyncio.subprocess.PIPE,
    )

    async def read_line(label: str, timeout: float = 15) -> dict | None:
        try:
            raw = await asyncio.wait_for(proc.stdout.readline(), timeout=timeout)
            if not raw:
                print(f"  [{label}] EOF from server", file=sys.stderr)
                return None
            text = raw.decode().strip()
            # Skip any non-JSON lines (tracing prints etc.)
            for candidate in text.splitlines():
                candidate = candidate.strip()
                if candidate.startswith("{"):
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        pass
            print(f"  [{label}] Non-JSON output: {text[:120]!r}", file=sys.stderr)
            return None
        except asyncio.TimeoutError:
            print(f"  [{label}] TIMEOUT waiting for server response", file=sys.stderr)
            return None

    results = []

    # ── 1. Initialize ─────────────────────────────────────────────────
    proc.stdin.write(_msg({
        "jsonrpc": "2.0", "id": 0, "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities":    {},
            "clientInfo":      {"name": "protocol-test", "version": "1.0"},
        }
    }))
    await proc.stdin.drain()

    init_resp = await read_line("initialize", timeout=30)
    if init_resp and "result" in init_resp:
        srv = init_resp["result"].get("serverInfo", {})
        print(f"  initialize: OK  server={srv.get('name')} v{srv.get('version')}", file=sys.stderr)
    else:
        print(f"  initialize: FAILED  resp={init_resp}", file=sys.stderr)
        proc.kill()
        return []

    # ── 2. initialized notification ───────────────────────────────────
    proc.stdin.write(_msg({"jsonrpc": "2.0", "method": "notifications/initialized"}))
    await proc.stdin.drain()

    # ── 3. Tool calls ─────────────────────────────────────────────────
    for call_id, (tool, args, timeout) in enumerate(tool_calls, start=1):
        t0 = time.perf_counter()
        print(f"\n  Calling {tool}({args})  [timeout={timeout}s]...", file=sys.stderr)

        proc.stdin.write(_msg({
            "jsonrpc": "2.0", "id": call_id, "method": "tools/call",
            "params": {"name": tool, "arguments": args},
        }))
        await proc.stdin.drain()

        resp = await read_line(f"{tool}", timeout=timeout)
        elapsed = (time.perf_counter() - t0) * 1000

        if resp is None:
            results.append({"tool": tool, "status": "TIMEOUT", "elapsed_ms": elapsed})
            print(f"  {tool}: TIMEOUT after {elapsed:.0f}ms", file=sys.stderr)
        elif "error" in resp:
            results.append({"tool": tool, "status": "ERROR", "error": resp["error"], "elapsed_ms": elapsed})
            print(f"  {tool}: ERROR {resp['error']}  ({elapsed:.0f}ms)", file=sys.stderr)
        elif "result" in resp:
            content = resp["result"].get("content", [{}])
            text = content[0].get("text", "") if content else ""
            results.append({"tool": tool, "status": "OK", "result": text[:200], "elapsed_ms": elapsed})
            print(f"  {tool}: OK  ({elapsed:.0f}ms)  → {text[:80]!r}", file=sys.stderr)
        else:
            results.append({"tool": tool, "status": "UNKNOWN", "resp": resp, "elapsed_ms": elapsed})
            print(f"  {tool}: UNKNOWN  resp={resp}", file=sys.stderr)

    # ── 4. Drain stderr from server ───────────────────────────────────
    proc.kill()
    try:
        stderr_out = await asyncio.wait_for(proc.stderr.read(), timeout=2)
        if stderr_out:
            print(f"\n  [server stderr]:\n{stderr_out.decode('utf-8', errors='replace')[:800]}", file=sys.stderr)
    except asyncio.TimeoutError:
        pass

    return results


async def main() -> None:
    all_results: dict[str, list[dict]] = {}

    # ── Test 1: minimal server (no pipeline imports) ──────────────────
    print("\n" + "#"*60, file=sys.stderr)
    print("TEST 1: test_minimal_mcp.py — no imports", file=sys.stderr)
    print("#"*60, file=sys.stderr)

    minimal = ROOT / "mcp" / "test_minimal_mcp.py"
    if minimal.exists():
        all_results["minimal"] = await run_mcp_session(minimal, [
            ("hello",         {"name": "test"},   10),
            ("slow_hello",    {"name": "test"},   30),
            ("slow_hello_20", {"name": "test"},   90),
        ])
    else:
        print("  SKIP: test_minimal_mcp.py not found", file=sys.stderr)

    # ── Test 2: import server (all pipeline imports) ──────────────────
    print("\n" + "#"*60, file=sys.stderr)
    print("TEST 2: test_import_mcp.py — pipeline imports + quick_search", file=sys.stderr)
    print("#"*60, file=sys.stderr)

    import_server = ROOT / "mcp" / "test_import_mcp.py"
    if import_server.exists():
        all_results["imports"] = await run_mcp_session(import_server, [
            ("quick_search", {"query": "pembrolizumab NSCLC"}, 60),
        ])
    else:
        print("  SKIP: test_import_mcp.py not found", file=sys.stderr)

    # ── Test 3: real server search_literature ─────────────────────────
    print("\n" + "#"*60, file=sys.stderr)
    print("TEST 3: mcp_server.py — search_literature", file=sys.stderr)
    print("#"*60, file=sys.stderr)

    real_server = ROOT / "mcp" / "mcp_server.py"
    if real_server.exists():
        all_results["real"] = await run_mcp_session(real_server, [
            ("search_literature", {"query": "pembrolizumab NSCLC"}, 60),
        ])
    else:
        print("  SKIP: mcp_server.py not found", file=sys.stderr)

    # ── Summary ───────────────────────────────────────────────────────
    print("\n\n" + "#"*60)
    print("PROTOCOL TEST SUMMARY")
    print("#"*60)
    for server_name, calls in all_results.items():
        print(f"\n  [{server_name}]")
        for r in calls:
            ms = r.get("elapsed_ms", 0)
            status = r["status"]
            result_preview = r.get("result", r.get("error", ""))
            print(f"    {r['tool']:<20}  {status:<8}  {ms:>8.0f} ms  {str(result_preview)[:60]}")


if __name__ == "__main__":
    asyncio.run(main())
