# Langfuse Observability — Integration Guide

## What is Langfuse

Langfuse is an open-source LLM observability platform. Every pipeline run produces one trace in the Langfuse dashboard containing:

- Full LangGraph execution tree (supervisor → agents → synthesizer)
- Token counts per LLM call (input + output)
- Cost per call, auto-calculated from model + token counts
- Latency per node and end-to-end
- Full prompts and completions for every LLM call

**Why Langfuse over LangSmith:**

| | Langfuse | LangSmith |
|---|---|---|
| Open source | Yes | No |
| Self-hostable | Yes (Docker) | No — cloud only |
| Data privacy | Stays local if self-hosted | Always sent to LangChain servers |
| RAGAS integration | Built-in score logging | Manual |
| Free tier | 50k traces/month | Limited |

---

## What Gets Traced

Every `/research` request becomes one Langfuse trace:

```
research-pipeline            ~33s total
  CHAIN  supervisor           ~3s    gpt-4o-mini — query decomposition
  AGENT  specialist_agent     ~18s   gpt-4o-mini — retrieval + grading + section
  CHAIN  synthesizer          ~8s    gpt-4o      — final report
  CHAIN  RunnableSequence     ~2-8s  prompt | llm chain executions
  CHAIN  ChatPromptTemplate   ~0s    prompt building (near zero)
  CHAIN  route_to_agents      ~0s    Send() dispatch (pure Python)
```

Every `/search` request becomes one trace:

```
search-pipeline              ~3-5s total
  CHAIN  RunnableSequence    query rewriter execution
  CHAIN  ChatOpenAI          embedding call
```

**CHAIN** = a LangChain Runnable built with the `|` pipe operator  
**AGENT** = a LangGraph node dispatched via `Send()`

---

## Files Changed

### `langfuse_setup.py` (new file — project root)

Central utility module with three public functions:

```python
langfuse_enabled() -> bool
    # True if keys set + package installed + not disabled
    # Called before every Langfuse operation

get_callback_handler(session_id, trace_name, metadata, user_id)
    -> tuple[handler, config_extras]
    # Creates one handler per request (never reuse across requests)
    # Returns (None, {}) if Langfuse not configured — pipeline runs unchanged

flush_handler(handler) -> None
    # Sends buffered events to Langfuse before request completes
    # Safe to call with None
```

### `agents.py` (2 changes)

```python
# Change 1 — new import
from langfuse_setup import get_callback_handler

# Change 2 — run_research() accepts optional invoke_config
async def run_research(
    query:         str,
    invoke_config: dict = None,   # pre-built config from get_callback_handler()
) -> AgentOutput:
    result = await _graph.ainvoke(initial_state, config=invoke_config or {})
    return result["final_output"]
```

When `invoke_config` is `None`, `_graph.ainvoke()` is identical to the original call.

### `api/main.py` (4 change areas)

```python
# New imports
import uuid
from langfuse_setup import get_callback_handler, flush_handler, langfuse_enabled

# /research — one handler per request
handler, invoke_config = get_callback_handler(
    session_id = f"research-{request_id}",
    trace_name = "research-pipeline",
    metadata   = {"endpoint": "/research", "query": request.query[:200]},
)
try:
    output = await run_research(request.query, invoke_config=invoke_config)
finally:
    flush_handler(handler)   # always flushes, even on exception

# /search — handler created for request-level metadata, _ discards config
# (retrieve() signature unchanged — no child spans on /search)
handler, _ = get_callback_handler(session_id=f"search-{request_id}", ...)
try:
    result = await retrieve(...)
finally:
    flush_handler(handler)

# /health — langfuse key added (informational, not a health requirement)
api_keys = {
    "openai":   bool(os.getenv("OPENAI_API_KEY", "").startswith("sk-")),
    "cohere":   bool(os.getenv("COHERE_API_KEY", "")),
    "langfuse": langfuse_enabled(),
}
# status = "healthy" requires only openai + cohere — langfuse is optional
```

---

## v4 Breaking Changes

The integration was written for Langfuse v4 (you have 4.7.1). Three things changed from v2/v3:

**1. Import path**
```python
# v2 — broken
from langfuse.callback import CallbackHandler

# v4 — correct
from langfuse.langchain import CallbackHandler
```

**2. Constructor arguments removed**
```python
# v2 — broken on v4
CallbackHandler(public_key="pk-lf-...", secret_key="sk-lf-...", host="...")

# v4 — initialise Langfuse() client once with credentials
# CallbackHandler() finds it automatically with no args
Langfuse(public_key="pk-lf-...", secret_key="sk-lf-...", host="...")
handler = CallbackHandler()
```

**3. Session/trace metadata location**
```python
# v2 — broken on v4
CallbackHandler(session_id="...", trace_name="...")

# v4 — metadata goes in the invoke() config dict
config = {
    "callbacks": [handler],
    "run_name":  "research-pipeline",
    "metadata":  {"langfuse_session_id": "...", ...}
}
_graph.ainvoke(state, config=config)
```

This is why `get_callback_handler()` returns a tuple `(handler, config_extras)` — the config dict carries the session ID and trace name, not the handler object.

---

## Setup

### 1. Install

```bash
pip install langfuse
```

Add to `requirements.txt`:
```
langfuse>=2.0.0
```

### 2. Get API keys

**Option A — Cloud (recommended, 2 minutes):**
1. Sign up at [cloud.langfuse.com](https://cloud.langfuse.com) — free tier
2. Create project: `clinical-research-copilot`
3. Settings → API Keys → Create key pair

**Option B — Self-hosted (data stays local):**
```bash
# docker-compose.yml
version: "3.9"
services:
  langfuse-db:
    image: postgres:16
    environment:
      POSTGRES_USER: langfuse
      POSTGRES_PASSWORD: langfuse
      POSTGRES_DB: langfuse
    volumes:
      - langfuse_db:/var/lib/postgresql/data

  langfuse:
    image: langfuse/langfuse:latest
    depends_on: [langfuse-db]
    ports: ["3000:3000"]
    environment:
      DATABASE_URL: postgresql://langfuse:langfuse@langfuse-db:5432/langfuse
      NEXTAUTH_SECRET: changeme
      NEXTAUTH_URL: http://localhost:3000
      SALT: changeme

volumes:
  langfuse_db:
```
```bash
docker compose up -d
# Open http://localhost:3000, create account + project + keys
```

### 3. Add to `.env`

```bash
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com   # or http://localhost:3000
```

To disable without removing keys:
```bash
LANGFUSE_ENABLED=false
```

---

## Verify It Works

### Step 1 — Health check shows Langfuse enabled

```bash
curl http://localhost:8000/health
```

Expected:
```json
{
  "status": "healthy",
  "database": "connected",
  "api_keys": {"openai": true, "cohere": true, "langfuse": true}
}
```

### Step 2 — Run the debug script

```bash
python test_langfuse_debug.py
```

This checks every failure point: .env loading, key format, package install, handler creation, a real traced LLM call, and a direct SDK connection test.

### Step 3 — Fire a research query

```bash
curl -X POST http://localhost:8000/research \
  -H "Content-Type: application/json" \
  -d '{"query": "pembrolizumab overall survival NSCLC"}'
```

Then open your Langfuse dashboard → Traces. You should see one trace named `research-pipeline` within a few seconds.

---

## Dashboard — What to Look For

### Trace view for one research query

```
research-pipeline                          33.5s   ~$0.003
  CHAIN  supervisor                         3.1s   gpt-4o-mini
  AGENT  specialist_agent                  18.1s   gpt-4o-mini
  CHAIN  synthesizer                        8.2s   gpt-4o
  CHAIN  RunnableSequence                   2-8s
  CHAIN  route_to_agents                    0.0s
```

### Key metrics to monitor

| Metric | What it tells you |
|---|---|
| `specialist_agent` p95 latency | Grader retry loop kicking in — evidence hard to find |
| `synthesizer` token count | Report length — high tokens = long report |
| Cost per trace | Budget tracking — should be $0.002-$0.005 per query |
| `supervisor` latency | Query decomposition — should be consistent ~3s |

### Observed performance (from production runs)

| Component | p50 | p95 |
|---|---|---|
| Full research pipeline | 33.47s | ~35s |
| specialist_agent | 18.10s | 21.75s |
| synthesizer | 8.22s | 8.22s |
| supervisor | 3.08s | 3.08s |

The specialist agent is 55% of total time. p95 variance at 21.75s comes from grader retry loops when the first retrieval attempt fails quality thresholds.

---

## Using Langfuse for RAGAS Evaluation

When running RAGAS evals, pass a shared session ID so all evaluation queries appear grouped in one Langfuse session:

```python
# In run_ragas.py
handler, invoke_config = get_callback_handler(
    session_id = "ragas-eval-v1",
    trace_name = "ragas-evaluation",
    metadata   = {
        "eval_type":   "golden_dataset",
        "n_questions": 42,
        "dataset_version": "v4"
    }
)
output = await run_research(query, invoke_config=invoke_config)
flush_handler(handler)
```

In the Langfuse dashboard you can then filter by `session_id = "ragas-eval-v1"` to see all 42 evaluation runs together — cross-reference which queries produced low RAGAS scores against the exact prompts and retrieved chunks that generated them.

---

## Graceful Degradation

The integration is designed to be fully optional. If `LANGFUSE_PUBLIC_KEY` is not set:

- `langfuse_enabled()` returns `False`
- `get_callback_handler()` returns `(None, {})`
- `run_research(query, invoke_config={})` calls `_graph.ainvoke(state, config={})` — identical to the original
- `flush_handler(None)` is a no-op
- `/health` shows `"langfuse": false` but status remains `"healthy"`

The pipeline is completely unaffected.
