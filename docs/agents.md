# agents.py — Documentation

## Position in Pipeline

```
grader.py (GraderOutput)
        ↓
agents.py   ← YOU ARE HERE
        ↓
mcp_server.py / api/main.py (AgentOutput)
```

---

## Architecture Flow

```
run_research(query)
        │
        ▼
┌─────────────────────────────────────────────────────────────────────┐
│  LangGraph StateGraph                                               │
│                                                                      │
│  GraphState flows through every node:                               │
│    query, supervisor_plan, agent_results, final_output              │
│                                                                      │
│  START → supervisor_node                                            │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  supervisor_node                                                    │
│                                                                      │
│  _SUPERVISOR_CHAIN.ainvoke({query})                                 │
│  → SupervisorPlan(tasks=[AgentTask, ...])                           │
│                                                                      │
│  Examples:                                                           │
│  "pembrolizumab OS NSCLC?"     → [efficacy]                        │
│  "pembro vs nivo survival"     → [efficacy, competitor]             │
│  "pembro side effects?"        → [safety]                           │
│  "safe AND effective review"   → [efficacy, safety]                 │
│                                                                      │
│  Each AgentTask has:                                                 │
│    agent_type:    "efficacy" | "safety" | "mechanism" | "competitor"│
│    focused_query: query rewritten for this agent's domain           │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ supervisor_plan in state
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  route_to_agents()   returns list[Send]                             │
│                                                                      │
│  [Send("specialist_agent", {task: efficacy_task}),                  │
│   Send("specialist_agent", {task: competitor_task})]                │
│                                                                      │
│  LangGraph fires all Send() concurrently                            │
│  Total time = slowest agent, not sum of all agents                  │
└──────────┬────────────────────────────────────────┬─────────────────┘
           │ Send(efficacy)                          │ Send(competitor)
           ▼                                         ▼
┌─────────────────────────┐           ┌─────────────────────────────┐
│  specialist_agent_node  │           │  specialist_agent_node      │
│  task = efficacy        │           │  task = competitor          │
│                         │           │                             │
│  get_agent_filters()    │           │  get_agent_filters()        │
│  evidence_level_max=2   │           │  drug=None                  │
│  year_min=2018          │           │  evidence_level_max=2       │
│  endpoints="OS"         │           │  year_min=2018              │
│         ↓               │           │           ↓                 │
│  retrieve_and_grade()   │           │  retrieve_and_grade()       │
│  full retrieval loop    │           │  full retrieval loop        │
│         ↓               │           │           ↓                 │
│  GraderOutput           │           │  GraderOutput               │
│  verdict + chunks       │           │  verdict + chunks           │
│         ↓               │           │           ↓                 │
│  _format_chunks()       │           │  _format_chunks()           │
│  accepted only — no     │           │  accepted only — no         │
│  hallucination from     │           │  hallucination from         │
│  rejected evidence      │           │  rejected evidence          │
│         ↓               │           │           ↓                 │
│  _format_verdict_note() │           │  _format_verdict_note()     │
│  warning if partial     │           │  warning if partial         │
│         ↓               │           │           ↓                 │
│  LLM → AgentSection     │           │  LLM → AgentSection         │
│  title + content +      │           │  title + content +          │
│  citations + quality    │           │  citations + quality        │
│         ↓               │           │           ↓                 │
│  AgentResult            │           │  AgentResult                │
│  {"agent_results": [R]} │           │  {"agent_results": [R]}     │
└──────────┬──────────────┘           └─────────────┬───────────────┘
           │                                         │
           │   operator.add merges concurrently      │
           └─────────────────┬───────────────────────┘
                             │ [efficacy_result, competitor_result]
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  synthesizer_node   (uses gpt-4o — stronger model)                  │
│                                                                      │
│  _compute_overall_verdict() — worst-case across all agents          │
│    all accepted → "accepted"                                        │
│    any partial  → "partial"                                         │
│    any insufficient → "insufficient"                                │
│                                                                      │
│  _build_evidence_summary() — one-line per agent: verdict + RCT count│
│                                                                      │
│  _SYNTHESIZER_CHAIN.ainvoke()                                       │
│    receives: all sections_text + evidence_summary + overall_verdict  │
│    writes: cohesive markdown report with executive summary          │
│    preserves all citations — does not introduce new claims          │
│                                                                      │
│  → AgentOutput(report, sections, verdict, evidence_summary, query)  │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼ END
                    AgentOutput → mcp_server / api
```

---

## Step-by-Step Process

### Step 1 — Configuration

```python
SYNTHESIS_MODEL  = "gpt-4o"        # synthesis requires long-form coherent writing
AGENT_MODEL      = "gpt-4o-mini"   # extraction tasks — cheaper model adequate
SUPERVISOR_MODEL = "gpt-4o-mini"   # deciding which agents — simpler task
```

Three different models for three different cognitive tasks. The synthesizer receives multiple evidence sections and must produce a coherent long-form report with consistent citations — qualitatively harder than extraction. `gpt-4o-mini` handles structured extraction reliably at lower cost.

---

### Step 2 — Pydantic models (all from LLM)

**`AgentTask`** — one specialist task assigned by the supervisor:
- `agent_type` — constrained by `Literal["efficacy", "safety", "mechanism", "competitor"]`
- `focused_query` — query rewritten for this agent's domain (e.g. safety agent gets "adverse events" version)
- `validate_tasks` validator deduplicates by `agent_type` — supervisor should not activate the same agent twice

**`SupervisorPlan`** — the supervisor's full decomposition. The `validate_tasks` validator also caps at 4 agents.

**`AgentSection`** — one agent's written output. `min_length=50` on `content` prevents one-liner responses. `verdict_note` field auto-populated when evidence is limited — forces the LLM to acknowledge it. All three are Pydantic because they come from LLM generation (untrusted).

---

### Step 3 — Dataclasses (internal trusted state)

**`AgentResult`** — wraps `AgentSection` (Pydantic, from LLM) with `GraderOutput` (dataclass, from grader). Built by `run_specialist_agent()` — trusted internal code.

**`AgentOutput`** — the final deliverable. Built by `synthesizer_node()`. Contains the markdown `report`, all `sections`, `overall_verdict`, `evidence_summary`. Converted to a Pydantic `AgentResponse` model in `api/main.py` for JSON serialization at the API boundary.

Both are dataclasses because they are constructed by trusted code from already-validated inputs.

---

### Step 4 — `GraphState` and `operator.add`

```python
class GraphState(dict):
    query:           str
    supervisor_plan: SupervisorPlan | None
    agent_results:   Annotated[list[AgentResult], operator.add]
    final_output:    AgentOutput | None
```

**`Annotated[list[AgentResult], operator.add]`** — the critical LangGraph pattern for concurrent writes. When multiple agents run in parallel via `Send()`, each returns `{"agent_results": [result]}`. Without `operator.add`, the second agent's result overwrites the first. With `operator.add`, LangGraph knows to concatenate (Python list `+`) instead:

```
EfficacyAgent returns: {"agent_results": [efficacy_result]}
CompetitorAgent returns: {"agent_results": [competitor_result]}

LangGraph merges: agent_results = [efficacy_result, competitor_result]
```

---

### Step 5 — LLM chains

All chains built at module load time — module-level constants, not recreated per query.

```python
_structured_supervisor = _supervisor_llm.with_structured_output(SupervisorPlan)
_structured_section    = _agent_llm.with_structured_output(AgentSection)

_AGENT_CHAINS = {
    agent_type: prompt | _structured_section
    for agent_type, prompt in _AGENT_PROMPTS.items()
}
```

Four agent chains built in one dict comprehension. All four ready at import. The synthesizer chain uses `_synthesis_llm` without `with_structured_output()` — it produces free-form markdown, not a structured schema.

---

### Step 6 — Supervisor node

Reads the user query and produces a `SupervisorPlan`. The system prompt gives explicit activation rules:

```
Simple efficacy question → [efficacy] only
Comparison question      → [efficacy, competitor]
Comprehensive review     → [efficacy, safety] minimum
NEVER activate all 4 unless every domain explicitly needed
```

If the chain fails (network error, LLM refusal), a fallback plan activates the efficacy agent with the original query — the system always does something useful rather than crashing.

---

### Step 7 — `get_agent_filters()` — the clinical heart

```python
match agent_type:
    case "efficacy":
        return MetadataFilter(
            evidence_level_max = 2,    # exclude case reports
            year_min           = 2018, # recent trials
            endpoints          = "OS", # survival focus
        )
    case "safety":
        return MetadataFilter(
            endpoints = "AE",
            # NO evidence_level_max — case reports valid for rare AEs
            # NO year_min — older safety reports still relevant
        )
    case "competitor":
        return MetadataFilter(
            drug               = None, # search across drugs
            evidence_level_max = 2,
        )
```

Each agent type has clinically different evidence requirements that cannot coexist in one filter:

| Agent | Evidence level | Year filter | Why |
|---|---|---|---|
| Efficacy | ≤ 2 (no case reports) | ≥ 2018 | Efficacy must come from controlled trials |
| Safety | None (case reports valid) | None | Rare AEs only appear in case reports |
| Mechanism | None | None | Biology doesn't expire |
| Competitor | ≤ 2 | ≥ 2018 | Comparison needs reliable evidence; drug=None |

Python 3.13 `match/case` — four discrete cases with different return values. Cleaner than four `if/elif` blocks.

---

### Step 8 — Agent prompts and `_format_verdict_note()`

```python
_AGENT_SYSTEM_TEMPLATES = {
    "efficacy":   "...OS, PFS, ORR outcomes — be specific about hazard ratios...",
    "safety":     "...Case reports valid for rare AEs — note when from single cases...",
    "mechanism":  "...Explain PD-1/PD-L1 pathway involvement specifically...",
    "competitor": "...Note if comparison is direct (RCT) or indirect (NMA)...",
}
```

Each agent type has domain-specific grounding rules in its system prompt. The safety analyst is told case reports are valid — the efficacy analyst is not. This distinction exists for clinical reasons encoded in the prompt.

**`_format_verdict_note()`** — the feedback channel from grader to agent:

```python
if grader_output.verdict == "accepted":  return ""    # no warning needed
if grader_output.verdict == "partial":
    return "WARNING: Only 2 chunks passed. Use 'limited evidence suggests...'"
return "WARNING: No chunks passed. Do NOT generate clinical claims without evidence."
```

Injected into `{verdict_note}` in the system prompt. Forces the LLM to write hedged language when evidence is limited — "limited evidence suggests" not "demonstrates conclusively."

**`_format_chunks_for_prompt()`** — only accepted chunks included:

```
[1] Drug: pembrolizumab | Cancer: nsclc | Study: RCT | N=1,966
    Journal: NEJM (2023) | PMID: 38234521 | Endpoints: OS,PFS
    Content: <first 600 chars>
```

Rejected chunks are completely excluded. The agent cannot hallucinate based on evidence the grader rejected — it simply never sees it.

---

### Step 9 — `run_specialist_agent()`

Five sequential steps:

1. `get_agent_filters(task.agent_type)` — preset filters for this role
2. `retrieve_and_grade(focused_query, agent_filters)` — full pipeline
3. `_format_chunks_for_prompt(grader_output)` — accepted chunks only
4. `chain.ainvoke({formatted_chunks, verdict_note, ...})` → `AgentSection`
5. Auto-generate citations if LLM omitted them

Auto-citation fallback at step 5 — if the LLM generated good clinical content but forgot to format citations, we build them from chunk metadata:
```python
section.citations = [
    f"[PMID: {g.chunk.source}] {g.chunk.journal} ({g.chunk.year}) — {g.chunk.study_type}"
    for g in grader_output.accepted_chunks
]
```

---

### Step 10 — `route_to_agents()` and `Send()`

```python
def route_to_agents(state: GraphState) -> list[Send]:
    return [
        Send("specialist_agent", {**state, "task": task})
        for task in plan.tasks
    ]
```

Returns a list of `Send` objects — one per activated agent. LangGraph fires all concurrently. Each `Send` injects its specific `AgentTask` into the receiving node's state. There is only one `specialist_agent` node — the injected `task` tells it which agent to behave as.

**Why `Send()` not static edges:** Static edges always run all 4 agents. `Send()` runs only what the supervisor chose. A simple efficacy question activating all 4 agents wastes 3 API calls and adds unnecessary latency.

---

### Step 11 — Synthesizer node

**`_compute_overall_verdict()`** — worst-case logic:
```python
verdicts = [r.grader_output.verdict for r in agent_results]
if "insufficient" in verdicts: return "insufficient"
if "partial" in verdicts:      return "partial"
return "accepted"
```

One agent with no evidence makes the whole report insufficient. A clinician cannot make a complete treatment decision when safety evidence is missing even if efficacy evidence is comprehensive. Worst-case is the clinically correct choice.

**`_build_evidence_summary()`** — one line per agent:
```
EFFICACY: verdict=accepted | chunks=4 | RCTs=3 | attempts=1
SAFETY: verdict=partial | chunks=2 | RCTs=1 | attempts=2
```

Passed to the synthesizer prompt — gives the synthesizer explicit context about evidence quality across all domains.

**Why `gpt-4o` for synthesis:** The synthesizer must write a long-form coherent report across multiple evidence domains, maintain consistent citation format, avoid introducing new claims not in any section, and make nuanced judgments when sections conflict. The quality difference between `gpt-4o-mini` and `gpt-4o` on long-form synthesis is material.

---

### Step 12 — Graph construction

```python
graph = StateGraph(GraphState)
graph.add_node("supervisor",       supervisor_node)
graph.add_node("specialist_agent", specialist_agent_node)
graph.add_node("synthesizer",      synthesizer_node)

graph.add_edge(START,              "supervisor")
graph.add_edge("specialist_agent", "synthesizer")
graph.add_conditional_edges("supervisor", route_to_agents, ["specialist_agent"])
graph.add_edge("synthesizer", END)

return graph.compile()
```

`graph.compile()` validates the graph structure — no dead ends, all referenced nodes exist — and returns a compiled `Runnable`. Compiled once at module load as `_graph` — not rebuilt per query.

**Why `StateGraph` not `MessageGraph`:** We pass structured dataclasses (`AgentResult`, `SupervisorPlan`) not just chat messages. `StateGraph` supports arbitrary typed state. `MessageGraph` is for conversation-style LLM applications.

---

### Step 13 — `run_research()` public API

```python
async def run_research(query: str) -> AgentOutput:
    initial_state = {
        "query":           query,
        "supervisor_plan": None,
        "agent_results":   [],
        "final_output":    None,
    }
    result = await _graph.ainvoke(initial_state)
    return result["final_output"]
```

One function, one argument, one return value. The entire pipeline — supervisor, parallel agents, synthesizer — happens inside `_graph.ainvoke()`. Called by `mcp_server.py` and `api/main.py`.

---

## Data contracts

| Input | Source | Purpose |
|---|---|---|
| `query` | user / API | raw clinical question |

| Output | Type | Consumer |
|---|---|---|
| `report` | `str` markdown | API response / MCP tool |
| `sections` | `list[AgentSection]` | structured API access |
| `overall_verdict` | `Literal[...]` | API response metadata |
| `evidence_summary` | `str` | logging / display |
| `agent_results` | `list[AgentResult]` | debugging / eval |

---

## Key decisions

**Why specialist agents over one general agent:** Efficacy requires `evidence_level_max=2` (no case reports). Safety requires no evidence level filter (case reports valid for rare AEs). These requirements are mutually exclusive for the same filter. Specialist agents retrieve independently with domain-appropriate filters.

**Why `operator.add` on `agent_results`:** Enables truly concurrent parallel agent execution. Without it, the second agent's result would overwrite the first — losing evidence from the first agent silently.

**Why the synthesizer does not use `with_structured_output()`:** The final report is free-form markdown — a coherent long-form document. Constraining it to a Pydantic schema would limit the writing quality. The synthesizer's system prompt provides the structure rules instead of schema constraints.

**Why `verdict_note` flows through from grader to synthesizer:** Evidence quality must propagate into the final report language. The chain: `GraderOutput.verdict` → `_format_verdict_note()` → injected into agent system prompt → agent writes hedged content → synthesizer receives hedged sections → final report reflects the limitations. Breaking this chain at any point would produce overconfident reports from weak evidence.

