"""
agents.py — LangGraph Multi-Agent Orchestration
=================================================
Python 3.13 | LangChain + LangGraph throughout

Position in pipeline:
  grader.py → agents.py → mcp_server.py / api/main.py

What it does:
  Orchestrates multiple specialist agents via LangGraph to produce
  a comprehensive, cited clinical research brief from a user query.

  The supervisor node reads the query and dynamically decides which
  specialist agents to activate. Each agent independently retrieves
  and grades evidence for its domain. The synthesizer merges all
  agent outputs into one cohesive markdown report.

Architecture:
  SupervisorNode
    → Send() API dispatches to specialist agents in parallel
      → EfficacyAgent   (OS, PFS, ORR from strong clinical trials)
      → SafetyAgent      (AE, toxicity including case reports)
      → MechanismAgent   (how the drug works — biology/pharmacology)
      → CompetitorAgent  (head-to-head comparisons, alternatives)
    → SynthesizerNode merges all sections → AgentOutput

Why LangGraph:
  - Send() API enables dynamic parallel dispatch — supervisor decides
    at runtime which agents to activate based on query content
  - StateGraph maintains typed state across all nodes
  - Built-in checkpointing for fault tolerance
  - Native async support — agents run concurrently

Why specialist agents instead of one general agent:
  Each agent has preset MetadataFilter defaults tuned for its role.
  EfficacyAgent: evidence_level_max=2 (no case reports), year_min=2018
  SafetyAgent:   no evidence filter (rare AEs appear in case reports)
  These presets cannot coexist in one agent for one query.
  Specialist agents retrieve independently and concurrently.

LangChain usage:
  ChatOpenAI with_structured_output — supervisor plan + agent sections
  ChatPromptTemplate — all prompts versioned and typed
  All calls traced in LangSmith automatically
"""

import asyncio
import re
from dataclasses import dataclass, field
from typing import Literal, Annotated
import operator

from pydantic import BaseModel, Field, field_validator
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from dotenv import load_dotenv

from retriever import MetadataFilter
from grader import GraderOutput, GradedChunk, retrieve_and_grade
from langfuse_setup import get_callback_handler  # noqa: F401 — LANGFUSE CHANGE 1 of 2

load_dotenv()


# ─────────────────────────────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────────────────────────────

SYNTHESIS_MODEL  = "gpt-4o"        # synthesizer uses stronger model
AGENT_MODEL      = "gpt-4o-mini"   # specialist agents use cheaper model
SUPERVISOR_MODEL = "gpt-4o-mini"   # supervisor uses cheaper model

# Agent type literals — controls which agents the supervisor activates
AgentType = Literal["efficacy", "safety", "mechanism", "competitor"]


# ─────────────────────────────────────────────────────────────────────
# 2. PYDANTIC MODELS — LLM outputs (untrusted → validated)
# ─────────────────────────────────────────────────────────────────────

class AgentTask(BaseModel):
    """
    One specialist agent task — assigned by the supervisor.
    Supervisor produces a list of these, one per agent to activate.

    agent_type: which specialist handles this task
    focused_query: query rewritten for this agent's domain
      e.g. for EfficacyAgent: "pembrolizumab overall survival NSCLC"
      e.g. for SafetyAgent:   "pembrolizumab adverse events toxicity NSCLC"
    rationale: why this agent is needed for the user's query
    """
    agent_type:    Literal["efficacy", "safety", "mechanism", "competitor"]
    focused_query: str = Field(
        ...,
        min_length=5,
        description=(
            "Focused version of the query for this specific agent's domain. "
            "EfficacyAgent: include outcome (OS, PFS, ORR) explicitly. "
            "SafetyAgent: include 'adverse events' or 'toxicity' or 'safety'. "
            "MechanismAgent: include 'mechanism' or 'how does' or 'pathway'. "
            "CompetitorAgent: include both drug names if comparing."
        )
    )
    rationale: str = Field(
        default="",
        description="One sentence explaining why this agent is needed."
    )

    @field_validator("focused_query", mode="before")
    @classmethod
    def clean_query(cls, v: object) -> str:
        if not v or not str(v).strip():
            raise ValueError("focused_query cannot be empty")
        return str(v).strip()


class SupervisorPlan(BaseModel):
    """
    Supervisor's decomposition of the user query into agent tasks.
    Schema passed to with_structured_output() — LLM constrained to
    return exactly this structure.

    tasks: list of AgentTask — one per agent to activate (1-4)
    overall_reasoning: supervisor's explanation of its decomposition
    """
    tasks: list[AgentTask] = Field(
        ...,
        min_length=1,
        description=(
            "List of specialist agent tasks to run. "
            "Activate only agents relevant to the query: "
            "Simple efficacy question → [efficacy] only. "
            "Safety question → [safety] only. "
            "Comparison question → [efficacy, competitor]. "
            "Comprehensive research → [efficacy, safety, mechanism]. "
            "Never activate all 4 unless the query explicitly needs all domains."
        )
    )
    overall_reasoning: str = Field(
        default="",
        description="One sentence explaining the decomposition strategy."
    )

    @field_validator("tasks", mode="before")
    @classmethod
    def validate_tasks(cls, v: object) -> list:
        if not isinstance(v, list) or len(v) == 0:
            raise ValueError("tasks must be a non-empty list")
        # Deduplicate by agent_type — supervisor should not assign same agent twice
        seen: set[str] = set()
        deduped = []
        for task in v:
            agent_type = task.get("agent_type") if isinstance(task, dict) else getattr(task, "agent_type", None)
            if agent_type and agent_type not in seen:
                seen.add(agent_type)
                deduped.append(task)
        return deduped[:4]   # cap at 4 agents


class AgentSection(BaseModel):
    """
    One specialist agent's output — a section of the final report.
    Pydantic because it comes from LLM generation (untrusted).

    title: section heading (e.g. "Efficacy Evidence")
    content: markdown-formatted clinical findings with inline citations
    citations: list of citation strings extracted from accepted_chunks
    evidence_quality: summary of evidence strength for this section
    limitations: what evidence was missing or weak
    verdict_note: populated when GraderOutput.verdict != "accepted"
    """
    title: str = Field(
        ...,
        description="Section heading. E.g. 'Efficacy Evidence', 'Safety Profile'."
    )
    content: str = Field(
        ...,
        min_length=50,
        description=(
            "Markdown-formatted clinical findings. "
            "Ground every claim in the provided chunks — no hallucination. "
            "Use inline citations: [PMID: 38234521] or [KEYNOTE-189, NEJM 2023]. "
            "If verdict is partial/insufficient, acknowledge limited evidence explicitly."
        )
    )
    citations: list[str] = Field(
        default_factory=list,
        description=(
            "List of citation strings. Format: "
            "'[PMID: {{source}}] {{journal}} ({{year}}) — {study_type}' "
            "One citation per accepted chunk used."
        )
    )
    evidence_quality: str = Field(
        default="",
        description=(
            "One sentence summarizing evidence quality. "
            "E.g. 'Based on 3 RCTs (n=1,966 to n=2,000) from 2018-2023.'"
        )
    )
    limitations: str = Field(
        default="",
        description=(
            "Key gaps in retrieved evidence. "
            "E.g. 'No direct comparison with nivolumab found in knowledge base.'"
        )
    )
    verdict_note: str = Field(
        default="",
        description=(
            "Populated only when verdict is partial or insufficient. "
            "Explains why evidence is limited and how to interpret the section."
        )
    )

    @field_validator("content", mode="before")
    @classmethod
    def clean_content(cls, v: object) -> str:
        if not v or not str(v).strip():
            raise ValueError("content cannot be empty")
        return str(v).strip()

    @field_validator("citations", mode="before")
    @classmethod
    def clean_citations(cls, v: object) -> list[str]:
        if not isinstance(v, list):
            return []
        return [str(c).strip() for c in v if c and str(c).strip()]


# ─────────────────────────────────────────────────────────────────────
# 3. AGENT OUTPUT DATACLASSES — internal trusted state
# ─────────────────────────────────────────────────────────────────────

@dataclass
class AgentResult:
    """
    One specialist agent's complete result.
    Dataclass — constructed by trusted agent code, not from LLM JSON directly.
    The AgentSection (Pydantic) lives inside it — that part came from LLM.

    agent_type: which agent produced this
    section: the LLM-generated, Pydantic-validated content section
    grader_output: the GraderOutput from retrieve_and_grade()
                   agents.py passes this to synthesizer for evidence weighting
    """
    agent_type:     str
    section:        AgentSection
    grader_output:  GraderOutput


@dataclass
class AgentOutput:
    """
    Complete final output from the agent graph.
    Passed to api/main.py for the API response.
    Dataclass — assembled by our synthesizer node from trusted internal state.

    report: full markdown research brief (synthesized from all sections)
    sections: individual agent sections (for structured API access)
    overall_verdict: worst verdict across all agents
      "accepted" → all agents found good evidence
      "partial"  → at least one agent had limited evidence
      "insufficient" → at least one agent found no evidence
    evidence_summary: one paragraph summarizing evidence across all agents
    query: the original user query
    """
    report:           str
    sections:         list[AgentSection]
    overall_verdict:  Literal["accepted", "partial", "insufficient"]
    evidence_summary: str
    query:            str
    agent_results:    list[AgentResult] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────
# 4. LANGGRAPH STATE
# ─────────────────────────────────────────────────────────────────────

class GraphState(dict):
    """
    LangGraph state — flows through every node in the graph.
    Uses TypedDict-style annotations for LangGraph compatibility.

    query:            original user query
    supervisor_plan:  SupervisorPlan from supervisor node
    agent_results:    list of AgentResult — one per activated agent
                      Annotated[list, operator.add] enables concurrent
                      appending from parallel agent nodes via Send()
    final_output:     AgentOutput from synthesizer node
    """
    query:           str
    supervisor_plan: SupervisorPlan | None
    agent_results:   Annotated[list[AgentResult], operator.add]
    final_output:    AgentOutput | None


# ─────────────────────────────────────────────────────────────────────
# 5. LLM INSTANCES
# ─────────────────────────────────────────────────────────────────────

_supervisor_llm = ChatOpenAI(
    model=SUPERVISOR_MODEL,
    temperature=0,
    max_tokens=600,
)

_agent_llm = ChatOpenAI(
    model=AGENT_MODEL,
    temperature=0,
    max_tokens=1000,
)

_synthesis_llm = ChatOpenAI(
    model=SYNTHESIS_MODEL,
    temperature=0,
    max_tokens=2000,    # longer output — full report
)

# Structured output chains
_structured_supervisor = _supervisor_llm.with_structured_output(SupervisorPlan)
_structured_section    = _agent_llm.with_structured_output(AgentSection)


# ─────────────────────────────────────────────────────────────────────
# 6. SUPERVISOR NODE
# ─────────────────────────────────────────────────────────────────────

_SUPERVISOR_SYSTEM = """You are a clinical research query decomposition specialist.
Your job: decompose a user's pharma/clinical query into focused tasks
for specialist agents. Activate only the agents genuinely needed.

Available specialist agents:
  efficacy    — finds clinical efficacy evidence (OS, PFS, ORR, DFS)
                use for: survival benefit, response rate, trial results
  safety      — finds adverse event and safety data
                use for: side effects, toxicity, immune-related events
  mechanism   — finds pharmacological mechanism evidence
                use for: how the drug works, pathway, biology
  competitor  — finds comparative and head-to-head evidence
                use for: drug comparisons, alternatives, versus queries

Activation rules:
  - Simple efficacy question → [efficacy] only
  - Simple safety question   → [safety] only
  - Mechanism question       → [mechanism] only
  - Comparison question      → [efficacy, competitor]
  - Comprehensive question   → [efficacy, safety] minimum
  - NEVER activate all 4 unless every domain is explicitly needed

Focused query per agent:
  Each agent gets a query rewritten for its domain.
  EfficacyAgent focused query must include the clinical outcome explicitly.
  SafetyAgent focused query must include adverse events or safety or toxicity.
  Always use full drug names (pembrolizumab, not pembro)."""

_SUPERVISOR_HUMAN = "User query: {query}"

_SUPERVISOR_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _SUPERVISOR_SYSTEM),
    ("human",  _SUPERVISOR_HUMAN),
])

_SUPERVISOR_CHAIN = _SUPERVISOR_PROMPT | _structured_supervisor


async def supervisor_node(state: GraphState) -> dict:
    """
    Supervisor node — entry point of the LangGraph graph.
    Reads the user query and produces a SupervisorPlan.
    The plan determines which agents are activated via Send().

    Returns state update: {"supervisor_plan": SupervisorPlan}
    """
    query = state["query"]
    print(f"\nSupervisor: decomposing query '{query[:60]}...'")

    try:
        plan: SupervisorPlan = await _SUPERVISOR_CHAIN.ainvoke({"query": query})
        print(f"Supervisor: activating {len(plan.tasks)} agents: "
              f"{[t.agent_type for t in plan.tasks]}")
        print(f"Supervisor reasoning: {plan.overall_reasoning}")
        return {"supervisor_plan": plan}

    except Exception as e:
        print(f"Supervisor failed: {e}. Falling back to efficacy only.")
        fallback_plan = SupervisorPlan(
            tasks=[AgentTask(
                agent_type    = "efficacy",
                focused_query = query,
                rationale     = "Fallback — supervisor failed",
            )],
            overall_reasoning = "Supervisor error — defaulting to efficacy agent.",
        )
        return {"supervisor_plan": fallback_plan}


# ─────────────────────────────────────────────────────────────────────
# 7. SPECIALIST AGENT FILTERS — preset defaults per role
# ─────────────────────────────────────────────────────────────────────

def get_agent_filters(
    agent_type:   str,
    drug:         str | None = None,
    cancer_type:  str | None = None,
) -> MetadataFilter:
    """
    Returns preset MetadataFilter for each specialist agent type.

    Why presets matter:
      Each agent's role determines what evidence is appropriate.
      EfficacyAgent must exclude case reports — a single patient
      case report is not valid evidence for efficacy claims.
      SafetyAgent must INCLUDE case reports — rare adverse events
      (1 in 10,000) will never appear in large trials.
      MechanismAgent cares about drug and biology, not study design.

    These presets are overridden by any explicitly provided drug/cancer_type
    from the supervisor's focused query extraction.

    EfficacyAgent:
      evidence_level_max=2 — exclude case reports (level 3)
      year_min=2018         — recent clinical trials only
      endpoints=OS          — survival is primary outcome for efficacy

    SafetyAgent:
      No evidence_level_max — case reports valid for rare AEs
      No year_min           — older AE reports still relevant
      endpoints=AE          — adverse events specifically

    MechanismAgent:
      No evidence_level filter — review articles useful for mechanism
      No year_min             — foundational biology doesn't expire

    CompetitorAgent:
      drug=None              — must search across drugs (no single drug filter)
      evidence_level_max=2   — comparison needs reliable evidence
    """
    base = MetadataFilter(drug=drug, cancer_type=cancer_type)

    match agent_type:
        case "efficacy":
            return MetadataFilter(
                drug               = drug,
                cancer_type        = cancer_type,
                evidence_level_max = 2,
                year_min           = 2018,
                endpoints          = "OS",
            )
        case "safety":
            return MetadataFilter(
                drug        = drug,
                cancer_type = cancer_type,
                endpoints   = "AE",
                # No evidence_level_max — case reports valid for rare AEs
                # No year_min — older safety reports still relevant
            )
        case "mechanism":
            return MetadataFilter(
                drug        = drug,
                cancer_type = cancer_type,
                # No evidence_level filter — reviews valid for mechanism
                # No year_min — biology doesn't expire quickly
            )
        case "competitor":
            return MetadataFilter(
                drug               = None,   # search across drugs
                cancer_type        = cancer_type,
                evidence_level_max = 2,
                year_min           = 2018,
            )
        case _:
            return base


# ─────────────────────────────────────────────────────────────────────
# 8. SPECIALIST AGENT PROMPTS
# ─────────────────────────────────────────────────────────────────────

def _format_chunks_for_prompt(grader_output: GraderOutput) -> str:
    """
    Formats accepted chunks into a numbered evidence list for the agent prompt.
    Only accepted chunks are included — graded and validated evidence only.
    Rejected chunks are explicitly excluded — no hallucination from weak evidence.

    Format per chunk:
      [1] Drug: pembrolizumab | Cancer: nsclc | Study: RCT | N=1,966
          Journal: NEJM (2023) | PMID: 38234521 | Endpoints: OS,PFS
          Content: <first 600 chars of chunk content>
    """
    if not grader_output.accepted_chunks:
        return "No accepted evidence chunks available."

    lines = []
    for i, g in enumerate(grader_output.accepted_chunks, 1):
        c = g.chunk
        lines.append(
            f"[{i}] Drug: {c.drug} | Cancer: {c.cancer_type} | "
            f"Study: {c.study_type} | N={c.sample_size or 'NR'}\n"
            f"    Journal: {c.journal} ({c.year}) | PMID: {c.source}\n"
            f"    Endpoints: {c.endpoints or 'NR'} | "
            f"Evidence level: {c.evidence_level}\n"
            f"    Content: {c.content[:600]}"
        )

    return "\n\n".join(lines)


def _format_verdict_note(grader_output: GraderOutput) -> str:
    """
    Returns a warning string for the agent prompt when evidence is limited.
    Empty string when verdict is accepted — no warning needed.
    """
    if grader_output.verdict == "accepted":
        return ""
    if grader_output.verdict == "partial":
        return (
            f"\nWARNING: Only {len(grader_output.accepted_chunks)} chunk(s) "
            f"passed quality grading (minimum: 3). "
            f"Acknowledge limited evidence in your section. "
            f"Use hedging language: 'limited evidence suggests', 'based on "
            f"small studies'. Do NOT overstate findings.\n"
        )
    return (
        "\nWARNING: No chunks passed quality grading. "
        "State clearly that no sufficient evidence was found in the "
        "knowledge base for this specific query. "
        "Do NOT generate clinical claims without evidence.\n"
    )


_AGENT_SYSTEM_TEMPLATES: dict[str, str] = {
    "efficacy": """You are a clinical efficacy evidence analyst for a pharma RAG system.
Your role: synthesize retrieved clinical trial evidence on drug efficacy outcomes.

Ground rules:
- Base every claim on the provided evidence chunks ONLY
- Use inline citations: [PMID: {{source}}] or trial name + journal + year
- Focus on OS, PFS, ORR outcomes — be specific about hazard ratios, medians
- Note evidence_level and sample_size when citing — they signal strength
- If evidence is limited (warned below), acknowledge this explicitly{verdict_note}""",

    "safety": """You are a clinical safety evidence analyst for a pharma RAG system.
Your role: synthesize retrieved evidence on drug adverse events and safety profile.

Ground rules:
- Base every claim on the provided evidence chunks ONLY
- Include both common AEs (rash, fatigue) and rare but serious AEs (pneumonitis)
- For checkpoint inhibitors: always address immune-related adverse events (irAEs)
- Specify grades where reported (Grade 3-4 AEs are clinically significant)
- Case reports are valid for rare AEs — note when evidence is from single cases{verdict_note}""",

    "mechanism": """You are a clinical pharmacology analyst for a pharma RAG system.
Your role: synthesize retrieved evidence on drug mechanism of action and biology.

Ground rules:
- Base every claim on the provided evidence chunks ONLY
- Explain molecular/cellular mechanism clearly — assume reader has medical background
- Connect mechanism to clinical outcomes where evidence supports it
- For checkpoint inhibitors: explain PD-1/PD-L1 pathway involvement specifically{verdict_note}""",

    "competitor": """You are a comparative clinical evidence analyst for a pharma RAG system.
Your role: synthesize retrieved head-to-head comparison evidence between drugs.

Ground rules:
- Base every claim on the provided evidence chunks ONLY
- Be precise about which comparison the evidence supports
- Note if comparison is direct (RCT) or indirect (network meta-analysis)
- Acknowledge when no direct comparison exists in retrieved evidence{verdict_note}""",
}

_AGENT_HUMAN = """Query for this section: {focused_query}

Retrieved evidence chunks ({n_chunks} accepted, {n_total} total retrieved):
{formatted_chunks}

Write your section. Title it appropriately for your role."""

_AGENT_PROMPTS: dict[str, ChatPromptTemplate] = {
    agent_type: ChatPromptTemplate.from_messages([
        ("system", template),
        ("human",  _AGENT_HUMAN),
    ])
    for agent_type, template in _AGENT_SYSTEM_TEMPLATES.items()
}

_AGENT_CHAINS: dict[str, object] = {
    agent_type: prompt | _structured_section
    for agent_type, prompt in _AGENT_PROMPTS.items()
}


# ─────────────────────────────────────────────────────────────────────
# 9. SPECIALIST AGENT NODE
# ─────────────────────────────────────────────────────────────────────

async def run_specialist_agent(task: AgentTask) -> AgentResult:
    """
    Runs one specialist agent end-to-end:
      1. Build preset filters for this agent type
      2. Extract drug/cancer from focused_query via query rewriter
      3. Call retrieve_and_grade() — full retrieval + grading loop
      4. Format evidence for the agent prompt
      5. Call LLM to generate AgentSection
      6. Return AgentResult

    Called via LangGraph Send() — each activated agent runs this
    function independently and concurrently.

    Falls back to an empty AgentSection if LLM generation fails —
    the synthesizer handles missing sections gracefully.
    """
    print(f"\n  [{task.agent_type.upper()}] Starting: '{task.focused_query[:50]}...'")

    # Step 1+2: retrieve and grade evidence
    # get_agent_filters provides sensible defaults; the rewriter inside
    # retrieve_and_grade() will extract drug/cancer from focused_query
    agent_filters = get_agent_filters(task.agent_type)
    grader_output = await retrieve_and_grade(
        query   = task.focused_query,
        filters = agent_filters,
    )

    print(f"  [{task.agent_type.upper()}] Verdict: {grader_output.verdict} "
          f"| Accepted: {len(grader_output.accepted_chunks)} chunks "
          f"| Attempts: {grader_output.attempts}")

    # Step 3: format evidence for prompt
    formatted_chunks = _format_chunks_for_prompt(grader_output)
    verdict_note     = _format_verdict_note(grader_output)

    # Step 4: generate section via LLM
    chain = _AGENT_CHAINS[task.agent_type]

    try:
        section: AgentSection = await chain.ainvoke({
            "focused_query":    task.focused_query,
            "n_chunks":         len(grader_output.accepted_chunks),
            "n_total":          len(grader_output.all_graded_chunks),
            "formatted_chunks": formatted_chunks,
            "verdict_note":     verdict_note,
        })

        # Auto-generate citations from accepted chunks if LLM didn't
        if not section.citations and grader_output.accepted_chunks:
            section.citations = [
                f"[PMID: {g.chunk.source}] {g.chunk.journal} "
                f"({g.chunk.year}) — {g.chunk.study_type}"
                for g in grader_output.accepted_chunks
            ]

        # Add verdict note to section if evidence was limited
        if grader_output.verdict != "accepted" and not section.verdict_note:
            section.verdict_note = grader_output.failure_reason

        print(f"  [{task.agent_type.upper()}] Section generated: "
              f"'{section.title}' | {len(section.citations)} citations")

    except Exception as e:
        print(f"  [{task.agent_type.upper()}] Section generation failed: {e}")
        section = AgentSection(
            title             = f"{task.agent_type.title()} Evidence",
            content           = (
                f"Evidence generation failed for this section: {type(e).__name__}. "
                f"Grader verdict: {grader_output.verdict}."
            ),
            evidence_quality  = "Generation error",
            limitations       = str(e),
        )

    return AgentResult(
        agent_type    = task.agent_type,
        section       = section,
        grader_output = grader_output,
    )


async def specialist_agent_node(state: GraphState) -> dict:
    """
    LangGraph node wrapper for specialist agent execution.
    Receives individual AgentTask via Send() from the router.
    Returns state update that appends AgentResult to agent_results.

    The Annotated[list, operator.add] on agent_results means
    multiple concurrent agents can each append their result safely.
    """
    task   = state["task"]   # injected by Send()
    result = await run_specialist_agent(task)
    return {"agent_results": [result]}


# ─────────────────────────────────────────────────────────────────────
# 10. ROUTER — supervisor plan → Send() to agents
# ─────────────────────────────────────────────────────────────────────

def route_to_agents(state: GraphState) -> list[Send]:
    """
    Router function — reads supervisor_plan and dispatches each task
    to the specialist_agent_node via LangGraph Send() API.

    Send() enables dynamic parallel dispatch:
      - Supervisor decides at runtime which agents are needed
      - All activated agents run concurrently (asyncio)
      - Each agent receives its own AgentTask via state["task"]
      - Results accumulate in state["agent_results"] via operator.add

    Why Send() instead of static edges:
      Static edges would always run all 4 agents regardless of query.
      Send() dispatches only what the supervisor chose — efficient and
      appropriate. A simple efficacy question doesn't need the mechanism agent.
    """
    plan = state.get("supervisor_plan")
    if not plan or not plan.tasks:
        # Fallback if supervisor produced no plan
        return [Send("specialist_agent", {
            **state,
            "task": AgentTask(
                agent_type    = "efficacy",
                focused_query = state["query"],
                rationale     = "Fallback — no supervisor plan",
            )
        })]

    return [
        Send("specialist_agent", {**state, "task": task})
        for task in plan.tasks
    ]


# ─────────────────────────────────────────────────────────────────────
# 11. SYNTHESIZER NODE
# ─────────────────────────────────────────────────────────────────────

_SYNTHESIZER_SYSTEM = """You are a clinical research report synthesizer.
You receive sections from multiple specialist agents and merge them into
one cohesive, comprehensive research brief.

Rules:
- Do NOT include inline citations in the prose (no [PMID: XXXXX] or [Trial, Journal Year] in body text)
- Write clean, evidence-based narrative — all sources are listed in the References section appended after the report
- Do not introduce new clinical claims not present in any section
- If any section has limited evidence, reflect that in the overall report
- Write in a structured markdown format with clear section headings
- Add an executive summary at the top (3-5 sentences)
- Add a limitations section at the end combining all agent limitations"""

_SYNTHESIZER_HUMAN = """Original user query: {query}

Sections to synthesize:
{sections_text}

Evidence quality summary across all agents:
{evidence_summary}

Overall evidence verdict: {overall_verdict}

Write the complete research brief."""

_SYNTHESIZER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _SYNTHESIZER_SYSTEM),
    ("human",  _SYNTHESIZER_HUMAN),
])

_SYNTHESIZER_CHAIN = _SYNTHESIZER_PROMPT | _synthesis_llm


def _compute_overall_verdict(
    agent_results: list[AgentResult],
) -> Literal["accepted", "partial", "insufficient"]:
    """
    Computes the worst verdict across all agents.
    One insufficient agent makes the whole output insufficient.
    One partial agent makes the whole output partial.
    All accepted → accepted.

    Why worst-case: if the safety section has no evidence but efficacy
    does, a clinician cannot make a complete treatment decision. The
    report should reflect that safety evidence is missing.
    """
    verdicts = [r.grader_output.verdict for r in agent_results]
    if "insufficient" in verdicts:
        return "insufficient"
    if "partial" in verdicts:
        return "partial"
    return "accepted"


def _build_evidence_summary(agent_results: list[AgentResult]) -> str:
    """
    Builds a plain-text evidence summary across all agents.
    Passed to the synthesizer prompt for context.
    """
    lines = []
    for r in agent_results:
        go = r.grader_output
        n_rcts = sum(
            1 for g in go.accepted_chunks
            if g.chunk.study_type == "RCT"
        )
        lines.append(
            f"{r.agent_type.upper()}: verdict={go.verdict} | "
            f"chunks={len(go.accepted_chunks)} | "
            f"RCTs={n_rcts} | "
            f"attempts={go.attempts}"
        )
    return "\n".join(lines)


def _build_references_section(agent_results: list[AgentResult]) -> str:
    """
    Collects citations from all agent sections, deduplicates by PMID,
    and returns a numbered markdown References block with PubMed links.
    Falls back to the full citation string as the dedup key when no PMID found.
    """
    seen_keys: set[str] = set()
    unique_citations: list[tuple[str, str]] = []  # (pmid_key, citation_str)

    for r in agent_results:
        for citation in r.section.citations:
            match = re.search(r'\[PMID:\s*(\S+?)\]', citation)
            key = match.group(1).rstrip(']') if match else citation
            if key not in seen_keys:
                seen_keys.add(key)
                unique_citations.append((key, citation))

    if not unique_citations:
        return ""

    lines = ["\n\n---\n", "## References\n"]
    for i, (key, citation) in enumerate(unique_citations, 1):
        if key.isdigit():
            lines.append(
                f"{i}. {citation} "
                # f"[[PubMed]](https://pubmed.ncbi.nlm.nih.gov/{key}/)"
            )
        else:
            lines.append(f"{i}. {citation}")
    return "\n".join(lines)


async def synthesizer_node(state: GraphState) -> dict:
    """
    Synthesizer node — final node in the graph.
    Merges all AgentResult sections into one cohesive report.
    Computes overall verdict from all agent verdicts.
    Produces AgentOutput — the final deliverable.

    Uses gpt-4o (stronger model) because synthesis requires
    coherent long-form writing across multiple evidence domains.
    """
    agent_results   = state.get("agent_results", [])
    query           = state["query"]

    if not agent_results:
        return {"final_output": AgentOutput(
            report           = "No agent results available.",
            sections         = [],
            overall_verdict  = "insufficient",
            evidence_summary = "No agents ran successfully.",
            query            = query,
        )}

    print(f"\nSynthesizer: merging {len(agent_results)} agent sections")

    # Build sections text for prompt
    sections_text = "\n\n---\n\n".join([
        f"## {r.section.title}\n\n"
        f"{r.section.content}\n\n"
        f"**Citations:** {', '.join(r.section.citations)}\n"
        f"**Evidence quality:** {r.section.evidence_quality}\n"
        f"**Limitations:** {r.section.limitations}\n"
        + (f"**Note:** {r.section.verdict_note}\n" if r.section.verdict_note else "")
        for r in agent_results
    ])

    overall_verdict  = _compute_overall_verdict(agent_results)
    evidence_summary = _build_evidence_summary(agent_results)

    try:
        response = await _SYNTHESIZER_CHAIN.ainvoke({
            "query":            query,
            "sections_text":    sections_text,
            "evidence_summary": evidence_summary,
            "overall_verdict":  overall_verdict,
        })
        report = response.content.strip()

    except Exception as e:
        print(f"Synthesizer failed: {e}. Using concatenated sections.")
        report = f"# Research Brief\n\n**Query:** {query}\n\n" + sections_text

    # Strip any inline [PMID: XXXXX] citations the LLM may have included despite instructions
    report = re.sub(r'\s*\[PMID:\s*\S+?\]', '', report).strip()

    references = _build_references_section(agent_results)
    if references:
        report += references

    final_output = AgentOutput(
        report           = report,
        sections         = [r.section for r in agent_results],
        overall_verdict  = overall_verdict,
        evidence_summary = evidence_summary,
        query            = query,
        agent_results    = agent_results,
    )

    print(f"Synthesizer: report generated | "
          f"verdict={overall_verdict} | "
          f"{len(report)} chars")

    return {"final_output": final_output}


# ─────────────────────────────────────────────────────────────────────
# 12. LANGGRAPH GRAPH CONSTRUCTION
# ─────────────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """
    Assembles the LangGraph StateGraph.

    Graph structure:
      START → supervisor → [router] → specialist_agent(s) → synthesizer → END

    Key LangGraph concepts:
      StateGraph:  typed state flows through all nodes
      Send() API:  supervisor dynamically dispatches to agents
      Annotated[list, operator.add]: concurrent agents safely append results

    Why StateGraph not MessageGraph:
      We pass structured dataclasses (AgentResult, SupervisorPlan)
      not just messages. StateGraph supports arbitrary typed state.
    """
    graph = StateGraph(GraphState)

    # Add nodes
    graph.add_node("supervisor",         supervisor_node)
    graph.add_node("specialist_agent",   specialist_agent_node)
    graph.add_node("synthesizer",        synthesizer_node)

    # Static edges
    graph.add_edge(START,         "supervisor")
    graph.add_edge("specialist_agent", "synthesizer")

    # Dynamic dispatch via Send()
    graph.add_conditional_edges(
        "supervisor",
        route_to_agents,
        ["specialist_agent"],
    )

    graph.add_edge("synthesizer", END)

    return graph.compile()


# Compiled graph — module-level singleton
_graph = build_graph()


# ─────────────────────────────────────────────────────────────────────
# 13. PUBLIC API
# ─────────────────────────────────────────────────────────────────────

# LANGFUSE CHANGE 2 of 2:
# Original signature:  async def run_research(query: str) -> AgentOutput:
# New signature adds:  callback_handler=None
# Original body:       result = await _graph.ainvoke(initial_state)
# New body adds:       invoke_config block before ainvoke

async def run_research(
    query:            str,
    callback_handler = None,   # kept for backward compat, not used directly
    invoke_config:    dict = None,   # Langfuse config from get_callback_handler()
) -> AgentOutput:
    """
    Main entry point for agents.py.
    Called by mcp_server.py and api/main.py.

    Runs the full LangGraph pipeline:
      supervisor → specialist agents (parallel) → synthesizer

    Arguments:
      query:         raw user query — any format, any abbreviation
      invoke_config: config dict from langfuse_setup.get_callback_handler()
                     e.g. {"callbacks": [handler], "run_name": "...", "metadata": {...}}
                     Pass None (default) to run without tracing — identical behaviour.

    Returns:
      AgentOutput with complete research brief, citations, verdict
    """
    print(f"\n{'='*60}")
    print(f"Research query: {query}")
    print(f"{'='*60}")

    initial_state: GraphState = {
        "query":           query,
        "supervisor_plan": None,
        "agent_results":   [],
        "final_output":    None,
    }

    # invoke_config=None or {} are both no-ops — identical to original
    result = await _graph.ainvoke(initial_state, config=invoke_config or {})
    return result["final_output"]


# ─────────────────────────────────────────────────────────────────────
# 14. QUICK TEST
# ─────────────────────────────────────────────────────────────────────

async def _test() -> None:
    """Quick smoke test — python agents.py"""
    print("Agents smoke test\n" + "=" * 55)

    test_queries = [
        # Simple efficacy — should activate efficacy agent only
        # "What is the overall survival benefit of pembrolizumab in NSCLC?",
        # Comparison — should activate efficacy + competitor
        "How does pembrolizumab compare to nivolumab in NSCLC survival?",
        # Safety focused — should activate safety agent
        # "What are the immune-related adverse events of pembrolizumab?",
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        output = await run_research(query)

        print(f"\nVerdict  : {output.overall_verdict}")
        print(f"Sections : {[s.title for s in output.sections]}")
        print(f"Evidence : {output.evidence_summary}")
        print(f"\nReport preview (first 400 chars):")
        print(output.report[:400])
        print("\n" + "-" * 60)


if __name__ == "__main__":
    open("graph.png", "wb").write(_graph.get_graph().draw_mermaid_png())
    asyncio.run(_test())