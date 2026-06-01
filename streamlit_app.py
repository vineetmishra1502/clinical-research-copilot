"""
streamlit_app.py — Clinical Research Copilot
=============================================
LangGraph-powered multi-agent research platform.
AI-powered oncology evidence synthesis across 12,000+ PubMed abstracts.

Run:
  Terminal 1: uvicorn api.main:app --host 0.0.0.0 --port 8000
  Terminal 2: streamlit run streamlit_app.py
"""

import streamlit as st
import requests
import time

API_URL = "http://localhost:8000"

# ─────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Clinical Research Copilot",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', -apple-system, sans-serif; }
    .block-container { padding-top: 1rem; max-width: 1100px; }
    #MainMenu, footer { visibility: hidden; }

    /* ── Hero ── */
    .hero {
        padding: 32px 0 24px;
        margin-bottom: 8px;
    }
    .hero-title {
        font-size: 30px;
        font-weight: 800;
        color: #0f172a;
        margin: 0;
        letter-spacing: -0.6px;
    }
    .hero-tag {
        font-size: 15px;
        font-weight: 500;
        color: #2563eb;
        margin: 2px 0 8px;
    }
    .hero-desc {
        font-size: 14px;
        color: #64748b;
        line-height: 1.6;
        margin-bottom: 16px;
        max-width: 720px;
    }

    /* ── Tech stack chips ── */
    .tech-stack {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 8px;
        margin-top: 12px;
    }
    .tech-stack-label {
        font-size: 12px;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-right: 4px;
    }
    .tech-chip {
        display: inline-block;
        padding: 5px 14px;
        border-radius: 8px;
        font-size: 13px;
        font-weight: 600;
        letter-spacing: 0.2px;
    }
    .tc-blue   { background: #dbeafe; color: #1e40af; border: 1px solid #bfdbfe; }
    .tc-green  { background: #d1fae5; color: #065f46; border: 1px solid #a7f3d0; }
    .tc-dark   { background: #1e293b; color: #e2e8f0; border: 1px solid #334155; }
    .tc-amber  { background: #fef3c7; color: #92400e; border: 1px solid #fde68a; }
    .tc-purple { background: #ede9fe; color: #5b21b6; border: 1px solid #ddd6fe; }

    /* ── Pipeline stepper ── */
    .stepper {
        display: flex;
        gap: 0;
        margin: 20px 0;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 16px 10px;
        overflow-x: auto;
    }
    .step {
        flex: 1;
        text-align: center;
        padding: 8px 4px;
        position: relative;
        min-width: 100px;
    }
    .step:not(:last-child)::after {
        content: '';
        position: absolute;
        right: -2px;
        top: 50%;
        transform: translateY(-50%);
        width: 0;
        height: 0;
        border-top: 8px solid transparent;
        border-bottom: 8px solid transparent;
        border-left: 8px solid #cbd5e1;
    }
    .step-icon {
        width: 36px;
        height: 36px;
        border-radius: 10px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 16px;
        margin-bottom: 4px;
    }
    .step-icon.pending { background: #f1f5f9; }
    .step-icon.running { background: #dbeafe; animation: pulse 1.5s infinite; }
    .step-icon.done    { background: #d1fae5; }
    .step-name {
        font-size: 11px;
        font-weight: 600;
        color: #334155;
        line-height: 1.3;
    }
    .step-status {
        font-size: 10px;
        font-weight: 500;
        margin-top: 2px;
    }
    .step-status.s-pending { color: #94a3b8; }
    .step-status.s-running { color: #2563eb; }
    .step-status.s-done    { color: #059669; }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.6; }
    }

    /* ── Metric cards ── */
    .metrics {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
        margin: 16px 0;
    }
    .mcard {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px 20px;
        text-align: center;
    }
    .mcard-label {
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.7px;
        color: #94a3b8;
        margin-bottom: 4px;
    }
    .mcard-val {
        font-size: 24px;
        font-weight: 700;
        line-height: 1.1;
    }
    .v-green  { color: #059669; }
    .v-amber  { color: #d97706; }
    .v-red    { color: #dc2626; }
    .v-blue   { color: #2563eb; }
    .v-dark   { color: #1e293b; }

    /* ── Evidence summary box ── */
    .evidence-box {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 12px;
        padding: 14px 20px;
        margin: 12px 0;
        display: flex;
        align-items: center;
        gap: 14px;
    }
    .evidence-box.partial {
        background: #fffbeb;
        border-color: #fde68a;
    }
    .evidence-box.insufficient {
        background: #fef2f2;
        border-color: #fecaca;
    }
    .evidence-label {
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        padding: 4px 12px;
        border-radius: 6px;
        white-space: nowrap;
    }
    .el-green  { background: #d1fae5; color: #065f46; }
    .el-amber  { background: #fef3c7; color: #92400e; }
    .el-red    { background: #fee2e2; color: #991b1b; }
    .evidence-text {
        font-size: 14px;
        color: #334155;
        line-height: 1.5;
    }

    /* ── Chunk cards ── */
    .chunk {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 20px 24px;
        margin-bottom: 16px;
        transition: all 0.15s;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    }
    .chunk:hover {
        border-color: #93c5fd;
        box-shadow: 0 4px 16px rgba(0,0,0,0.06);
    }
    .chunk-top {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin-bottom: 14px;
    }
    .chunk-top-left {
        display: flex;
        align-items: center;
        gap: 12px;
        flex: 1;
    }
    .chunk-num {
        width: 30px;
        height: 30px;
        border-radius: 8px;
        background: #1e3a5f;
        color: #fff;
        font-size: 14px;
        font-weight: 700;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
    }
    .chunk-name {
        font-size: 15px;
        font-weight: 600;
        color: #0f172a;
    }
    .chunk-badges {
        display: flex;
        gap: 6px;
        flex-wrap: wrap;
        margin-top: 4px;
    }
    .chunk-score-box {
        text-align: right;
        flex-shrink: 0;
    }
    .chunk-score {
        font-size: 22px;
        font-weight: 700;
        color: #2563eb;
    }
    .chunk-score-label {
        font-size: 10px;
        color: #94a3b8;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.5px;
    }

    /* ── Chunk metadata grid ── */
    .meta-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
        gap: 1px;
        background: #f1f5f9;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        overflow: hidden;
        margin-top: 12px;
    }
    .meta-cell {
        background: #fff;
        padding: 10px 14px;
    }
    .meta-key {
        font-size: 10px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        color: #94a3b8;
        margin-bottom: 2px;
    }
    .meta-val {
        font-size: 13px;
        font-weight: 500;
        color: #1e293b;
    }
    .meta-val a {
        color: #2563eb;
        text-decoration: none;
    }
    .meta-val a:hover { text-decoration: underline; }

    /* ── Badges ── */
    .badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.2px;
    }
    .b-green  { background: #d1fae5; color: #065f46; }
    .b-blue   { background: #dbeafe; color: #1e40af; }
    .b-amber  { background: #fef3c7; color: #92400e; }
    .b-purple { background: #ede9fe; color: #5b21b6; }
    .b-gray   { background: #f1f5f9; color: #475569; }
    .b-dark   { background: #1e293b; color: #e2e8f0; }
    .b-teal   { background: #ccfbf1; color: #115e59; }

    /* ── Report ── */
    .report-box {
        background: #fff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 32px 36px;
        margin: 16px 0;
        line-height: 1.85;
        font-size: 15px;
        color: #334155;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }

    /* ── Section card ── */
    .sec-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #2563eb;
        border-radius: 0 12px 12px 0;
        padding: 20px 24px;
        margin-bottom: 14px;
    }

    /* ── Sidebar ── */
    .sb-brand {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 4px;
    }
    .sb-logo {
        font-size: 28px;
    }
    .sb-title {
        font-size: 18px;
        font-weight: 700;
        color: #0f172a;
        line-height: 1.2;
    }
    .sb-tagline {
        font-size: 12px;
        color: #64748b;
        margin-bottom: 12px;
        line-height: 1.4;
    }
    .sb-section {
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #94a3b8;
        margin: 18px 0 8px;
    }
    .sb-stat {
        display: flex;
        justify-content: space-between;
        padding: 6px 0;
        font-size: 13px;
        border-bottom: 1px solid #f1f5f9;
    }
    .sb-stat-k { color: #64748b; }
    .sb-stat-v { font-weight: 600; color: #1e293b; }

    .tech-row {
        display: flex;
        flex-wrap: wrap;
        gap: 4px;
        margin-top: 4px;
    }
    .tech-badge {
        font-size: 11px;
        font-weight: 500;
        padding: 3px 8px;
        border-radius: 6px;
        background: #f0f7ff;
        color: #1e40af;
        border: 1px solid #dbeafe;
    }

    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 14px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
    }
    .sp-ok   { background: #d1fae5; color: #065f46; }
    .sp-err  { background: #fee2e2; color: #991b1b; }
    .sp-dot  { width: 7px; height: 7px; border-radius: 50%; }
    .dot-g   { background: #10b981; }
    .dot-r   { background: #ef4444; }

    .divider { height: 1px; background: #e2e8f0; margin: 14px 0; border: none; }

    .example-label {
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #94a3b8;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────


@st.cache_data(ttl=60)
def api_health():
    try:
        r = requests.get(f"{API_URL}/health", timeout=10)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


@st.cache_data(ttl=300)
def api_drugs():
    try:
        r = requests.get(f"{API_URL}/drugs", timeout=10)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def vcss(v):
    return {"accepted": "v-green", "partial": "v-amber", "insufficient": "v-red"}.get(v, "v-dark")

def vbadge(v):
    c = {"accepted": "el-green", "partial": "el-amber", "insufficient": "el-red"}.get(v, "el-green")
    return f'<span class="evidence-label {c}">{v.upper()}</span>'

def ebox_cls(v):
    return {"partial": "partial", "insufficient": "insufficient"}.get(v, "")

def ev_badge(l):
    m = {1: ("Level 1", "RCT", "b-green"), 2: ("Level 2", "Cohort", "b-blue"), 3: ("Level 3", "Case", "b-amber")}
    lbl, desc, css = m.get(l, (f"L{l}", "", "b-gray"))
    return f'<span class="badge {css}">{lbl}</span>'


def linkify_pmids(text: str) -> str:
    """Convert [PMID: XXXXX] references in report text to clickable PubMed links."""
    import re
    def replacer(match):
        pmid = match.group(1).strip()
        return f"[PMID: {pmid}](https://pubmed.ncbi.nlm.nih.gov/{pmid}/)"
    return re.sub(r'\[PMID:\s*(\d+)\]', replacer, text)


def pipeline_stepper(stage="idle"):
    # Matches actual agents.run_research() → _graph.ainvoke() flow:
    #
    # supervisor_node()
    #   → SupervisorPlan → route_to_agents() → list[Send()]
    #
    # specialist_agent_node() × N in parallel (via Send):
    #   → run_specialist_agent(task)
    #     → retrieve_and_grade(focused_query, agent_filters)
    #       → rewrite_query() + retrieve() + grade()
    #     → chain.ainvoke() → AgentSection
    #
    # synthesizer_node()
    #   → merges all sections → AgentOutput
    stages = [
        ("1", "Supervisor", "Analyzes query and selects specialist agents"),
        ("2", "Agent Dispatch", "Specialist agents activated in parallel via Send()"),
        ("3", "Evidence Retrieval", "Per agent: rewriting + dense/BM25 + RRF + Cohere rerank"),
        ("4", "Evidence Grading", "Per agent: LLM relevancy + formula faithfulness"),
        ("5", "Section Writing", "Per agent: domain expert generates cited findings"),
        ("6", "Report Synthesis", "Synthesizer merges all sections into final brief"),
    ]
    order = ["supervisor", "dispatch", "retrieval", "grading", "writing", "synthesis"]
    stage_idx = order.index(stage) if stage in order else -1

    html = '<div class="stepper">'
    for i, (icon, name, desc) in enumerate(stages):
        if stage == "done":
            s_cls = "done"
            s_text = '<span class="step-status s-done">Completed</span>'
        elif i < stage_idx:
            s_cls = "done"
            s_text = '<span class="step-status s-done">Completed</span>'
        elif i == stage_idx:
            s_cls = "running"
            s_text = '<span class="step-status s-running">Running...</span>'
        else:
            s_cls = "pending"
            s_text = '<span class="step-status s-pending">Pending</span>'

        html += (
            f'<div class="step">'
            f'<div class="step-icon {s_cls}"><span style="font-weight:700;font-size:13px;">{icon}</span></div>'
            f'<div class="step-name">{name}</div>'
            f'<div style="font-size:9px;color:#94a3b8;line-height:1.2;margin-top:1px;">{desc}</div>'
            f'{s_text}'
            f'</div>'
        )
    html += '</div>'
    return html


# ─────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        '<div class="sb-brand">'
        '<span class="sb-logo">🧬</span>'
        '<span class="sb-title">Clinical Research<br>Copilot</span>'
        '</div>'
        '<div class="sb-tagline">LangGraph-powered multi-agent research platform</div>',
        unsafe_allow_html=True,
    )

    health = api_health()
    if health and health.get("status") == "healthy":
        st.markdown(
            '<span class="status-pill sp-ok"><span class="sp-dot dot-g"></span> Connected</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="status-pill sp-err"><span class="sp-dot dot-r"></span> API offline</span>',
            unsafe_allow_html=True,
        )
        st.code("uvicorn api.main:app --host 0.0.0.0 --port 8000", language="bash")
        st.stop()

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="sb-section">Navigation</div>', unsafe_allow_html=True)

    mode = st.radio(
        "nav",
        ["Research", "Evidence Search"],
        label_visibility="collapsed",
    )

    # Load data
    drugs_data = api_drugs()

    # Pipeline summary
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="sb-section">How it works</div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:12px;color:#475569;line-height:1.6;">'
        '<strong>1.</strong> Query rewritten with alias normalization<br>'
        '<strong>2.</strong> Dense (HNSW) + BM25 (GIN) parallel retrieval<br>'
        '<strong>3.</strong> Weighted RRF fusion + Cohere cross-encoder rerank<br>'
        '<strong>4.</strong> LLM relevancy + formula faithfulness grading<br>'
        '<strong>5.</strong> Specialist agents analyze in parallel<br>'
        '<strong>6.</strong> Synthesizer produces cited research brief'
        '</div>',
        unsafe_allow_html=True,
    )

    # Stats
    if drugs_data:
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown('<div class="sb-section">Knowledge base</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="sb-stat"><span class="sb-stat-k">Drugs indexed</span>'
            f'<span class="sb-stat-v">{len(drugs_data["drugs"])}</span></div>'
            f'<div class="sb-stat"><span class="sb-stat-k">Cancer types</span>'
            f'<span class="sb-stat-v">{len(drugs_data["cancer_types"])}+</span></div>'
            f'<div class="sb-stat"><span class="sb-stat-k">Endpoints</span>'
            f'<span class="sb-stat-v">{len(drugs_data["endpoints"])}</span></div>'
            f'<div class="sb-stat"><span class="sb-stat-k">Study designs</span>'
            f'<span class="sb-stat-v">{len(drugs_data["study_types"])}</span></div>'
            f'<div class="sb-stat"><span class="sb-stat-k">Total abstracts</span>'
            f'<span class="sb-stat-v">12,000+</span></div>',
            unsafe_allow_html=True,
        )

        with st.expander("Drug coverage & aliases"):
            for drug in drugs_data["drugs"]:
                aliases = ", ".join(drug["aliases"]) if drug["aliases"] else "-"
                st.caption(f"**{drug['name'].title()}** — _{aliases}_")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="sb-section">Settings</div>', unsafe_allow_html=True)

    api_url_input = st.text_input(
        "API URL",
        value=st.session_state.get("api_url", API_URL),
        help="Backend server URL. Change if deployed remotely.",
    )
    if api_url_input != st.session_state.get("api_url", API_URL):
        st.session_state["api_url"] = api_url_input

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.caption("Data source: PubMed E-utilities API")
    st.caption("12,000+ abstracts across 5 oncology/pharma drugs")


# ─────────────────────────────────────────────────────────────────────
# Main — Hero
# ─────────────────────────────────────────────────────────────────────

st.markdown(
    '<div class="hero">'
    '<div class="hero-title">Clinical Research Copilot</div>'
    '<div class="hero-tag">LangGraph-Powered Multi-Agent Research Platform</div>'
    '<div class="hero-desc">'
    'Transform clinical questions into evidence-backed research reports using '
    'query rewriting, hybrid retrieval, evidence grading, specialist agents, and automated synthesis.'
    '</div>'
    '<div class="tech-stack">'
    '<div class="tech-stack-label">Built with</div>'
    '<span class="tech-chip tc-blue">LangGraph</span>'
    '<span class="tech-chip tc-green">FastAPI</span>'
    '<span class="tech-chip tc-dark">OpenAI GPT-4o</span>'
    '<span class="tech-chip tc-amber">Cohere Rerank v3</span>'
    '<span class="tech-chip tc-blue">pgvector HNSW</span>'
    '<span class="tech-chip tc-green">PostgreSQL BM25</span>'
    '<span class="tech-chip tc-purple">Pydantic</span>'
    '<span class="tech-chip tc-dark">asyncio + asyncpg</span>'
    '</div>'
    '</div>',
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────
# Research tab
# ─────────────────────────────────────────────────────────────────────

if mode == "Research":
    st.markdown("#### Ask a clinical research question")

    query = st.text_input(
        "rq",
        value=st.session_state.get("rq", ""),
        placeholder="Ask a clinical research question...",
        label_visibility="collapsed",
    )

    st.markdown('<div class="example-label">Try an example</div>', unsafe_allow_html=True)
    examples = [
        "What is the overall survival benefit of pembrolizumab in NSCLC?",
        "Compare pembrolizumab vs nivolumab in NSCLC",
        "Immune-related adverse events of pembrolizumab",
        "How does osimertinib work in EGFR-mutant NSCLC?",
        "Trastuzumab efficacy in HER2-positive breast cancer",
        "Metformin and cancer risk reduction",
    ]
    cols = st.columns(3)
    for i, eq in enumerate(examples):
        with cols[i % 3]:
            if st.button(eq, key=f"ex_{i}", use_container_width=True):
                st.session_state["rq"] = eq
                st.rerun()

    if st.button("Run deep research", type="primary", disabled=not query.strip()):
        # Pipeline stepper placeholder
        stepper_ph = st.empty()
        result_area = st.container()

        stepper_ph.markdown(pipeline_stepper("supervisor"), unsafe_allow_html=True)
        time.sleep(0.4)
        stepper_ph.markdown(pipeline_stepper("dispatch"), unsafe_allow_html=True)
        time.sleep(0.3)
        stepper_ph.markdown(pipeline_stepper("retrieval"), unsafe_allow_html=True)

        try:
            start = time.perf_counter()
            stepper_ph.markdown(pipeline_stepper("grading"), unsafe_allow_html=True)

            resp = requests.post(
                f"{API_URL}/research",
                json={"query": query.strip()},
                timeout=180,
            )
            elapsed = time.perf_counter() - start

            stepper_ph.markdown(pipeline_stepper("done"), unsafe_allow_html=True)

            if resp.status_code == 200:
                data = resp.json()

                with result_area:
                    # Metrics
                    vc = vcss(data["overall_verdict"])
                    st.markdown(
                        f'<div class="metrics">'
                        f'<div class="mcard"><div class="mcard-label">Verdict</div>'
                        f'<div class="mcard-val {vc}">{data["overall_verdict"].title()}</div></div>'
                        f'<div class="mcard"><div class="mcard-label">Agents activated</div>'
                        f'<div class="mcard-val v-blue">{len(data["sections"])}</div></div>'
                        f'<div class="mcard"><div class="mcard-label">Research time</div>'
                        f'<div class="mcard-val v-dark">{data["elapsed_seconds"]:.1f}s</div></div>'
                        f'<div class="mcard"><div class="mcard-label">Report size</div>'
                        f'<div class="mcard-val v-dark">{len(data["report"]):,}</div></div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                    # Evidence summary
                    eb_cls = ebox_cls(data["overall_verdict"])
                    st.markdown(
                        f'<div class="evidence-box {eb_cls}">'
                        f'{vbadge(data["overall_verdict"])}'
                        f'<div class="evidence-text"><strong>Evidence Summary</strong><br>'
                        f'{data["evidence_summary"]}</div></div>',
                        unsafe_allow_html=True,
                    )

                    # Report
                    st.markdown("---")
                    st.markdown("#### 📄 Research brief")
                    # Make PMIDs in the report clickable PubMed links
                    import re
                    report_text = data["report"]
                    report_text = re.sub(
                        r'\[PMID:\s*(\d+)\]',
                        r'[PMID: \1](https://pubmed.ncbi.nlm.nih.gov/\1/)',
                        report_text,
                    )
                    st.markdown(report_text)

                    # Sections
                    st.markdown("---")
                    st.markdown("#### 🔬 Agent sections")
                    for i, sec in enumerate(data["sections"], 1):
                        with st.expander(f"**{sec['title']}**", expanded=(i == 1)):
                            if sec.get("verdict_note"):
                                st.warning(sec["verdict_note"])
                            st.markdown(linkify_pmids(sec["content"]))
                            c1, c2 = st.columns(2)
                            with c1:
                                if sec.get("evidence_quality"):
                                    st.success(f"**Evidence quality:** {sec['evidence_quality']}")
                            with c2:
                                if sec.get("limitations"):
                                    st.info(f"**Limitations:** {sec['limitations']}")
                            if sec.get("citations"):
                                st.markdown("**Citations:**")
                                for cit in sec["citations"]:
                                    pmid = ""
                                    if "PMID:" in cit:
                                        pmid = cit.split("PMID:")[1].strip().split("]")[0].strip()
                                    link = f" — [View on PubMed](https://pubmed.ncbi.nlm.nih.gov/{pmid}/)" if pmid else ""
                                    st.markdown(f"- {cit}{link}")

                    with st.expander("📦 Raw API response"):
                        st.json(data)

            elif resp.status_code == 504:
                stepper_ph.markdown(pipeline_stepper("idle"), unsafe_allow_html=True)
                st.error("Pipeline timed out. Try a simpler query.")
            else:
                stepper_ph.markdown(pipeline_stepper("idle"), unsafe_allow_html=True)
                st.error(f"Error {resp.status_code}: {resp.json().get('detail', 'Unknown')}")

        except requests.Timeout:
            stepper_ph.markdown(pipeline_stepper("idle"), unsafe_allow_html=True)
            st.error("Request timed out after 180 seconds.")
        except Exception as e:
            stepper_ph.markdown(pipeline_stepper("idle"), unsafe_allow_html=True)
            st.error(f"{type(e).__name__}: {e}")


# ─────────────────────────────────────────────────────────────────────
# Evidence search tab
# ─────────────────────────────────────────────────────────────────────

elif mode == "Evidence Search":
    st.markdown("#### Search clinical evidence")

    query = st.text_input(
        "sq",
        placeholder="Search — e.g., pembrolizumab overall survival NSCLC",
        label_visibility="collapsed",
    )

    # Filters in main area — collapsible
    with st.expander("Filters", expanded=False):
        st.caption("Leave as 'Any' to let the AI extract filters from your query automatically.")

        d_opts = ["Any"] + ([d["name"] for d in drugs_data["drugs"]] if drugs_data else [])
        c_opts = ["Any"] + (sorted(drugs_data.get("cancer_types", [])) if drugs_data else [])
        e_opts = ["Any"] + (drugs_data.get("endpoints", []) if drugs_data else [])
        s_opts = ["Any"] + (sorted(drugs_data.get("study_types", [])) if drugs_data else [])

        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            drug_f = st.selectbox("Drug", d_opts)
            endpoint_f = st.selectbox("Endpoint", e_opts)
        with fc2:
            cancer_f = st.selectbox(
                "Cancer type",
                c_opts,
                help="Most common types shown. Rare types accessible via free-text query.",
            )
            study_f = st.selectbox("Study design", s_opts)
        with fc3:
            year_f = st.slider("Published after", 2000, 2026, 2015)

    # Active filters summary
    active = []
    if drug_f != "Any": active.append(f"drug={drug_f}")
    if cancer_f != "Any": active.append(f"cancer={cancer_f}")
    if year_f > 2000: active.append(f"year>={year_f}")
    if endpoint_f != "Any": active.append(f"endpoint={endpoint_f}")
    if study_f != "Any": active.append(f"design={study_f}")

    if active:
        st.caption(f"Active filters: {' | '.join(active)}")

    if st.button("Search evidence", type="primary", disabled=not query.strip()):
        payload = {"query": query.strip()}
        if drug_f != "Any": payload["drug"] = drug_f
        if cancer_f != "Any": payload["cancer_type"] = cancer_f
        if year_f > 2000: payload["year_min"] = year_f
        if endpoint_f != "Any": payload["endpoints"] = endpoint_f
        if study_f != "Any": payload["study_type"] = study_f

        with st.spinner("Searching knowledge base..."):
            try:
                resp = requests.post(f"{API_URL}/search", json=payload, timeout=30)

                if resp.status_code == 200:
                    data = resp.json()

                    # Metrics
                    st.markdown(
                        f'<div class="metrics">'
                        f'<div class="mcard"><div class="mcard-label">Chunks returned</div>'
                        f'<div class="mcard-val v-blue">{data["n_returned"]}</div></div>'
                        f'<div class="mcard"><div class="mcard-label">Candidates screened</div>'
                        f'<div class="mcard-val v-dark">{data["n_candidates"]}</div></div>'
                        f'<div class="mcard"><div class="mcard-label">Retrieval confidence</div>'
                        f'<div class="mcard-val v-dark">{data.get("confidence", "N/A").title()}</div></div>'
                        f'<div class="mcard"><div class="mcard-label">Search latency</div>'
                        f'<div class="mcard-val v-dark">{data["elapsed_seconds"]:.1f}s</div></div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                    rq = data.get("rewritten_query", "")
                    if rq and rq != query.strip():
                        st.info(f"**Query rewritten:** {rq}")

                    st.markdown("---")

                    # Chunks
                    if data["chunks"]:
                        for i, ch in enumerate(data["chunks"], 1):
                            sample = f"N={ch['sample_size']:,}" if ch.get("sample_size") else "N=NR"
                            pmid = ch["source"]
                            score = f"{ch['rerank_score']:.2f}"

                            st.markdown(
                                f'<div class="chunk">'
                                f'<div class="chunk-top">'
                                f'<div class="chunk-top-left">'
                                f'<div class="chunk-num">{i}</div>'
                                f'<div>'
                                f'<div class="chunk-name">{ch["drug"].title()} — {ch["cancer_type"].title()}</div>'
                                f'<div class="chunk-badges">'
                                f'{ev_badge(ch["evidence_level"])}'
                                f'<span class="badge b-gray">{ch["study_type"]}</span>'
                                f'</div></div></div>'
                                f'<div class="chunk-score-box">'
                                f'<div class="chunk-score">{score}</div>'
                                f'<div class="chunk-score-label">Relevance</div>'
                                f'</div></div>'
                                f'<div class="meta-grid">'
                                f'<div class="meta-cell"><div class="meta-key">Drug</div><div class="meta-val">{ch["drug"].title()}</div></div>'
                                f'<div class="meta-cell"><div class="meta-key">Cancer type</div><div class="meta-val">{ch["cancer_type"]}</div></div>'
                                f'<div class="meta-cell"><div class="meta-key">Study design</div><div class="meta-val">{ch["study_type"]}</div></div>'
                                f'<div class="meta-cell"><div class="meta-key">Journal</div><div class="meta-val">{ch["journal"][:30]}</div></div>'
                                f'<div class="meta-cell"><div class="meta-key">Year</div><div class="meta-val">{ch["year"]}</div></div>'
                                f'<div class="meta-cell"><div class="meta-key">Sample size</div><div class="meta-val">{sample}</div></div>'
                                f'<div class="meta-cell"><div class="meta-key">Endpoint</div><div class="meta-val">{ch.get("endpoints", "NR")}</div></div>'
                                f'<div class="meta-cell"><div class="meta-key">PMID</div>'
                                f'<div class="meta-val"><a href="https://pubmed.ncbi.nlm.nih.gov/{pmid}/" target="_blank">{pmid} ↗</a></div></div>'
                                f'</div></div>',
                                unsafe_allow_html=True,
                            )

                            with st.expander(f"View abstract — chunk {i}", expanded=False):
                                st.text(ch["content"])
                    else:
                        st.warning("No evidence found. Try broadening your query or removing filters.")

                    with st.expander("📦 Raw API response"):
                        st.json(data)
                else:
                    st.error(f"Error {resp.status_code}: {resp.json().get('detail', 'Unknown')}")

            except requests.Timeout:
                st.error("Search timed out.")
            except Exception as e:
                st.error(f"{type(e).__name__}: {e}")