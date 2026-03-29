"""
app.py — Streamlit Dashboard for Rumor Detection Lab
Imports pipeline from rumor.py. Run with: streamlit run app.py
"""

import streamlit as st
import json
import time
import datetime
import numpy as np
import pandas as pd
from langchain_core.documents import Document

# ── import everything from core pipeline ─────────────────────────────────────
from rumor import (
    graph,
    _load_cluster_df,
    CLUSTER_CSV_PATH,
    SIMILARITY_THRESHOLD,
    RAG_WEAK_THRESHOLD,
    HEALTH_DOMAINS,
    RumorState,
)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Rumor Detection Lab",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@700;800&family=Inter:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0b0f14;
    color: #d4dbe8;
}
h1, h2, h3 { font-family: 'Syne', sans-serif; }

.rd-header {
    background: linear-gradient(135deg, #0d1b2a 0%, #112240 100%);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 28px 36px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.rd-header::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; bottom: 0;
    background: repeating-linear-gradient(
        45deg, transparent, transparent 20px,
        rgba(30,90,150,0.03) 20px, rgba(30,90,150,0.03) 21px
    );
}
.rd-title    { font-size: 2.2rem; font-weight: 800; color: #e8f4fd; margin: 0; letter-spacing: -0.5px; }
.rd-subtitle { color: #6a8fb5; font-size: 0.88rem; margin-top: 6px; font-family: 'DM Mono', monospace; }

/* ── CONSOLE ── */
.console-wrap {
    background: #060a0f;
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 20px;
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    line-height: 1.8;
    min-height: 60px;
}
.console-line        { color: #4a7aa5; }
.console-line.ok     { color: #34d399; }
.console-line.warn   { color: #fbbf24; }
.console-line.err    { color: #f87171; }
.console-line.dim    { color: #2a4a6a; }
.console-cursor      { display: inline-block; width: 7px; height: 13px; background: #38bdf8; vertical-align: middle; animation: blink 1s step-end infinite; }
@keyframes blink { 50% { opacity: 0; } }

/* ── Verdict badge ── */
.verdict-badge { display: inline-block; padding: 6px 18px; border-radius: 20px; font-family: 'DM Mono', monospace; font-size: 0.82rem; font-weight: 500; text-transform: uppercase; letter-spacing: 1px; }
.verdict-supported    { background: #0d2e1a; color: #4ade80; border: 1px solid #16a34a; }
.verdict-contradicted { background: #2e0d0d; color: #f87171; border: 1px solid #dc2626; }
.verdict-conflicting  { background: #2e1d0d; color: #fbbf24; border: 1px solid #d97706; }
.verdict-insufficient { background: #1a1a2e; color: #94a3b8; border: 1px solid #475569; }

/* ── Path badge ── */
.path-hit  { display: inline-block; padding: 4px 14px; border-radius: 6px; background: #0d2240; color: #38bdf8; border: 1px solid #0284c7; font-family: 'DM Mono', monospace; font-size: 0.75rem; }
.path-miss { display: inline-block; padding: 4px 14px; border-radius: 6px; background: #1a1040; color: #a78bfa; border: 1px solid #7c3aed; font-family: 'DM Mono', monospace; font-size: 0.75rem; }
.path-web  { display: inline-block; padding: 4px 14px; border-radius: 6px; background: #0d2e1a; color: #34d399; border: 1px solid #059669; font-family: 'DM Mono', monospace; font-size: 0.75rem; }

/* ── Result panel ── */
.result-panel { background: #0d1b2a; border: 1px solid #1e3a5f; border-radius: 12px; padding: 24px; margin-bottom: 16px; }
.result-panel h4 { font-family: 'Syne', sans-serif; color: #7dd3fc; margin-bottom: 12px; font-size: 1rem; }

.claim-field .key { font-family: 'DM Mono', monospace; font-size: 0.72rem; color: #4a7aa5; text-transform: uppercase; letter-spacing: 0.5px; }
.claim-field .val { font-size: 0.9rem; color: #d4dbe8; margin-top: 2px; margin-bottom: 8px; }

.doc-card { background: #091422; border: 1px solid #1a3050; border-radius: 8px; padding: 14px 16px; margin-bottom: 10px; font-size: 0.85rem; }
.doc-card .doc-source { font-family: 'DM Mono', monospace; font-size: 0.72rem; color: #4a7aa5; margin-bottom: 6px; }
.doc-card .doc-text   { color: #b8c8d8; line-height: 1.5; }
.doc-web { border-color: #1a4030 !important; }
.doc-web .doc-source { color: #34d399 !important; }

.sim-bar-wrap { background: #0d1b2a; border-radius: 4px; height: 8px; margin-top: 6px; }
.sim-bar      { height: 8px; border-radius: 4px; background: linear-gradient(90deg, #0284c7, #7dd3fc); }

.history-row { display: flex; align-items: center; gap: 12px; padding: 10px 14px; border-bottom: 1px solid #112240; font-size: 0.84rem; }
.history-row:last-child { border-bottom: none; }
.history-ts    { font-family: 'DM Mono', monospace; font-size: 0.7rem; color: #4a7aa5; min-width: 70px; }
.history-rumor { color: #b8c8d8; flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

/* ── Streamlit overrides ── */
.stTextArea textarea { background: #0d1b2a !important; border: 1px solid #1e3a5f !important; color: #d4dbe8 !important; border-radius: 8px !important; }
.stButton > button { background: linear-gradient(135deg, #0369a1, #0284c7) !important; color: white !important; border: none !important; border-radius: 8px !important; font-family: 'Syne', sans-serif !important; font-size: 1rem !important; font-weight: 700 !important; padding: 12px 32px !important; width: 100% !important; transition: all 0.2s !important; }
.stButton > button:hover { background: linear-gradient(135deg, #0284c7, #38bdf8) !important; }
.stExpander { border: 1px solid #1e3a5f !important; border-radius: 8px !important; }
[data-testid="stSidebar"] { background: #080d13 !important; border-right: 1px solid #1e3a5f !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "run_count" not in st.session_state:
    st.session_state.run_count = {"total": 0, "hit": 0, "miss": 0, "web": 0}

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def verdict_badge(v: str) -> str:
    return f'<span class="verdict-badge verdict-{v}">{v}</span>'

def claim_field(key: str, val) -> str:
    if val is None: val = "—"
    return f'<div class="claim-field"><div class="key">{key}</div><div class="val">{val}</div></div>'

VERDICT_ICONS = {
    "supported":    ("✅", "Documents confirm the claim."),
    "contradicted": ("❌", "Documents contradict the claim."),
    "conflicting":  ("⚠️", "Mixed evidence found."),
    "insufficient": ("ℹ️", "Insufficient evidence in retrieved documents."),
}

# ─────────────────────────────────────────────────────────────────────────────
# LIVE CONSOLE RENDERER
# ─────────────────────────────────────────────────────────────────────────────
def render_console(lines: list[tuple[str, str]], running: bool = False) -> str:
    """lines = list of (css_class, text). css_class ∈ {ok, warn, err, dim, ''}"""
    inner = ""
    for cls, text in lines:
        inner += f'<div class="console-line {cls}">{text}</div>\n'
    if running:
        inner += '<div class="console-line"><span class="console-cursor"></span></div>'
    return f'<div class="console-wrap">{inner}</div>'


# ─────────────────────────────────────────────────────────────────────────────
# STREAMING PIPELINE WITH LIVE CONSOLE
# ─────────────────────────────────────────────────────────────────────────────
def run_with_live_console(rumor: str, console_slot):
    """
    Runs the LangGraph pipeline step-by-step via .stream(),
    updating the console_slot placeholder after each node.
    Returns the final state dict.
    """
    log: list[tuple[str, str]] = []
    final_state: dict = {}

    def push(cls: str, text: str):
        log.append((cls, text))
        console_slot.markdown(render_console(log, running=True), unsafe_allow_html=True)

    push("dim", "┌─ pipeline starting ─────────────────────────────")
    push("", f"  rumor: {rumor[:80]}{'…' if len(rumor) > 80 else ''}")

    init_state: RumorState = {
        "rumor":              rumor,
        "claim":              None,
        "embedding":          None,
        "sim_score":          None,
        "matched_cluster_id": None,
        "rag_docs":           None,
        "rag_scores":         None,
        "web_docs":           None,
        "web_triggered":      None,
        "plain_json":         None,
        "validation":         None,
        "new_cluster_id":     None,
    }

    NODE_LABELS = {
        "extract_claim":       ("🧠", "Extracting atomic claim…"),
        "attach_embedding":    ("📐", "Computing embedding vector…"),
        "similarity_check":    ("🔍", "Checking cluster similarity…"),
        "append_to_cluster":   ("🗂 ", "Appending to existing cluster…"),
        "rag_retrieve":        ("📚", "Querying RAG vector store…"),
        "web_search_fallback": ("🌐", "RAG weak — fetching web sources…"),
        "validate":            ("⚖️ ", "Validating claim against evidence…"),
        "create_cluster":      ("🆕", "Creating new cluster…"),
    }

    t0 = time.time()

    for chunk in graph.stream(init_state, stream_mode="updates"):
        for node_name, node_output in chunk.items():
            label, desc = NODE_LABELS.get(node_name, ("▸", node_name))

            # ── per-node log lines ─────────────────────────────────────────
            if node_name == "extract_claim":
                push("ok", f"  {label} {desc}")
                claim = node_output.get("claim") or {}
                push("", f"     canonical: {claim.get('canonical_text', '?')}")
                push("", f"     type: {claim.get('claim_type','?')}  entities: {claim.get('entities','?')}")

            elif node_name == "attach_embedding":
                push("ok", f"  {label} {desc}")
                emb = node_output.get("embedding") or []
                push("dim", f"     vector dim: {len(emb)}")

            elif node_name == "similarity_check":
                push("ok", f"  {label} {desc}")
                score  = node_output.get("sim_score", 0)
                cid    = node_output.get("matched_cluster_id")
                thresh = SIMILARITY_THRESHOLD
                bar    = "█" * int(score * 20) + "░" * (20 - int(score * 20))
                if cid is not None:
                    push("ok",  f"     [{bar}] {score:.4f}  ✓ HIT → cluster {cid}")
                else:
                    push("warn",f"     [{bar}] {score:.4f}  ✗ MISS (threshold {thresh})")

            elif node_name == "append_to_cluster":
                push("ok", f"  {label} {desc}")

            elif node_name == "rag_retrieve":
                docs   = node_output.get("rag_docs") or []
                scores = node_output.get("rag_scores") or []
                push("ok", f"  {label} {desc}")
                if docs:
                    best = max(scores) if scores else 0
                    push("", f"     {len(docs)} doc(s) retrieved  best_score={best:.3f}")
                    for i, doc in enumerate(docs):
                        sc  = f"  score={scores[i]:.3f}" if i < len(scores) else ""
                        src = doc.metadata.get("source","?") if isinstance(doc, Document) else "?"
                        push("dim", f"     [{i}] [{src}]{sc}  {doc.page_content[:60]}…")
                    if scores and max(scores) < RAG_WEAK_THRESHOLD:
                        push("warn", f"     ⚠ best score {max(scores):.3f} < threshold {RAG_WEAK_THRESHOLD} → triggering web fallback")
                    else:
                        push("ok",  f"     ✓ RAG sufficient, skipping web fallback")
                else:
                    # distinguish a corrupt/empty index from a genuine no-match
                    from rumor import _CHROMA_OK
                    if not _CHROMA_OK:
                        push("warn", "     ⚠ Chroma index unavailable (corrupt or missing) → web fallback")
                    else:
                        push("warn", f"     no docs returned → triggering web fallback")

            elif node_name == "web_search_fallback":
                web = node_output.get("web_docs") or []
                push("ok", f"  {label} {desc}")
                push("", f"     {len(web)} result(s) from DuckDuckGo")
                for i, r in enumerate(web[:4]):
                    push("dim", f"     [{i}] [{r.get('source','?')}]  {r.get('snippet','')[:60]}…")
                if not web:
                    push("warn","     no web results returned")

            elif node_name == "validate":
                val = node_output.get("validation")
                push("ok", f"  {label} {desc}")
                if val and val.results:
                    v = val.results[0].verdict
                    colors = {"supported":"ok","contradicted":"err","conflicting":"warn","insufficient":""}
                    icon, _ = VERDICT_ICONS.get(v, ("?",""))
                    push(colors.get(v,""), f"     verdict: {icon} {v.upper()}")

            elif node_name == "create_cluster":
                cid = node_output.get("new_cluster_id")
                push("ok", f"  {label} {desc}")
                push("ok", f"     cluster_id={cid} written to {CLUSTER_CSV_PATH}")

            # accumulate state
            final_state.update(node_output)

    elapsed = time.time() - t0
    push("dim", f"└─ done in {elapsed:.2f}s ──────────────────────────────")
    # stop blinking cursor
    console_slot.markdown(render_console(log, running=False), unsafe_allow_html=True)

    # merge init state so downstream code can read rumor/claim etc.
    merged = dict(init_state)
    merged.update(final_state)
    return merged, elapsed

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔬 Rumor Detection Lab")
    st.markdown("---")

    rc = st.session_state.run_count
    st.markdown(f"""
    <div style="font-family:'DM Mono',monospace;font-size:0.78rem;color:#4a7aa5;line-height:2">
    TOTAL RUNS &nbsp;&nbsp;&nbsp; <span style="color:#7dd3fc">{rc['total']}</span><br>
    CLUSTER HITS &nbsp;&nbsp; <span style="color:#38bdf8">{rc['hit']}</span><br>
    NEW CLUSTERS &nbsp;&nbsp; <span style="color:#a78bfa">{rc['miss']}</span><br>
    WEB FALLBACKS &nbsp; <span style="color:#34d399">{rc['web']}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Cluster Store")
    df_sidebar = _load_cluster_df()
    if df_sidebar.empty:
        st.caption("No clusters yet.")
    else:
        for _, row in df_sidebar.tail(5).iloc[::-1].iterrows():
            parts = json.loads(row["participants"]) if isinstance(row["participants"], str) else []
            # first participant holds the founding claim
            first = parts[0] if parts else {}
            parent_text = first.get("canonical_text") or first.get("claim") or "—"
            claim_type  = first.get("claim_type", "").upper()
            entities    = first.get("entities") or []
            entity_str  = " → ".join(entities) if entities else ""
            st.markdown(f"""
            <div style="padding:10px 0;border-bottom:1px solid #112240">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px">
                    <span style="font-family:'DM Mono',monospace;font-size:0.68rem;color:#0284c7;background:#0d2240;padding:1px 7px;border-radius:4px">C{row['cluster_id']}</span>
                    <span style="font-family:'DM Mono',monospace;font-size:0.68rem;color:#4a7aa5">{len(parts)} rumor{'s' if len(parts)!=1 else ''}</span>
                </div>
                <div style="font-size:0.78rem;color:#b8c8d8;line-height:1.4;margin-bottom:3px">{parent_text[:72]}{'…' if len(parent_text)>72 else ''}</div>
                <div style="display:flex;gap:6px;flex-wrap:wrap">
                    {'<span style="font-family:DM Mono,monospace;font-size:0.65rem;color:#4a7aa5;background:#091422;padding:1px 6px;border-radius:3px">' + claim_type + '</span>' if claim_type else ''}
                    {'<span style="font-family:DM Mono,monospace;font-size:0.65rem;color:#2a5a7a">' + entity_str + '</span>' if entity_str else ''}
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    with st.expander("🌐 Trusted Health Domains"):
        for d in HEALTH_DOMAINS:
            st.markdown(f"<span style='font-family:DM Mono,monospace;font-size:0.7rem;color:#4a7aa5'>{d}</span>", unsafe_allow_html=True)

    st.markdown("---")
    st.caption("Powered by Mistral AI + LangGraph + Chroma")

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="rd-header">
    <div class="rd-title">🔬 Rumor Detection Lab</div>
    <div class="rd-subtitle">CLAIM EXTRACTION · VECTOR CLUSTERING · RAG VALIDATION · WEB FALLBACK</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# INPUT + ABOUT PANEL
# ─────────────────────────────────────────────────────────────────────────────
col_input, col_about = st.columns([3, 2])

with col_about:
    st.markdown(f"""
    <div style="background:#060e1a;border:1px solid #1e3a5f;border-radius:10px;padding:18px 20px;height:100%">
        <div style="font-family:'Syne',sans-serif;font-size:0.95rem;color:#7dd3fc;margin-bottom:12px;letter-spacing:0.3px">
            How it works
        </div>
        <div style="display:flex;flex-direction:column;gap:8px">
            <div style="display:flex;gap:10px;align-items:flex-start">
                <span style="font-family:'DM Mono',monospace;font-size:0.7rem;color:#0284c7;background:#0d2240;padding:2px 7px;border-radius:4px;white-space:nowrap;margin-top:1px">01</span>
                <span style="font-size:0.82rem;color:#94a3b8;line-height:1.5"><b style="color:#b8c8d8">Extract</b> — the LLM pulls one atomic claim from free text, assigns type, entities, and a canonical sentence.</span>
            </div>
            <div style="display:flex;gap:10px;align-items:flex-start">
                <span style="font-family:'DM Mono',monospace;font-size:0.7rem;color:#0284c7;background:#0d2240;padding:2px 7px;border-radius:4px;white-space:nowrap;margin-top:1px">02</span>
                <span style="font-size:0.82rem;color:#94a3b8;line-height:1.5"><b style="color:#b8c8d8">Cluster</b> — cosine similarity against stored embeddings. Score ≥ {SIMILARITY_THRESHOLD} is a HIT; the rumor joins that cluster.</span>
            </div>
            <div style="display:flex;gap:10px;align-items:flex-start">
                <span style="font-family:'DM Mono',monospace;font-size:0.7rem;color:#0284c7;background:#0d2240;padding:2px 7px;border-radius:4px;white-space:nowrap;margin-top:1px">03</span>
                <span style="font-size:0.82rem;color:#94a3b8;line-height:1.5"><b style="color:#b8c8d8">RAG</b> — retrieves supporting docs from Chroma. If best score &lt; {RAG_WEAK_THRESHOLD}, falls back to DuckDuckGo health sources.</span>
            </div>
            <div style="display:flex;gap:10px;align-items:flex-start">
                <span style="font-family:'DM Mono',monospace;font-size:0.7rem;color:#0284c7;background:#0d2240;padding:2px 7px;border-radius:4px;white-space:nowrap;margin-top:1px">04</span>
                <span style="font-size:0.82rem;color:#94a3b8;line-height:1.5"><b style="color:#b8c8d8">Verdict</b> — LLM classifies as <span style="color:#4ade80">supported</span>, <span style="color:#f87171">contradicted</span>, <span style="color:#fbbf24">conflicting</span>, or <span style="color:#94a3b8">insufficient</span>.</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_input:
    rumor_text = st.text_area(
        "Enter a health rumor or claim to verify",
        placeholder="e.g. Drinking bleach cures COVID-19…",
        height=148,
        label_visibility="collapsed",
        key="rumor_input",
    )

run_btn = st.button("▶  Analyze Claim", use_container_width=False)

# ─────────────────────────────────────────────────────────────────────────────
# RUN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
if run_btn and rumor_text.strip():
    # Live console placeholder
    st.markdown("#### 🖥 Pipeline Console")
    console_slot = st.empty()
    console_slot.markdown(
        render_console([("dim", "  initializing…")], running=True),
        unsafe_allow_html=True
    )

    result, elapsed = run_with_live_console(rumor_text, console_slot)

    # Update session stats
    is_hit = result.get("matched_cluster_id") is not None
    st.session_state.run_count["total"] += 1
    if is_hit:
        st.session_state.run_count["hit"] += 1
    else:
        st.session_state.run_count["miss"] += 1
    if result.get("web_triggered"):
        st.session_state.run_count["web"] += 1

    ts = datetime.datetime.now().strftime("%H:%M:%S")
    st.session_state.history.insert(0, (ts, rumor_text, result))

    # ── PATH BADGE ────────────────────────────────────────────────────────────
    path_label = "HIT" if is_hit else ("MISS + WEB" if result.get("web_triggered") else "MISS")
    path_class = "path-hit" if is_hit else ("path-web" if result.get("web_triggered") else "path-miss")

    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:12px;margin:16px 0 8px">
        <span class="{path_class}">{path_label}</span>
        <span style="font-family:'DM Mono',monospace;font-size:0.72rem;color:#4a7aa5">⏱ {elapsed:.2f}s</span>
        <span style="font-family:'DM Mono',monospace;font-size:0.72rem;color:#4a7aa5">sim={result.get('sim_score',0):.4f}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── TWO-COLUMN RESULTS ────────────────────────────────────────────────────
    claim = result.get("claim") or result.get("plain_json") or {}
    col_left, col_right = st.columns([1, 1])

    with col_left:
        # Claim card
        st.markdown(f"""
        <div class="result-panel">
            <h4>📋 Extracted Claim</h4>
            {claim_field("Canonical Text", claim.get("canonical_text"))}
            {claim_field("Claim", claim.get("claim"))}
            {claim_field("Type", (claim.get("claim_type") or "").upper())}
            {claim_field("Entities", " → ".join(claim.get("entities") or []))}
            {claim_field("Time", claim.get("time"))}
            {claim_field("Location", claim.get("location"))}
        </div>
        """, unsafe_allow_html=True)

        # Similarity bar
        sim_score = result.get("sim_score") or 0
        sim_pct   = min(100, int(sim_score * 100))
        cluster_note = (
            f"<div style='margin-top:10px;font-size:0.82rem;color:#38bdf8'>✓ Matched cluster <b>{result.get('matched_cluster_id')}</b></div>"
            if is_hit else
            f"<div style='margin-top:10px;font-size:0.82rem;color:#a78bfa'>New cluster <b>{result.get('new_cluster_id','—')}</b> created</div>"
        )
        st.markdown(f"""
        <div class="result-panel">
            <h4>📊 Similarity Score</h4>
            <div style="font-family:'DM Mono',monospace;font-size:1.4rem;color:#7dd3fc">{sim_score:.4f}</div>
            <div class="sim-bar-wrap"><div class="sim-bar" style="width:{sim_pct}%"></div></div>
            <div style="display:flex;justify-content:space-between;font-family:'DM Mono',monospace;font-size:0.68rem;color:#4a7aa5;margin-top:4px">
                <span>0.0</span><span>threshold={SIMILARITY_THRESHOLD}</span><span>1.0</span>
            </div>
            {cluster_note}
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        # ── VERDICT — shown for BOTH hit and miss ─────────────────────────────
        validation = result.get("validation")
        if validation and validation.results:
            verdict = validation.results[0].verdict
            icon, desc = VERDICT_ICONS.get(verdict, ("?", ""))
            web_badge = "<div style='margin-top:8px'><span class='path-web'>🌐 Web Fallback Used</span></div>" if result.get("web_triggered") else ""
            hit_note  = f"<div style='margin-top:6px;font-size:0.78rem;color:#38bdf8;font-family:DM Mono,monospace'>cluster {result.get('matched_cluster_id')} re-validated</div>" if is_hit else ""
            st.markdown(f"""
            <div class="result-panel">
                <h4>⚖️ Verdict</h4>
                {verdict_badge(verdict)}
                <div style="margin-top:14px;font-size:0.82rem;color:#94a3b8;line-height:1.5">{icon} {desc}</div>
                {hit_note}
                {web_badge}
            </div>
            """, unsafe_allow_html=True)

        # HIT — cluster participants preview
        if is_hit:
            df_c = _load_cluster_df()
            cid  = result.get("matched_cluster_id")
            row  = df_c[df_c["cluster_id"] == cid]
            if not row.empty:
                parts = json.loads(row.iloc[0]["participants"]) if isinstance(row.iloc[0]["participants"], str) else []
                st.markdown(f"""<div class="result-panel"><h4>🗂 Cluster {cid} — {len(parts)} Participant(s)</h4>""", unsafe_allow_html=True)
                for i, p in enumerate(parts[-3:]):
                    st.markdown(f"""
                    <div class="doc-card">
                        <div class="doc-source">PARTICIPANT {i+1}</div>
                        <div class="doc-text">{p.get('canonical_text','')}</div>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

        # RAG docs
        rag_docs   = result.get("rag_docs") or []
        rag_scores = result.get("rag_scores") or []
        if rag_docs:
            st.markdown('<div class="result-panel"><h4>📚 RAG Documents</h4>', unsafe_allow_html=True)
            for i, doc in enumerate(rag_docs):
                content   = doc.page_content if isinstance(doc, Document) else str(doc)
                meta      = doc.metadata     if isinstance(doc, Document) else {}
                score_str = f" · score={rag_scores[i]:.3f}" if i < len(rag_scores) else ""
                st.markdown(f"""
                <div class="doc-card">
                    <div class="doc-source">[{i}] {meta.get('source','')} {meta.get('year','')}{score_str}</div>
                    <div class="doc-text">{content[:300]}{"…" if len(content)>300 else ""}</div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # Web docs
        web_docs = result.get("web_docs") or []
        if web_docs:
            st.markdown('<div class="result-panel"><h4>🌐 Web Sources (DuckDuckGo Fallback)</h4>', unsafe_allow_html=True)
            for i, doc in enumerate(web_docs):
                st.markdown(f"""
                <div class="doc-card doc-web">
                    <div class="doc-source">[{i}] {doc.get('source','')} — <a href="{doc.get('url','')}" style="color:#34d399">{doc.get('url','')[:60]}…</a></div>
                    <div class="doc-text">{doc.get('snippet','')[:300]}</div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

elif run_btn:
    st.warning("Please enter a rumor or claim to analyze.")

# ─────────────────────────────────────────────────────────────────────────────
# HISTORY
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.history:
    st.markdown("---")
    st.markdown("### 📜 Session History")
    history_html = '<div style="background:#0d1b2a;border:1px solid #1e3a5f;border-radius:10px;padding:8px 0;margin-top:8px">'
    for ts, rumor, res in st.session_state.history[:10]:
        is_h = res.get("matched_cluster_id") is not None
        val  = res.get("validation")
        verdict_str = ""
        if val and val.results:
            v = val.results[0].verdict
            colors = {"supported":"#4ade80","contradicted":"#f87171","conflicting":"#fbbf24","insufficient":"#94a3b8"}
            verdict_str = f'<span style="font-family:DM Mono,monospace;font-size:0.7rem;color:{colors.get(v,"#94a3b8")}">{v}</span>'
        path_c = "#38bdf8" if is_h else ("#34d399" if res.get("web_triggered") else "#a78bfa")
        path_t = "HIT" if is_h else ("WEB" if res.get("web_triggered") else "MISS")
        history_html += f"""
        <div class="history-row">
            <span class="history-ts">{ts}</span>
            <span style="font-family:'DM Mono',monospace;font-size:0.7rem;color:{path_c};min-width:38px">{path_t}</span>
            <span class="history-rumor">{rumor[:80]}</span>
            {verdict_str}
        </div>
        """
    history_html += "</div>"
    st.markdown(history_html, unsafe_allow_html=True)