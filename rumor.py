"""
rumor.py — Core pipeline: models, schemas, graph, helpers.
Import this from app.py (Streamlit) or any other entry-point.
"""

from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser
from langchain_chroma import Chroma
from pydantic import BaseModel, Field
from typing import Literal, Optional, TypedDict, List
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

import numpy as np
import pandas as pd
import requests
import ast
import json
import os

load_dotenv()

# ── tuneable constants ────────────────────────────────────────────────────────
SIMILARITY_THRESHOLD = 0.85
RAG_WEAK_THRESHOLD   = 0.40
CLUSTER_CSV_PATH     = "cluster.csv"

HEALTH_DOMAINS = [
    "who.int", "cdc.gov", "nih.gov", "pubmed.ncbi.nlm.nih.gov",
    "mayoclinic.org", "webmd.com", "healthline.com", "medlineplus.gov",
    "nejm.org", "thelancet.com", "bmj.com", "jamanetwork.com",
    "medscape.com", "cancer.org", "heart.org", "diabetes.org",
    "clevelandclinic.org", "hopkinsmedicine.org", "health.harvard.edu",
]

# ── models & stores ───────────────────────────────────────────────────────────
embed_model  = MistralAIEmbeddings()
json_model   = ChatMistralAI(model="mistral-small-latest", temperature=0.1)
jsonmodel    = ChatMistralAI(model="mistral-small-latest", temperature=0.1)

try:
    vector_store = Chroma(
        embedding_function=embed_model,
        persist_directory="rumor_detection",
        collection_name="facts"
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    _CHROMA_OK = True
except Exception as _chroma_init_err:
    print(f"[WARN] Chroma init failed ({_chroma_init_err}); RAG disabled, web fallback will be used.")
    vector_store = None
    retriever    = None
    _CHROMA_OK   = False

# ── pydantic schemas ──────────────────────────────────────────────────────────
class Claim(BaseModel):
    claim_id: int = Field(description="Unique index of this claim inside the rumor")
    claim: str = Field(
        description=(
            "Atomic factual proposition containing exactly ONE subject and ONE object. "
            "Do not copy the full rumor sentence. Split conjunctions into separate claims."
        )
    )
    claim_type: Literal[
        "health", "death", "policy", "event",
        "statistic", "relationship", "other"
    ] = Field(description="Strict category label")
    entities: list[str] = Field(
        min_length=2, max_length=2,
        description="Exactly two entities: subject and object"
    )
    time:     Optional[str] = Field(default=None, description="Explicit time reference if present")
    location: Optional[str] = Field(default=None, description="Explicit location reference if present")
    canonical_text: str = Field(
        description=(
            "Controlled identity sentence: <subject> <relation> <object> [context]. "
            "Must represent only this claim."
        )
    )


class ClaimLabel(BaseModel):
    verdict: Literal["supported", "contradicted", "conflicting", "insufficient"] = Field(
        description="Label for the claim at the same index"
    )

class ValidationOutput(BaseModel):
    results: list[ClaimLabel] = Field(
        description="List index corresponds exactly to claims list index"
    )

# ── state ─────────────────────────────────────────────────────────────────────
class RumorState(TypedDict):
    rumor:               str
    claim:               Optional[dict]
    embedding:           Optional[List[float]]
    sim_score:           Optional[float]
    matched_cluster_id:  Optional[int]
    rag_docs:            Optional[List[Document]]
    rag_scores:          Optional[List[float]]
    web_docs:            Optional[List[dict]]
    web_triggered:       Optional[bool]
    plain_json:          Optional[dict]
    validation:          Optional[ValidationOutput]
    new_cluster_id:      Optional[int]

# ── prompts ───────────────────────────────────────────────────────────────────
json_extract_prompt = ChatPromptTemplate.from_template("""
You are a structured fact extraction system.

Return ONLY valid JSON matching the schema exactly.
Do not add commentary.

--------------------------------------------------
TASK
From a rumor, extract ONE atomic factual claim suitable for verification and clustering.

Each claim must represent ONE real-world relationship:

    subject → relation → object

--------------------------------------------------
ATOMICITY RULE (CRITICAL)

Split conjunctions:

"X and Y cause Z"
→ X causes Z
→ Y causes Z

"X causes Y and Z"
→ X causes Y
→ X causes Z

Never keep conjunctions inside a claim.
Each claim must stand independently.

--------------------------------------------------
CLAIM RULES

The claim field:
- minimal factual statement
- no "and/or/with"
- no explanation
- no context phrases

--------------------------------------------------
ENTITY RULES

entities MUST contain exactly two items:
[subject, object]

Do NOT include time/location/conditions.

--------------------------------------------------
CLAIM TYPE (STRICT ENUM)

Choose one:
health | death | policy | event | statistic | relationship | other

--------------------------------------------------
TIME & LOCATION RULE

If missing → return null (NOT "NAN")

--------------------------------------------------
CANONICAL TEXT CONTRACT

Format:
<subject> <relation> <object> [context]

Allowed relations:
prevents | causes | cures | treats | increases | decreases | kills |
contains | leads_to | results_in | died_from | implemented | occurred_in | affects

Rules:
- lowercase except proper nouns
- remove modal words
- ≤ 12 words
- health claims end with "in humans"
- must match the claim meaning exactly

--------------------------------------------------
OUTPUT FORMAT (STRICT — single object, NOT a list)

{{
    "claim_id": integer (start at 0),
    "claim": string,
    "claim_type": one of enum,
    "entities": [subject, object],
    "time": string or null,
    "location": string or null,
    "canonical_text": string
}}

Return STRICT JSON only.

--------------------------------------------------
Rumor: {rumor}
""")

parser = PydanticOutputParser(pydantic_object=ValidationOutput)

validation_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are a strict factual verification classifier.

You will receive:

1) A list of CLAIMS
2) A list of DOCUMENT GROUPS

Each claim at index i must ONLY be evaluated using the
documents at index i.

Never mix indices.

Labels:

supported:
documents clearly confirm the claim

contradicted:
documents clearly deny the claim

conflicting:
documents contain both support and contradiction

insufficient:
documents related but no proof

Rules:
- No outside knowledge
- No guessing
- Output labels must match claim count exactly
- Order must be preserved

Return JSON only:

{format_instructions}
"""
    ),
    (
        "human",
        """
CLAIMS:
{claims}

DOCUMENTS:
{documents}
"""
    )
]).partial(format_instructions=parser.get_format_instructions())

# ── CSV helpers ───────────────────────────────────────────────────────────────
def _parse_emb(s: str) -> list:
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return ast.literal_eval(s)


def _load_cluster_df(csv_path: str = CLUSTER_CSV_PATH) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        return pd.DataFrame(columns=["cluster_id", "embedding_representation", "participants"])
    return pd.read_csv(csv_path)


def _load_cluster_matrix(df: pd.DataFrame):
    cluster_ids = df["cluster_id"].tolist()
    matrix = np.array(
        df["embedding_representation"].apply(_parse_emb).tolist(),
        dtype=np.float32
    )
    norms  = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.clip(norms, 1e-10, None)
    return cluster_ids, matrix


def _next_cluster_id(df: pd.DataFrame) -> int:
    if df.empty:
        return 1
    return int(df["cluster_id"].max()) + 1

# ── DuckDuckGo health search ──────────────────────────────────────────────────
def duckduckgo_health_search(query: str, max_results: int = 5) -> List[dict]:
    results = []
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; RumorDetectionBot/1.0)"}
        domain_filter = " OR ".join(f"site:{d}" for d in HEALTH_DOMAINS[:8])
        search_query  = f"{query} ({domain_filter})"
        resp = requests.get(
            "https://api.duckduckgo.com/",
            params={"q": search_query, "format": "json", "no_redirect": 1, "no_html": 1},
            headers=headers,
            timeout=10,
        )
        data = resp.json()
        for item in data.get("RelatedTopics", [])[:max_results]:
            text = item.get("Text", "")
            url  = item.get("FirstURL", "")
            if text and url:
                domain = url.split("/")[2] if url.startswith("http") else ""
                results.append({
                    "title":   item.get("Text", "")[:80],
                    "snippet": text,
                    "url":     url,
                    "source":  domain,
                })
        if data.get("AbstractText"):
            results.insert(0, {
                "title":   data.get("Heading", "Abstract"),
                "snippet": data["AbstractText"],
                "url":     data.get("AbstractURL", ""),
                "source":  data.get("AbstractSource", "DuckDuckGo"),
            })
    except Exception:
        pass

    if not results:
        try:
            resp = requests.get(
                "https://html.duckduckgo.com/html/",
                params={"q": f"{query} health medical"},
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=10,
            )
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(resp.text, "html.parser")
            for result in soup.select(".result__body")[:max_results]:
                title_el   = result.select_one(".result__title")
                snippet_el = result.select_one(".result__snippet")
                link_el    = result.select_one(".result__url")
                if snippet_el:
                    url    = link_el.get_text(strip=True) if link_el else ""
                    domain = url.split("/")[0] if url else ""
                    if any(hd in domain for hd in HEALTH_DOMAINS):
                        results.append({
                            "title":   title_el.get_text(strip=True) if title_el else "",
                            "snippet": snippet_el.get_text(strip=True),
                            "url":     url,
                            "source":  domain,
                        })
        except Exception:
            pass

    return results[:max_results]


def web_docs_to_documents(web_results: List[dict]) -> List[Document]:
    return [
        Document(
            page_content=r["snippet"],
            metadata={"source": r["source"], "url": r["url"], "title": r["title"]}
        )
        for r in web_results
    ]

# ── node functions ────────────────────────────────────────────────────────────
structured_llm = json_model.with_structured_output(Claim)


def extract_claim(state: RumorState) -> dict:
    result: Claim = (json_extract_prompt | structured_llm).invoke({"rumor": state["rumor"]})
    return {"claim": result.model_dump()}


def attach_embedding(state: RumorState) -> dict:
    embedding = embed_model.embed_documents([state["claim"]["canonical_text"]])[0]
    return {"embedding": embedding}


def similarity_check(state: RumorState) -> dict:
    claim_vec  = np.array(state["embedding"], dtype=np.float32)
    claim_norm = claim_vec / np.clip(np.linalg.norm(claim_vec), 1e-10, None)
    df = _load_cluster_df()
    if df.empty:
        return {"sim_score": 0.0, "matched_cluster_id": None}
    cluster_ids, matrix = _load_cluster_matrix(df)
    scores     = matrix @ claim_norm
    best_idx   = int(np.argmax(scores))
    best_score = float(scores[best_idx])
    return {
        "sim_score":          best_score,
        "matched_cluster_id": cluster_ids[best_idx] if best_score >= SIMILARITY_THRESHOLD else None,
    }


def append_to_cluster(state: RumorState) -> dict:
    df         = _load_cluster_df()
    cluster_id = state["matched_cluster_id"]
    row_mask   = df["cluster_id"] == cluster_id
    if not row_mask.any():
        return {}
    existing     = df.loc[row_mask, "participants"].values[0]
    participants = json.loads(existing) if isinstance(existing, str) else []
    participants.append(state["claim"])
    df.loc[row_mask, "participants"] = json.dumps(participants)
    df.to_csv(CLUSTER_CSV_PATH, index=False)
    return {}


def rag_retrieve(state: RumorState) -> dict:
    """Retrieve from Chroma. If the index is missing or corrupted, return empty
    so route_after_rag automatically triggers the web fallback."""
    if not _CHROMA_OK or retriever is None:
        print("[WARN] Chroma unavailable — skipping RAG, will use web fallback.")
        return {"rag_docs": [], "rag_scores": []}
    try:
        docs = retriever.invoke(state["claim"]["canonical_text"])
    except Exception as e:
        print(f"[WARN] Chroma query failed ({e}) — falling back to web search.")
        return {"rag_docs": [], "rag_scores": []}
    if docs:
        try:
            query_emb  = np.array(embed_model.embed_documents([state["claim"]["canonical_text"]])[0], dtype=np.float32)
            query_norm = query_emb / np.clip(np.linalg.norm(query_emb), 1e-10, None)
            doc_embs   = embed_model.embed_documents([d.page_content for d in docs])
            scores = []
            for de in doc_embs:
                dv = np.array(de, dtype=np.float32)
                dv = dv / np.clip(np.linalg.norm(dv), 1e-10, None)
                scores.append(float(query_norm @ dv))
        except Exception:
            scores = [0.0] * len(docs)
    else:
        scores = []
    return {"rag_docs": docs, "rag_scores": scores}


def web_search_fallback(state: RumorState) -> dict:
    canonical   = state["claim"]["canonical_text"]
    web_results = duckduckgo_health_search(canonical, max_results=5)
    return {"web_docs": web_results, "web_triggered": True}


def validate(state: RumorState) -> dict:
    claim    = state["claim"]
    rag_docs = state.get("rag_docs") or []
    web_docs = state.get("web_docs") or []

    all_docs: List[Document] = list(rag_docs)
    if web_docs:
        all_docs.extend(web_docs_to_documents(web_docs))

    doc_texts = [
        d.page_content if isinstance(d, Document) else str(d)
        for d in all_docs
    ]

    validation_result: ValidationOutput = (
        validation_prompt | jsonmodel | parser
    ).invoke({
        "claims":    [claim["canonical_text"]],
        "documents": [doc_texts],
    })
    return {"plain_json": claim, "validation": validation_result}


def create_cluster(state: RumorState) -> dict:
    df     = _load_cluster_df()
    new_id = _next_cluster_id(df)
    new_row = pd.DataFrame([{
        "cluster_id":               new_id,
        "embedding_representation": json.dumps(state["embedding"]),
        "participants":             json.dumps([state["claim"]]),
    }])
    pd.concat([df, new_row], ignore_index=True).to_csv(CLUSTER_CSV_PATH, index=False)
    return {"new_cluster_id": new_id}

# ── routing ───────────────────────────────────────────────────────────────────
def route_after_similarity(state: RumorState) -> str:
    score = state.get("sim_score") or 0.0
    return "append_to_cluster" if score >= SIMILARITY_THRESHOLD else "rag_retrieve"


def route_after_rag(state: RumorState) -> str:
    scores = state.get("rag_scores") or []
    if not scores or max(scores) < RAG_WEAK_THRESHOLD:
        return "web_search_fallback"
    return "validate"

# ── graph assembly ────────────────────────────────────────────────────────────
def build_graph():
    builder = StateGraph(RumorState)
    builder.add_node("extract_claim",      extract_claim)
    builder.add_node("attach_embedding",   attach_embedding)
    builder.add_node("similarity_check",   similarity_check)
    builder.add_node("append_to_cluster",  append_to_cluster)
    builder.add_node("rag_retrieve",       rag_retrieve)
    builder.add_node("web_search_fallback",web_search_fallback)
    builder.add_node("validate",           validate)
    builder.add_node("create_cluster",     create_cluster)

    builder.set_entry_point("extract_claim")
    builder.add_edge("extract_claim",    "attach_embedding")
    builder.add_edge("attach_embedding", "similarity_check")

    builder.add_conditional_edges(
        "similarity_check", route_after_similarity,
        {"append_to_cluster": "append_to_cluster", "rag_retrieve": "rag_retrieve"}
    )

    # HIT path also goes to validate before END
    builder.add_edge("append_to_cluster", "rag_retrieve")

    builder.add_conditional_edges(
        "rag_retrieve", route_after_rag,
        {"web_search_fallback": "web_search_fallback", "validate": "validate"}
    )

    builder.add_edge("web_search_fallback", "validate")
    builder.add_edge("validate",            "create_cluster")
    builder.add_edge("create_cluster",      END)

    return builder.compile()

graph = build_graph()

# ── CLI helpers ───────────────────────────────────────────────────────────────
def run_pipeline(rumor: str) -> dict:
    return graph.invoke({
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
    })


def print_result(result: dict) -> None:
    sep = "=" * 60
    print(f"\n{sep}")
    is_hit = result.get("matched_cluster_id") is not None
    claim  = result.get("claim") or result.get("plain_json") or {}

    if is_hit:
        print(f"PATH           : HIT  (cluster {result['matched_cluster_id']})")
        print(f"sim_score      : {result['sim_score']:.4f}")
    else:
        print(f"PATH           : MISS → new cluster {result.get('new_cluster_id')}")
        print(f"sim_score      : {result['sim_score']:.4f}")

    print(f"canonical_text : {claim.get('canonical_text')}")
    print(f"claim_type     : {claim.get('claim_type')}")
    print(f"entities       : {claim.get('entities')}")

    val = result.get("validation")
    if val and val.results:
        print(f"\nVERDICT")
        for i, lbl in enumerate(val.results):
            print(f"  [{i}] {lbl.verdict}")

    rag_docs = result.get("rag_docs") or []
    print(f"\nRAG DOCS ({len(rag_docs)} retrieved)")
    for i, doc in enumerate(rag_docs):
        content = doc.page_content if isinstance(doc, Document) else str(doc)
        meta    = doc.metadata    if isinstance(doc, Document) else {}
        print(f"  [{i}] {content[:120]}")
        if meta:
            print(f"       source={meta.get('source','')}  year={meta.get('year','')}  credibility={meta.get('credibility','')}")

    web_docs = result.get("web_docs") or []
    if web_docs:
        print(f"\nWEB DOCS ({len(web_docs)} fetched)")
        for i, d in enumerate(web_docs):
            print(f"  [{i}] [{d.get('source','')}] {d.get('snippet','')[:120]}")
            print(f"       {d.get('url','')}")

    print(f"{sep}\n")


if __name__ == "__main__":
    result = run_pipeline(
        "Early detection through screening significantly does not improve the chances "
        "of successful treatment for colorectal cancer."
    )
    print_result(result)