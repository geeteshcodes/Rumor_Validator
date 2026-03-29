# 🔬 Rumor Detection Lab

A health-claim fact-checking pipeline built on **Mistral AI**, **LangGraph**, **ChromaDB**, and **Streamlit**. Paste any health rumor and the system extracts an atomic claim, clusters it against known rumors, retrieves supporting evidence, and returns a structured verdict — all with a live step-by-step console showing exactly what the pipeline is doing.

---

## How it works

```
rumor text
    │
    ▼
[extract_claim] ──── LLM extracts one atomic claim + canonical sentence
    │
    ▼
[attach_embedding] ── Mistral embeds the canonical text
    │
    ▼
[similarity_check] ── cosine similarity vs all cluster centroids in cluster.csv
    │
    ├── score ≥ 0.85 → [append_to_cluster] ─┐
    │                                        │
    └── score < 0.85 ──────────────────────┐ │
                                           ▼ ▼
                                     [rag_retrieve] ── query ChromaDB (k=2)
                                           │
                                    ┌──────┴──────┐
                          best score │             │ best score
                             ≥ 0.40  │             │ < 0.40 or empty
                                     ▼             ▼
                               [validate]   [web_search_fallback]
                                                   │
                                                   ▼
                                             [validate] ── LLM verdict
                                                   │
                                                   ▼
                                          [create_cluster] → cluster.csv
```

Every claim — whether it matched an existing cluster or not — goes through RAG and validation. A cluster HIT means the rumor is semantically similar to one already seen; the verdict still runs so you always get a `supported / contradicted / conflicting / insufficient` label.

---

## Features

- **Atomic claim extraction** — the LLM decomposes free-form text into a single subject→relation→object triple, assigns a claim type (`health`, `death`, `policy`, `event`, `statistic`, `relationship`, `other`), extracts entities, time, and location, and produces a ≤12-word canonical sentence used for embedding and display
- **Vector clustering** — claim embeddings are compared against stored cluster centroids using cosine similarity; threshold-based routing determines HIT vs MISS
- **RAG validation** — ChromaDB retrieves the two closest fact documents; explicit cosine scores decide whether the evidence is strong enough to validate against
- **Web fallback** — when RAG scores fall below 0.40 or the index is empty/corrupted, DuckDuckGo is queried and results are filtered to 19 trusted health domains (WHO, CDC, NIH, PubMed, Lancet, NEJM, BMJ, etc.)
- **Structured verdicts** — a strict LLM classifier returns one of four labels with no hallucination path: it can only use the documents it was given
- **Persistent clusters** — `cluster.csv` stores cluster ID, the founding embedding, and all participant claims as JSON; new runs append or create rows without a database server
- **Live pipeline console** — Streamlit streams node-by-node output including similarity bar, doc sources with scores, web results, and verdict as each step completes
- **Graceful Chroma failure** — if the HNSW index is missing or corrupted, the pipeline falls through to web search instead of crashing

---

## Project structure

```
.
├── rumor.py          # core pipeline: models, schemas, graph, nodes, routing
├── app.py            # Streamlit dashboard (imports from rumor.py)
├── cluster.csv       # persistent cluster store (auto-created on first run)
├── rumor_detection/  # ChromaDB persist directory
├── .env              # API keys (see setup)
└── README.md
```

`rumor.py` is framework-agnostic — you can call `run_pipeline(rumor_text)` directly from a FastAPI endpoint, a CLI script, or a notebook without touching the Streamlit layer.

---

## Setup

**1. Clone and create environment**

```bash
git clone https://github.com/your-username/rumor-detection-lab.git
cd rumor-detection-lab
conda create -n rumor python=3.11
conda activate rumor
```

**2. Install dependencies**

```bash
pip install streamlit langchain langchain-mistralai langchain-chroma \
            langgraph chromadb pydantic python-dotenv \
            numpy pandas requests beautifulsoup4
```

**3. Set API key**

Create a `.env` file:

```
MISTRAL_API_KEY=your_mistral_api_key_here
```

Get a key at [console.mistral.ai](https://console.mistral.ai).

**4. Seed ChromaDB (optional)**

If you have a facts corpus, ingest it into ChromaDB under the `facts` collection in the `rumor_detection/` directory. The pipeline will work without it — it falls back to web search — but RAG validation will be stronger with a seeded store.

**5. Run**

```bash
# Streamlit dashboard
streamlit run app.py

# CLI (single rumor)
python rumor.py
```

---

## Configuration

All tunable constants live at the top of `rumor.py`:

| Constant | Default | Effect |
|---|---|---|
| `SIMILARITY_THRESHOLD` | `0.85` | Cosine score above which a rumor is assigned to an existing cluster |
| `RAG_WEAK_THRESHOLD` | `0.40` | Minimum RAG doc score before falling back to web search |
| `CLUSTER_CSV_PATH` | `cluster.csv` | Path to the cluster persistence file |

Trusted health domains for web fallback are defined in the `HEALTH_DOMAINS` list and can be extended freely.

---

## Output schema

`run_pipeline()` returns a dict with these keys:

| Key | Present when | Description |
|---|---|---|
| `claim` | always | Extracted claim dict (canonical_text, claim_type, entities, time, location) |
| `sim_score` | always | Best cosine similarity against existing clusters |
| `matched_cluster_id` | HIT | Cluster ID the rumor was assigned to |
| `new_cluster_id` | MISS | Newly created cluster ID |
| `rag_docs` | always | List of LangChain `Document` objects retrieved from ChromaDB |
| `rag_scores` | always | Cosine similarity score for each RAG doc |
| `web_docs` | web fallback | List of dicts with `snippet`, `source`, `url`, `title` |
| `web_triggered` | web fallback | `True` if DuckDuckGo was used |
| `validation` | always | `ValidationOutput` with a `results` list of `ClaimLabel` objects |

---

## Tech stack

| Component | Library |
|---|---|
| LLM | `mistral-small-latest` via `langchain-mistralai` |
| Embeddings | Mistral embeddings via `langchain-mistralai` |
| Pipeline orchestration | LangGraph `StateGraph` |
| Vector store | ChromaDB via `langchain-chroma` |
| Structured output | Pydantic v2 + `PydanticOutputParser` |
| Web search | DuckDuckGo Instant Answer API + HTML scrape fallback |
| UI | Streamlit |
| Cluster persistence | CSV via pandas |

---

## Limitations

- The pipeline extracts **one** atomic claim per rumor. Compound claims with multiple independent facts should be split before submission.
- Web fallback depends on DuckDuckGo returning snippets from health domains; results can be sparse for niche claims.
- `cluster.csv` uses the embedding of the **founding claim** as the cluster centroid. There is no centroid re-computation as new participants join — a dedicated re-clustering step would improve accuracy over time.
- Verdict quality scales directly with the quality of the ChromaDB fact corpus. An empty or small corpus will route most claims to web fallback.
