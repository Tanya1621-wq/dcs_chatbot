# DCS Issue Resolution Chatbot

A multilingual (Hindi · Hinglish · English) support assistant for the
**Digital Crop Survey (DCS)** application. Users describe issues in natural,
imperfect language; the bot returns the matching issue + resolution steps from
a curated knowledge base.

## What it does

- Accepts queries like *"app nahi chal raha"*, *"fallow land option nahi aa raha"*,
  or *"error 503 aa raha hai"*.
- Hybrid retrieval: **semantic search (FAISS + multilingual sentence
  transformers)** combined with **fuzzy matching (rapidfuzz)**.
- Returns the top 3 KB matches with confidence scores.
- Falls back to **Groq LLM** for clarifying questions when no result is
  confident enough.
- Streamlit UI with chat history, structured response cards, feedback
  thumbs-up/down, and an admin panel for KB upload.

## Architecture

```
              ┌──────────────────────────────────────────┐
 user query → │ preprocess → embed (paraphrase-          │
              │   multilingual-MiniLM-L12-v2) → FAISS    │──┐
              │ + rapidfuzz (title + category)           │  │ blended
              └──────────────────────────────────────────┘  │ score
                                                            ▼
                                       top-3 ranked SearchResult list
                                                            │
                              confident? (>= CONFIDENCE_THRESHOLD)
                                ┌──────────┴───────────┐
                              yes                      no
                                │                       │
                  show structured KB cards     ask Groq for a clarifying
                  (title, category, steps,    response, grounded with the
                   confidence bar)            top KB candidates
```

### Why hybrid?

- The multilingual transformer handles intent and Hindi/Hinglish variants
  ("nahi chal raha" → *not working*).
- Fuzzy matching catches literal tokens the embedder under-weights — error
  codes ("503", "429"), product strings ("fallow land", "Aadhaar").
- Final score = `SEMANTIC_WEIGHT * cosine + FUZZY_WEIGHT * fuzzy_ratio`
  (defaults `0.7` / `0.3`, overridable in `.env`).

### Why a Groq fallback?

When confidence is low we **never invent steps**. Instead Groq is given the
top KB candidates and instructed to either (a) restate the closest one in the
user's language, or (b) ask one targeted clarifier — the exact error text,
where it appears, the user's role.

## Project layout

```
project_root/
├── app/
│   ├── main.py            # Streamlit entry point
│   └── chat_ui.py         # chat panel + admin sidebar
├── services/
│   ├── embedding_service.py   # SentenceTransformer + FAISS
│   ├── search_service.py      # hybrid ranker + threshold gate
│   ├── fuzzy_service.py       # rapidfuzz scoring
│   ├── groq_service.py        # LLM fallback
│   └── kb_service.py          # XLSX/CSV ingest + SQLite + logs
├── data/
│   ├── knowledge_base.xlsx    # (or .csv) source KB
│   └── vector_store/          # persisted FAISS index + metadata
├── db/database.db             # SQLite: kb_entries, query_logs, feedback, meta
├── utils/
│   ├── config.py
│   └── logger.py
├── .env.example
├── requirements.txt
└── README.md
```

## Setup

### 1. Install dependencies

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> First run downloads the `paraphrase-multilingual-MiniLM-L12-v2` model
> (~470 MB) into the local HuggingFace cache.

### 2. Configure secrets

```powershell
Copy-Item .env.example .env
# then edit .env and set GROQ_API_KEY + ADMIN_PASSWORD
```

| Variable               | Purpose                                                |
| ---------------------- | ------------------------------------------------------ |
| `GROQ_API_KEY`         | Required for the LLM fallback. Bot still works without — it returns the closest KB match and asks for clarification offline. |
| `GROQ_MODEL`           | Defaults to `llama-3.3-70b-versatile`.                |
| `ADMIN_PASSWORD`       | Unlocks the KB upload panel in the sidebar.            |
| `SEMANTIC_WEIGHT`      | Default `0.7`.                                        |
| `FUZZY_WEIGHT`         | Default `0.3`.                                        |
| `CONFIDENCE_THRESHOLD` | Default `0.45`. Below this triggers Groq fallback.    |

### 3. Place the knowledge base

Drop `knowledge_base.xlsx` (or `.csv`) into `data/`. Required columns:

| issue\_id | issue\_title | issue\_category | issue\_description | resolution\_steps |
| --------- | ------------ | --------------- | ------------------ | ----------------- |

Only `issue_title` and `resolution_steps` are strictly required. A starter KB
of 25 DCS issues is shipped in `data/knowledge_base.csv`.

### 4. Run

```powershell
streamlit run app/main.py
```

Open the URL Streamlit prints (defaults to <http://localhost:8501>).

## Updating the knowledge base

Two options:

**Admin UI (recommended):** unlock the sidebar with `ADMIN_PASSWORD`, upload
a new XLSX/CSV, choose **Replace** or **Append**, click *Apply update*. The
file is persisted to `data/`, the SQLite tables are refreshed, and the FAISS
index is rebuilt — all in one click.

**By hand:** drop a new file at `data/knowledge_base.xlsx`, delete
`data/vector_store/*` and `db/database.db`, and restart Streamlit. The first
chat triggers ingest + index build.

## Storage

- **SQLite** (`db/database.db`): canonical store for KB rows, query logs, and
  feedback. Schema is created on first run.
- **FAISS** (`data/vector_store/kb.index` + `kb_meta.pkl`): persisted
  `IndexFlatIP` over L2-normalized embeddings. Reloaded on startup; rebuilt
  whenever the row-id list diverges from the DB.

## Telemetry

Every query is logged with timestamp, matched row, confidence, and source
(`hybrid` | `groq` | `offline_fallback`). Feedback buttons write a row keyed
to that query log so you can compute *helpfulness %* per source over time.

## Operational notes

- The model and FAISS index are loaded **once per process** via
  `@st.cache_resource`. The first request is slow; the rest are sub-100 ms.
- The bot is fully **stateless across processes** — restart anytime; nothing
  is held only in memory.
- No keyword-matching shortcuts. All retrieval routes through the hybrid
  pipeline.
- No secrets in code. `.env` is git-ignored; `.env.example` is the template.

## Troubleshooting

| Symptom                                            | Fix                                                                                |
| -------------------------------------------------- | ---------------------------------------------------------------------------------- |
| "Groq fallback ⚠️ offline (no API key)" in sidebar | Add `GROQ_API_KEY` to `.env` and restart.                                          |
| KB count shows 0                                   | Place `knowledge_base.xlsx`/`.csv` in `data/`, or upload via admin panel.          |
| Stale index after editing the CSV by hand          | Click **Rebuild embeddings** (admin) or delete `data/vector_store/*` and restart.  |
| First query takes 30+ s                            | Expected — sentence-transformer model is downloading. Subsequent queries are fast. |
