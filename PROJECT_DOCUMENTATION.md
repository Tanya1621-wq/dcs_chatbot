# DCS Issue Resolution Chatbot — Complete Documentation

> Audience: Python developers from beginner to senior. Read top to bottom on
> first pass; use the Table of Contents as a reference afterward.
>
> Each non-trivial code block has two readings: an **In plain words** sentence
> for newcomers, and a **Technical note** for senior readers.

---

## Table of Contents 

1. [Project Overview](#1-project-overview)
2. [End-to-End Workflow](#2-end-to-end-workflow)
   - [2.1 Architecture diagram](#21-architecture-diagram)
   - [2.2 Runtime flow (step by step)](#22-runtime-flow-step-by-step)
   - [2.3 Data flow](#23-data-flow)
3. [Project Structure](#3-project-structure)
4. [Setup and Run](#4-setup-and-run)
5. [File-by-file Walkthrough](#5-file-by-file-walkthrough)
   - [5.1 `app/main.py`](#51-appmainpy)
   - [5.2 `app/chat_ui.py`](#52-appchat_uipy)
   - [5.3 `app/__init__.py`](#53-app__init__py)
   - [5.4 `services/embedding_service.py`](#54-servicesembedding_servicepy)
   - [5.5 `services/fuzzy_service.py`](#55-servicesfuzzy_servicepy)
   - [5.6 `services/rerank_service.py`](#56-servicesrerank_servicepy)
   - [5.7 `services/search_service.py`](#57-servicessearch_servicepy)
   - [5.8 `services/groq_service.py`](#58-servicesgroq_servicepy)
   - [5.9 `services/kb_service.py`](#59-serviceskb_servicepy)
   - [5.10 `services/__init__.py`](#510-services__init__py)
   - [5.11 `utils/config.py`](#511-utilsconfigpy)
   - [5.12 `utils/logger.py`](#512-utilsloggerpy)
   - [5.13 `utils/__init__.py`](#513-utils__init__py)
6. [Non-Python Files](#6-non-python-files)
7. [Python Concepts Reference](#7-python-concepts-reference)
8. [Glossary](#8-glossary)
9. [How to Make Changes (Cookbook)](#9-how-to-make-changes-cookbook)
10. [Known Limitations & Future Work](#10-known-limitations--future-work)

---

## 1. Project Overview

**DCS Issue Resolution Chatbot** is a multilingual support assistant for the
**Digital Crop Survey (DCS)** application used by Indian agriculture
surveyors and field officers. Users describe a problem in natural,
imperfect language — Hindi, Hinglish (Hindi typed in Latin letters), or
English — and the bot returns the matching issue and resolution steps from
a curated knowledge base.

The bot does **hybrid retrieval**: a multilingual sentence-transformer
embeds the query and the KB into the same vector space (semantic match),
while `rapidfuzz` adds a literal-token signal so error codes and product
names are not lost in translation. A cross-encoder reranker re-scores the
top candidates for precision. If the top score is still below a
configurable confidence threshold, the bot falls back to **Groq** (an LLM
provider) — the LLM is *grounded* with the top KB candidates so it never
invents a resolution out of thin air.

The UI is a Streamlit app with structured response cards, thumbs-up/down
feedback, and an admin panel for uploading new knowledge bases.

### Who uses it
- Field surveyors hitting issues on the DCS Android app.
- District / state DCS coordinators who escalate or resolve issues.
- KB editors who upload curated problem/resolution rows via the admin
  panel.

### Tech stack at a glance

| Layer            | Library                                         | Role                                       |
| ---------------- | ----------------------------------------------- | ------------------------------------------ |
| UI               | `streamlit`                                     | Chat panel + admin sidebar                 |
| Embeddings       | `sentence-transformers`                         | Multilingual MiniLM bi-encoder + reranker  |
| Vector index     | `faiss-cpu`                                     | Fast cosine similarity over normalized vecs |
| Fuzzy matching   | `rapidfuzz`                                     | Token-set ratio for literal-keyword overlap |
| LLM fallback     | `groq` (Llama 3.3 70B by default)               | Grounded answer or clarifier               |
| Storage          | `sqlite3` (stdlib) + on-disk FAISS pickle       | KB rows, query logs, feedback              |
| Data handling    | `pandas`, `openpyxl`                            | Read XLSX / CSV KB                         |
| Config           | `python-dotenv`                                 | Load `.env` into `os.environ`              |

### Status / maturity

This is a **working application**, not a library. It is single-process,
file-backed (SQLite + FAISS index on disk), and intended to be deployed as
a Streamlit service. There is no test suite, no CI, no Dockerfile, and no
deployment automation in the repo today.

---

## 2. End-to-End Workflow

### 2.1 Architecture diagram

```
┌──────────────────────────────────────────────────────────────────────────┐
│                             Streamlit process                            │
│                                                                          │
│                ┌──────────────────────┐                                  │
│   browser ◄──► │   app/chat_ui.py     │                                  │
│                │  (UI + orchestration)│                                  │
│                └──────────┬───────────┘                                  │
│                           │                                              │
│      ┌────────────────────┼─────────────────────────┐                    │
│      ▼                    ▼                         ▼                    │
│ ┌──────────┐       ┌────────────────┐        ┌────────────┐              │
│ │ kb_      │       │ search_        │        │ groq_      │              │
│ │ service  │◄──────┤ service        │        │ service    │──► Groq API  │
│ │ (SQLite) │       │ (HybridSearcher│        │ (LLM)      │              │
│ └────┬─────┘       └─┬───────┬──────┘        └────────────┘              │
│      │               │       │                                           │
│      │               ▼       ▼                                           │
│      │      ┌────────────┐ ┌────────────────┐                            │
│      │      │ embedding_ │ │ fuzzy_service  │                            │
│      │      │ service    │ │ (rapidfuzz)    │                            │
│      │      │ (FAISS +   │ └────────────────┘                            │
│      │      │ MiniLM)    │                                               │
│      │      └─────┬──────┘ ┌────────────────┐                            │
│      │            │        │ rerank_service │                            │
│      │            │        │ (CrossEncoder) │                            │
│      │            │        └────────────────┘                            │
│      ▼            ▼                                                      │
│  db/database.db   data/vector_store/kb.index + kb_meta.pkl               │
└──────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Runtime flow (step by step)

What happens from `streamlit run app/main.py` to a rendered answer:

1. **Process boot.** Streamlit imports `app/main.py`. The script prepends
   the project root to `sys.path` and calls `chat_ui.render()`.
2. **First-time bootstrap.** `render()` calls `_bootstrap()` which is
   wrapped in `@st.cache_resource` — so it runs once per process. It:
   - Calls `kb_service.init_db()` to create SQLite tables if missing.
   - Calls `kb_service.kb_count()`. If zero, calls `load_kb_into_db()`
     which reads `data/knowledge_base.xlsx` (or `.csv`) into the
     `kb_entries` table.
   - Creates the `HybridSearcher` singleton via `get_searcher()` and calls
     `ensure_loaded()`. That tries to load `data/vector_store/kb.index`;
     if absent or stale (row-id list differs from the DB), it rebuilds
     embeddings with `paraphrase-multilingual-MiniLM-L12-v2` and saves
     them.
3. **Sidebar.** `_render_sidebar()` paints "Clear chat" + the admin panel
   (gated by `ADMIN_PASSWORD`).
4. **Chat input.** When the user submits a message:
   - `groq_service.rewrite_query(prompt)` (optional, skipped if no API
     key) trims filler words to make a tighter search query.
   - `searcher.search(query)` runs the **two-stage retrieval**:
     - *Stage 1:* FAISS top-K semantic + rapidfuzz over titles+categories
       → blended `hybrid = SEMANTIC_WEIGHT * sem + FUZZY_WEIGHT * fuz`.
     - *Stage 2:* if the cross-encoder reranker is available, re-score
       the top-`RERANK_TOP_K` pool and replace the ranking signal.
   - `searcher.is_low_confidence(results)` checks whether the top score
     ≥ `CONFIDENCE_THRESHOLD`.
5. **Branch on confidence.**
   - **High confidence:** render structured cards (title, category,
     resolution, confidence bar). Log to `query_logs` with
     `source="hybrid"`.
   - **Low confidence:** call `groq_service.fallback_answer(query,
     candidates, chat_history)`. Groq is given the candidates and
     instructed to either restate the closest match in the user's
     language or ask one clarifier. Log with `source="groq"` (or
     `"offline_fallback"` if Groq is not configured).
6. **Feedback.** Each answer renders 👍 / 👎 buttons. Clicking writes a
   row to `feedback` keyed to the `query_log` id.

### 2.3 Data flow

| Data            | Lives in                                | Written by            | Read by                      |
| --------------- | --------------------------------------- | --------------------- | ---------------------------- |
| KB rows         | `db/database.db` (`kb_entries`)         | `kb_service.load_kb_into_db` | `HybridSearcher.ensure_loaded` |
| KB source file  | `data/knowledge_base.xlsx` or `.csv`    | admin upload / hand   | `kb_service._read_kb_file`   |
| FAISS index     | `data/vector_store/kb.index`            | `FaissIndex.save`     | `FaissIndex.load`            |
| FAISS metadata  | `data/vector_store/kb_meta.pkl`         | `FaissIndex.save`     | `FaissIndex.load`            |
| Query logs      | `db/database.db` (`query_logs`)         | `kb_service.log_query`| feedback aggregation         |
| Feedback        | `db/database.db` (`feedback`)           | `kb_service.log_feedback` | `feedback_summary`       |
| LLM-generated paraphrases | `kb_entries.paraphrases` (JSON-encoded) | admin tool       | `KBEntry.search_text`        |
| Secrets         | `.env`                                  | human                 | `utils/config.py`            |

---

## 3. Project Structure

```
dcs_chatbot/
├── app/
│   ├── __init__.py            # empty — marks the folder as a package
│   ├── main.py                # Streamlit entry point (sys.path bootstrap)
│   └── chat_ui.py             # the whole UI + orchestration
├── services/
│   ├── __init__.py            # empty — package marker
│   ├── embedding_service.py   # SentenceTransformer + FAISS wrapper
│   ├── fuzzy_service.py       # rapidfuzz scoring
│   ├── rerank_service.py      # cross-encoder reranker (stage 2)
│   ├── search_service.py      # HybridSearcher (stage 1+2 orchestrator)
│   ├── groq_service.py        # Groq LLM client + prompts + fallback
│   └── kb_service.py          # KBEntry dataclass, SQLite, KB ingest
├── utils/
│   ├── __init__.py            # empty — package marker
│   ├── config.py              # env-var loading + paths + tunables
│   └── logger.py              # one shared logging configuration
├── data/
│   ├── knowledge_base.csv     # starter KB (25 DCS issues)
│   └── vector_store/          # persisted FAISS index (gitignored)
│       └── .gitkeep
├── db/                        # created at runtime (gitignored)
│   └── database.db
├── .env                       # user-supplied secrets (gitignored)
├── .gitignore
├── requirements.txt
└── README.md
```

A few observations about the layout:

- The split between `app/`, `services/`, and `utils/` is the conventional
  three-tier separation: presentation, domain logic, infrastructure.
- `services/__init__.py`, `app/__init__.py`, `utils/__init__.py` are all
  **empty** (0 bytes). They exist purely so Python recognises the folders
  as packages — see [§ 5.3](#53-app__init__py), [§ 5.10](#510-services__init__py),
  [§ 5.13](#513-utils__init__py).
- There is no `db/` checked into git — it is created on first run by
  `utils/config.py` (line 36-37) and populated by `kb_service.init_db()`.

---

## 4. Setup and Run

### Prerequisites

- Windows 10/11, macOS, or Linux.
- Python 3.10 or newer (the code uses PEP 604 / 585 builtin generics like
  `list[str]` and `dict[int, float]`).
- ~1 GB of disk for the embedding model + reranker download cache.
- A Groq API key (optional; the bot still works offline with a graceful
  fallback message).

### Install

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**In plain words:** create a fresh Python sandbox in `.venv/`, activate
it, and install the libraries listed in `requirements.txt`.

**Technical note:** the listed deps include `sentence-transformers`,
`faiss-cpu`, `rapidfuzz`, `groq`, `streamlit`, `pandas`, `openpyxl`,
`python-dotenv`, `numpy`. First run will additionally pull the
`paraphrase-multilingual-MiniLM-L12-v2` checkpoint (~470 MB) and the
`BAAI/bge-reranker-v2-m3` checkpoint into the HuggingFace cache.

### Configure

Create `.env` in the project root:

```
GROQ_API_KEY=sk_...        # optional but recommended
GROQ_MODEL=llama-3.3-70b-versatile
ADMIN_PASSWORD=changeme
SEMANTIC_WEIGHT=0.7
FUZZY_WEIGHT=0.3
CONFIDENCE_THRESHOLD=0.45
RERANKER_ENABLED=true
RERANK_TOP_K=15
```

All values are optional. Defaults live in `utils/config.py` and are sane
for the shipped KB.

### Run

```powershell
streamlit run app/main.py
```

The Streamlit URL (default `http://localhost:8501`) will appear in the
console.

### Verify

1. Sidebar should show "DCS Support Bot".
2. Type "error 503". You should see a structured card titled *Error 503*
   with confidence ≈ 90%+.
3. Type "app nahi chal raha". You should still get a card or — with low
   confidence — a Groq follow-up question.

---

## 5. File-by-file Walkthrough

### 5.1 `app/main.py`

**One-line summary:** Streamlit entry point that sets `sys.path` and
hands off to `chat_ui.render()`.

**Why it exists.** Streamlit launches the script you point it at *as a
top-level script*, not as `python -m app.main`. That means relative
imports won't work and `import services.kb_service` will fail unless the
project root is on `sys.path`. This file makes that work.

**Public surface.** None. Streamlit just executes the module top-to-bottom.

**Walkthrough:**

```python
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.chat_ui import render  # noqa: E402
render()
```

**In plain words:** find the project's top folder, make sure Python looks
there for imports, then call the function that draws the chat page.

**Technical note:** `Path(__file__).resolve().parent.parent` resolves
symlinks then walks two levels up: `app/main.py` → `app/` → project root.
The `# noqa: E402` silences flake8's "module-level import not at top of
file" lint, which is unavoidable here because the path mutation has to
happen *before* the `app.chat_ui` import is attempted.

**Python concepts used:** `pathlib.Path` (§ 7.1), `sys.path` mutation,
`from __future__ import annotations` (§ 7.2).

**Connections:**
- Imports: `app.chat_ui` (only public consumer is `render`).
- Imported by: nothing — this is the script Streamlit executes.

---

### 5.2 `app/chat_ui.py`

**One-line summary:** The full Streamlit UI — chat panel, admin sidebar,
result cards, feedback buttons.

**Why it exists.** This is the controller. Everything the user sees is
defined here, and every back-end service (`kb_service`, `search_service`,
`groq_service`) is invoked from this file.

**Public surface:**

| Symbol      | Kind     | Used by                  |
| ----------- | -------- | ------------------------ |
| `render()`  | function | `app/main.py`            |

Internal helpers (`_bootstrap`, `_result_card`, `_render_sidebar`,
`_render_admin`, `_render_chat`, `_render_feedback_buttons`,
`_render_feedback_summary`, `_chat_history_for_groq`) are private.

**Walkthrough:**

#### Cached bootstrap (chat_ui.py:16-26)

```python
@st.cache_resource(show_spinner="Loading retrieval engine…")
def _bootstrap():
    kb_service.init_db()
    if kb_service.kb_count() == 0:
        loaded = kb_service.load_kb_into_db()
        log.info("Bootstrapped KB with %d rows", loaded)
    searcher = get_searcher()
    searcher.ensure_loaded()
    return searcher
```

**In plain words:** the first time someone asks a question, set up the
database, load the knowledge base if empty, and build the search index.
After that, reuse the same searcher object for every visitor.

**Technical note:** `@st.cache_resource` is Streamlit's "compute once per
process" decorator (different from `@st.cache_data` which keys on inputs).
Because `_bootstrap` takes no arguments, it runs exactly once and the
cached return value is shared across all sessions in the same Streamlit
process. The 30+ second model download cost is paid once.

#### Result card (chat_ui.py:31-45)

```python
def _result_card(result: SearchResult) -> None:
    e = result.entry
    st.markdown(f"### {e.title}")
    ...
    st.progress(min(max(result.score, 0.0), 1.0),
                text=f"Confidence: {result.score:.0%}")
```

Renders a single KB hit as a heading, optional metadata line, the issue
description, the resolution steps, and a horizontal progress bar showing
confidence. The `min(max(..., 0.0), 1.0)` clamp protects against
floating-point drift outside the [0, 1] range.

#### Main render flow (chat_ui.py:59-68)

```python
def render() -> None:
    st.set_page_config(page_title="DCS Issue Resolution Chatbot",
                       page_icon="🌾", layout="wide")
    searcher = _bootstrap()
    _render_sidebar(searcher)
    _render_chat(searcher)
```

Three lines of UI logic: configure the tab title/icon, pull the cached
searcher, then paint sidebar and chat in sequence.

#### Admin panel (chat_ui.py:88-181)

The admin block is gated by a password check against `ADMIN_PASSWORD`.
Once unlocked, it offers four actions:

| Button                          | Effect                                                                |
| ------------------------------- | --------------------------------------------------------------------- |
| **Apply update** (after upload) | Persist file, reload SQLite, rebuild FAISS index.                     |
| **Rebuild embeddings**          | Force-rebuild the FAISS index from current SQLite rows.               |
| **Generate query paraphrases** | Use Groq to generate 6-8 paraphrases per KB row, store as JSON, rebuild. |
| **Lock admin**                  | Flip `is_admin` back off in the session.                              |

The paraphrase loop (chat_ui.py:156-177) is the only place in the UI that
uses `st.progress` to track an iterative job — important because Groq
calls take ~1 s each and the user needs feedback.

#### Chat handler (chat_ui.py:197-280)

The flow: replay history → take new prompt → optional Groq query rewrite
→ `searcher.search()` → branch on `is_low_confidence`. The high-conf
branch renders cards (and stores `kind="result_card"` in the message so
replay later can re-render them); the low-conf branch shows a single
markdown reply from Groq.

**In plain words:** for each new question, see if the KB has a confident
match. If yes, show the matches. If no, ask Groq for help — but show Groq
the candidates so it doesn't make things up.

**Technical note:** the message dict format is the project's own — it
distinguishes a `result_card` message from a plain text one with a `kind`
key. This avoids serialising Streamlit widgets and lets a re-render walk
the same `SearchResult` objects.

**Python concepts used:** decorators (§ 7.5), f-strings (§ 7.6), type
hints (§ 7.7), `dict.get` with default (§ 7.8), conditional list
comprehensions (§ 7.9), `st.session_state` mutation, Streamlit's `with
st.chat_message(...)` context manager.

**Connections:**
- Imports: `streamlit`, `services.groq_service`, `services.kb_service`,
  `services.search_service`, `utils.config`, `utils.logger`.
- Imported by: `app/main.py` only.

---

### 5.3 `app/__init__.py`

**One-line summary:** Empty package marker.

**Why it exists.** A directory becomes a Python package when it contains
`__init__.py`. This file is **0 bytes** — it intentionally exports
nothing. Without it, `from app.chat_ui import render` in `main.py` would
fail.

**Walkthrough:** there is nothing to walk through. The file is empty.

**Python concepts used:** none. The mere existence of the file is the
mechanism — see [§ 7.10 Packages and `__init__.py`](#710-packages-and-__init__py).

**Connections:** referenced implicitly by every `import app.<module>`.

---

### 5.4 `services/embedding_service.py`

**One-line summary:** Wraps a `SentenceTransformer` model and a flat
inner-product FAISS index.

**Why it exists.** Two responsibilities, both centralised here so other
modules don't need to know about FAISS or huggingface model objects:
1. Turn a list of texts into L2-normalised float32 vectors.
2. Persist / load / search those vectors via FAISS.

**Public surface:**

| Symbol           | Kind     | Notes                                                  |
| ---------------- | -------- | ------------------------------------------------------ |
| `embed(texts)`   | function | Returns `np.ndarray` of shape `(N, dim)`, float32, normalised |
| `FaissIndex`     | class    | `.build`, `.save`, `.load`, `.search`                 |
| `FaissIndex.row_ids` | attribute | Aligns FAISS positional ids back to KB row ids   |
| `FaissIndex.dim` | attribute | Vector dimensionality                                  |

**Walkthrough:**

#### Lazy, thread-safe model load (embedding_service.py:28-38)

```python
_model = None
_model_lock = threading.Lock()

def _load_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from sentence_transformers import SentenceTransformer
                _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model
```

**In plain words:** load the AI model the first time we need it, never
twice, and don't crash if two threads ask at the same time.

**Technical note:** this is the **double-checked locking** pattern. The
outer `if _model is None` is a fast path that avoids acquiring the lock
on every call. The inner check, after the lock is held, prevents two
threads from both creating models in a race window. The
`from sentence_transformers import ...` is *inside* the function so
import-time cost (and FutureWarnings from huggingface) only hits when
embeddings are actually needed.

#### Embedding (embedding_service.py:41-49)

```python
def embed(texts: list[str]) -> np.ndarray:
    model = _load_model()
    vecs = model.encode(texts, convert_to_numpy=True,
                        normalize_embeddings=True, show_progress_bar=False)
    return vecs.astype("float32")
```

**In plain words:** ask the model to turn a list of sentences into a
matrix of numbers, with each row scaled so its length is 1.

**Technical note:** `normalize_embeddings=True` ensures L2-norm = 1, so
inner product (FAISS `IndexFlatIP`) equals cosine similarity. The cast
to `float32` is required because FAISS indexes only accept float32, not
the float64 numpy default.

#### `FaissIndex.build` / `.save` / `.load` (embedding_service.py:64-107)

`build` creates a flat inner-product index from a list of texts and
their row ids. `save` writes two files: `kb.index` (binary FAISS format)
and `kb_meta.pkl` (the `row_ids` list and `dim`). `load` returns `None`
on missing or corrupt files so callers can choose to rebuild — this is
the graceful-degradation pattern (§ 7.11).

#### `FaissIndex.search` (embedding_service.py:109-123)

```python
scores, idxs = self.index.search(vec, k)
similarity = max(0.0, min(1.0, float(score)))
```

**In plain words:** ask FAISS for the closest k matches, then squash the
score into the [0, 1] range so the rest of the code can treat it as a
confidence.

**Technical note:** with normalised vectors, IP scores live in [-1, 1].
Clamping to [0, 1] is opinionated but harmless — anything below 0 is
"unrelated" and below the eventual confidence threshold anyway.

**Python concepts used:** `from __future__ import annotations` (§ 7.2),
type hints (§ 7.7), `pathlib.Path` (§ 7.1), `threading.Lock` + double-
checked locking (§ 7.12), pickle (§ 7.13), `@classmethod` (§ 7.14),
lazy imports (§ 7.15).

**Connections:**
- Imports: `numpy`, `faiss` (lazy), `sentence_transformers` (lazy),
  `utils.config`, `utils.logger`.
- Imported by: `services.search_service`.

---

### 5.5 `services/fuzzy_service.py`

**One-line summary:** Token-set fuzzy matching against KB title +
category, normalised to [0, 1].

**Why it exists.** The semantic embedder is great at *meaning* but can
under-weight literal tokens that matter — error codes ("503", "429"),
product strings ("fallow land", "Aadhaar"). Fuzzy matching on the title
catches those.

**Public surface:**

| Symbol                          | Kind     | Notes                                  |
| ------------------------------- | -------- | -------------------------------------- |
| `fuzzy_score(query, title, category)` | function | Single 0..1 score                |
| `score_all(query, entries)`     | function | Returns `{row_id: score}` for every KB |

**Walkthrough:**

```python
def fuzzy_score(query: str, title: str, category: str) -> float:
    if not query.strip():
        return 0.0
    title_score = fuzz.token_set_ratio(query, title or "") / 100.0
    cat_score = fuzz.token_set_ratio(query, category or "") / 100.0
    return 0.8 * title_score + 0.2 * cat_score
```

**In plain words:** compare the user's words to the issue title and
category. Title matches matter four times as much as category matches.

**Technical note:** `token_set_ratio` is order-agnostic — it splits both
strings into tokens, takes their set-intersection and set-difference,
and computes a ratio that ignores word order and duplicate tokens. So
"crop nahi dikh raha" and "nahi dikh crop" score identically. Output is
0–100, divided by 100 to live in the same [0, 1] space as semantic
similarity.

The `or ""` guards are belt-and-braces: although `_normalize_columns` in
`kb_service` already fills NaN with empty strings, defaulting here means
the function works on any callable input.

**Python concepts used:** type hints (§ 7.7), short-circuit `or` for
defaults, dictionary comprehension (§ 7.9).

**Connections:**
- Imports: `rapidfuzz`.
- Imported by: `services.search_service`.

---

### 5.6 `services/rerank_service.py`

**One-line summary:** Optional second-stage cross-encoder reranker that
re-scores the top hybrid candidates for higher precision.

**Why it exists.** Bi-encoders (the FAISS path) are fast because they
encode query and passage independently — but they lose accuracy because
the two encodings never "see" each other. A cross-encoder takes
`(query, passage)` together and produces a much more precise score, at
the cost of needing one model call per candidate. Running it on just
the top 15 from stage 1 keeps latency manageable while improving the
final ranking.

**Public surface:**

| Symbol                              | Kind     | Notes                                |
| ----------------------------------- | -------- | ------------------------------------ |
| `is_available()`                    | function | True iff enabled in config + model loaded |
| `rerank(query, candidates)`         | function | Returns `{row_id: score in [0,1]}`   |

**Walkthrough:**

#### Lazy load with a permanent-failure flag (rerank_service.py:31-52)

```python
_model = None
_model_lock = threading.Lock()
_load_failed = False

def _load_model():
    global _model, _load_failed
    if _model is not None:
        return _model
    if _load_failed or not RERANKER_ENABLED:
        return None
    with _model_lock:
        if _model is None and not _load_failed:
            try:
                from sentence_transformers import CrossEncoder
                _model = CrossEncoder(RERANKER_MODEL_NAME)
            except Exception as exc:
                log.error("Reranker failed to load (%s); disabling …", exc)
                _load_failed = True
                return None
    return _model
```

**In plain words:** try to load the reranker once. If it fails (e.g. no
internet, no disk space), remember the failure and stop trying for the
rest of the process — but keep serving search results without it.

**Technical note:** the `_load_failed` boolean is the project's
graceful-degradation switch. The same double-checked locking pattern as
the embedder, plus a sticky-failure flag so we don't retry the expensive
load on every search request.

#### Sigmoid (rerank_service.py:55-60)

```python
def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)
```

**In plain words:** turn the model's raw score into a number between 0
and 1, even when the input is very negative.

**Technical note:** BGE rerankers output **logits**, not probabilities.
Sigmoid maps logits to (0, 1). The two-branch implementation avoids
`math.exp` overflow when `x` is large and negative — the textbook
single-line `1 / (1 + exp(-x))` would overflow at `x ≈ -710`.

#### `rerank` (rerank_service.py:63-84)

```python
pairs = [[query, text] for _, text in candidates]
raw_scores = model.predict(pairs, show_progress_bar=False)
return {row_id: _sigmoid(float(score))
        for (row_id, _), score in zip(candidates, raw_scores)}
```

Builds `(query, candidate_text)` pairs, scores them in one batch, and
zips back to row ids. The `try/except` around the call returns `{}` on
failure — callers (`search_service`) treat empty as "no rerank info" and
fall back to hybrid scores.

**Python concepts used:** decorators (§ 7.5), threading + locks (§ 7.12),
graceful degradation (§ 7.11), list comprehensions (§ 7.9), `zip` over
parallel iterables, lazy import (§ 7.15).

**Connections:**
- Imports: `sentence_transformers` (lazy), `utils.config`, `utils.logger`.
- Imported by: `services.search_service`.

---

### 5.7 `services/search_service.py`

**One-line summary:** The two-stage retrieval orchestrator —
`HybridSearcher` combines bi-encoder + fuzzy → optional cross-encoder
rerank → returns top-K `SearchResult` objects.

**Why it exists.** This is the public retrieval API. Everything else —
embedding, fuzzy, rerank — is plumbing this file owns. The chat UI only
talks to this module.

**Public surface:**

| Symbol             | Kind                        | Notes                              |
| ------------------ | --------------------------- | ---------------------------------- |
| `SearchResult`     | dataclass                   | Holds entry + scores               |
| `HybridSearcher`   | class                       | `search`, `ensure_loaded`, `rebuild`, `is_ready`, `is_low_confidence` |
| `get_searcher()`   | function                    | Module-level singleton accessor    |

**Walkthrough:**

#### `SearchResult` (search_service.py:30-42)

```python
@dataclass
class SearchResult:
    entry: kb_service.KBEntry
    semantic: float
    fuzzy: float
    score: float
    hybrid: float = 0.0
    rerank: float = 0.0

    @property
    def confidence(self) -> float:
        return self.score
```

**In plain words:** a lightweight bundle holding one matched issue plus
all the scores that produced it.

**Technical note:** `@dataclass` auto-generates `__init__`, `__repr__`,
and `__eq__`. `score` is the *authoritative* ranking signal — it equals
`hybrid` if the reranker is off, or the rerank score if it ran.
`hybrid` and `rerank` are kept around so the UI / logs can inspect both.
The `confidence` property is alias sugar for callers who prefer that
word.

#### `HybridSearcher.ensure_loaded` (search_service.py:58-80)

```python
cached = FaissIndex.load()
entry_ids = [e.row_id for e in entries]
if cached is not None and cached.row_ids == entry_ids:
    self._index = cached
else:
    texts = [e.search_text() for e in entries]
    self._index = FaissIndex.build(entry_ids, texts)
    self._index.save()
```

**In plain words:** if there's a saved index that matches the current
KB rows, use it. Otherwise, build a fresh one and save it.

**Technical note:** the staleness check is intentionally simple — equal
`row_ids` *list* implies same KB. It will not detect mutation of an
existing row's text content (e.g. an admin tweaking a resolution_step
in SQLite directly). The admin "Rebuild embeddings" button is the
escape hatch for that.

#### `HybridSearcher.search` (search_service.py:92-147)

This is the heart of the project. Stage 1:

```python
sem_pool = max(RERANK_TOP_K, top_k * 4, 10)
sem_pairs = self._index.search(query_norm, top_k=sem_pool)
sem_scores = {row_id: s for row_id, s in sem_pairs}
fuz_scores = fuzzy_service.score_all(query_norm, self._entries)

candidate_ids = set(sem_scores) | {rid for rid, s in fuz_scores.items() if s >= 0.5}
```

**In plain words:** ask FAISS for a wide top-K, score every KB row with
fuzzy match, then take the union — anything that scores well on either
gets considered.

**Technical note:** `set(sem_scores)` is the keys of the dict (row ids).
The fuzzy union threshold 0.5 prevents 100% noise from joining the
candidate pool, but is loose enough to catch literal-keyword wins the
embedder missed. The `sem_pool = max(RERANK_TOP_K, top_k * 4, 10)`
sizing ensures the reranker has at least 15 (config-driven) candidates
to refine.

```python
hybrid = SEMANTIC_WEIGHT * sem + FUZZY_WEIGHT * fuz
results.append(SearchResult(entry=entry, semantic=sem, fuzzy=fuz,
                            hybrid=hybrid, rerank=0.0, score=hybrid))
results.sort(key=lambda r: r.hybrid, reverse=True)
```

Each row gets the weighted blend; defaults are `0.7 * semantic + 0.3 *
fuzzy`. Results are sorted by `hybrid` so the top of the pool is the
strongest stage-1 candidates.

Stage 2:

```python
if rerank_service.is_available() and results:
    pool = results[:RERANK_TOP_K]
    pairs = [(r.entry.row_id, r.entry.search_text()) for r in pool]
    rerank_scores = rerank_service.rerank(query_norm, pairs)
    if rerank_scores:
        for r in pool:
            r.rerank = rerank_scores.get(r.entry.row_id, 0.0)
            r.score = r.rerank
        pool.sort(key=lambda r: r.score, reverse=True)
        return pool[:top_k]

return results[:top_k]
```

**In plain words:** if the reranker is available, give it the top 15
hybrid candidates and let it produce a final, more accurate ranking. If
not, just return the hybrid ranking.

**Technical note:** when reranker runs, **`r.score` is overwritten with
the rerank score**. That has consequences for `CONFIDENCE_THRESHOLD` —
the threshold is implicitly being applied to the rerank score in that
branch, not to the hybrid score. If you tune the threshold, do it with
the reranker on or off matching production.

#### `is_low_confidence` (search_service.py:149-153)

```python
@staticmethod
def is_low_confidence(results: list[SearchResult]) -> bool:
    if not results:
        return True
    return results[0].score < CONFIDENCE_THRESHOLD
```

The confidence gate. Empty results count as low confidence. Default
threshold is 0.45.

#### Module-level singleton (search_service.py:166-173)

```python
_singleton: Optional[HybridSearcher] = None

def get_searcher() -> HybridSearcher:
    global _singleton
    if _singleton is None:
        _singleton = HybridSearcher()
    return _singleton
```

This pairs with `@st.cache_resource` in the UI: Streamlit's cache holds a
reference to the same singleton, so reloads do not re-instantiate the
searcher. See [§ 7.16 Module-level singletons](#716-module-level-singletons-vs-streamlit-cache).

**Python concepts used:** `@dataclass` (§ 7.17), `@property` (§ 7.18),
`@staticmethod` (§ 7.18), threading lock (§ 7.12), set comprehension
(§ 7.9), `set(dict)` keys, sorting with `key=lambda` (§ 7.19), module
singleton (§ 7.16).

**Connections:**
- Imports: `services.fuzzy_service`, `services.kb_service`,
  `services.rerank_service`, `services.embedding_service.FaissIndex`,
  `utils.config`, `utils.logger`.
- Imported by: `app.chat_ui`, `services.groq_service`.

---

### 5.8 `services/groq_service.py`

**One-line summary:** Groq LLM client with three prompts — query
rewriter, grounded fallback, paraphrase generator — plus an offline
fallback message.

**Why it exists.** When the search confidence is low we still want a
useful response. The LLM is given the top KB candidates and instructed
to either restate one of them in the user's language or ask a focused
clarifier. It is **never** allowed to invent steps. The same module
also handles the optional admin task of generating diverse paraphrases
of each KB entry to improve retrieval recall.

**Public surface:**

| Symbol                              | Kind     | Used by                            |
| ----------------------------------- | -------- | ---------------------------------- |
| `is_available()`                    | function | UI status badge                    |
| `rewrite_query(query)`              | function | UI; runs before each search        |
| `fallback_answer(query, candidates, chat_history)` | function | UI low-conf branch        |
| `generate_paraphrases(entry)`       | function | Admin "Generate paraphrases"       |
| `explain_match(query, entry, ...)` | function | Currently unused — kept for API parity |

**Walkthrough:**

#### Lazy client (groq_service.py:24-39)

```python
def _get_client():
    global _client
    if _client is not None:
        return _client
    if not GROQ_API_KEY:
        return None
    with _client_lock:
        if _client is None:
            try:
                from groq import Groq
                _client = Groq(api_key=GROQ_API_KEY)
            except Exception as exc:
                log.error("Failed to init Groq client: %s", exc)
                return None
    return _client
```

**In plain words:** make a Groq client the first time we need it. If
there's no API key, just return nothing — every caller is taught to
handle that.

#### `SYSTEM_PROMPT` (groq_service.py:42-67)

The prompt is split into two non-negotiable sections — `LANGUAGE RULE`
and `TASK`. The language rule explicitly enumerates how to detect
English vs Hinglish vs Hindi (Devanagari) and gives examples. This is
deliberately strict because the model would otherwise default to
Hinglish for almost everything (a known artefact of Indian-language
training data).

The task section pins down two allowed behaviours: rephrase the closest
candidate, **or** ask one focused clarifying question. Inventing
resolution steps is forbidden.

#### `rewrite_query` (groq_service.py:101-127)

```python
completion = client.chat.completions.create(
    model=GROQ_MODEL,
    messages=[{"role": "system", "content": REWRITE_PROMPT},
              {"role": "user", "content": query}],
    temperature=0.0, max_tokens=60,
)
rewritten = completion.choices[0].message.content.strip()
if not rewritten or len(rewritten) > 4 * len(query) + 50:
    return query
return rewritten
```

**In plain words:** ask the LLM to trim filler words from the user's
question, but if it gives back something empty or absurdly long, ignore
it and use the original.

**Technical note:** `temperature=0.0` for determinism, `max_tokens=60`
so a runaway response can't blow latency. The sanity check on length is
a guardrail against hallucinated explanations leaking back as a search
query. On any exception, return the original — the function is
**idempotent on the failure path**.

#### `fallback_answer` (groq_service.py:148-180)

```python
messages = [{"role": "system", "content": SYSTEM_PROMPT}]
if chat_history:
    messages.extend(chat_history[-6:])     # last 6 turns of context
messages.append({"role": "user", "content": user_block})
```

The user block contains the formatted candidates. Trimming chat history
to 6 turns is a simple cost cap — full history would balloon prompt size
on long conversations.

#### `_offline_fallback` (groq_service.py:183-199)

When Groq is unavailable, the bot still presents the closest KB match
(if any) and asks for clarification — this is the message users see
behind the `source="offline_fallback"` log entry.

#### `generate_paraphrases` (groq_service.py:242-287)

Asks the model for 6-8 user-style queries per KB entry, mixing English /
Hinglish / Hindi. Output is parsed by `_parse_paraphrases` which strips
numbering, bullets, and surrounding quotes per line. Stored as
JSON-encoded list in `kb_entries.paraphrases`. `KBEntry.search_text()`
appends paraphrases to the indexed text, boosting recall on verbose or
multilingual queries.

**Python concepts used:** lazy module imports (§ 7.15), threading lock
double-check (§ 7.12), exception → fallback pattern (§ 7.11),
multi-line string constants, list extension with slicing
(`chat_history[-6:]`).

**Connections:**
- Imports: `groq` (lazy), `services.kb_service.KBEntry`,
  `services.search_service.SearchResult`, `utils.config`, `utils.logger`.
- Imported by: `app.chat_ui`.

---

### 5.9 `services/kb_service.py`

**One-line summary:** Knowledge-base ingestion (XLSX/CSV → SQLite),
KB read API, and query/feedback logging.

**Why it exists.** SQLite is the canonical store. Every other module
reads KB rows from here. The same module also persists each query and
its outcome (`query_logs`) and the user's helpful/not-helpful click
(`feedback`).

**Public surface:**

| Symbol                            | Kind     | Notes                                       |
| --------------------------------- | -------- | ------------------------------------------- |
| `KBEntry`                         | dataclass | Fields + `search_text()` method            |
| `init_db()`                       | function | Idempotent CREATE TABLE                    |
| `load_kb_into_db(path, replace)`  | function | Returns row count                          |
| `save_uploaded_kb(bytes, name)`   | function | Persist upload to `data/`                  |
| `resolve_kb_path()`               | function | Find XLSX or CSV in `data/`                |
| `get_all_entries()`               | function | List of `KBEntry`                          |
| `get_entry(row_id)`               | function | Single `KBEntry` or None                   |
| `kb_count()`, `get_last_updated()`| function | UI status                                  |
| `entries_missing_paraphrases()`   | function | Admin paraphrase loop                      |
| `set_paraphrases(row_id, list)`   | function | Admin paraphrase loop                      |
| `log_query(...)`, `log_feedback(...)`, `feedback_summary()`, `recent_queries()` | function | Telemetry |

**Walkthrough:**

#### `KBEntry` dataclass (kb_service.py:27-42)

```python
@dataclass
class KBEntry:
    row_id: int
    issue_id: str
    title: str
    category: str
    description: str
    resolution_steps: str
    paraphrases: list[str] = field(default_factory=list)

    def search_text(self) -> str:
        parts = [self.title, self.category, self.description, self.resolution_steps]
        parts.extend(self.paraphrases)
        return " | ".join(p for p in parts if p)
```

**In plain words:** one row from the knowledge base, plus a method that
joins all its searchable text into one string for embedding.

**Technical note:** `field(default_factory=list)` is the canonical fix
for the "mutable default argument" trap — a plain `paraphrases: list =
[]` would share the same list across all instances. The empty-string
filter in `search_text` keeps the indexed text dense; FAISS works better
when irrelevant sentinel strings ("nan", "") are not present.

#### Schema (kb_service.py:45-77)

Four tables:

| Table         | Purpose                                                        |
| ------------- | -------------------------------------------------------------- |
| `kb_entries`  | Canonical KB rows; `paraphrases` is a JSON-encoded TEXT column |
| `meta`        | Key/value: stores `last_updated` ISO timestamp                 |
| `query_logs`  | One row per user query; ts, query, matched id, confidence, source |
| `feedback`    | Thumbs up/down, FK to `query_logs.id`                          |

#### Connection helper (kb_service.py:80-88)

```python
@contextmanager
def _conn():
    with _db_lock:
        c = sqlite3.connect(DB_PATH, isolation_level=None)
        c.row_factory = sqlite3.Row
        try:
            yield c
        finally:
            c.close()
```

**In plain words:** a small helper that hands you a database connection
and guarantees it's closed afterwards. Also makes sure two threads
aren't both writing at the same time.

**Technical note:** `isolation_level=None` puts SQLite in autocommit
mode; every statement commits as it executes. `sqlite3.Row` is the row
factory that lets you do `row["title"]` instead of `row[2]`. The
process-wide `_db_lock` is needed because `sqlite3` connections are not
safe to share across threads; Streamlit's reruns can stack connections
quickly. Performance is fine because the writes are tiny.

#### `init_db` migration (kb_service.py:91-98)

```python
c.executescript(SCHEMA)
try:
    c.execute("ALTER TABLE kb_entries ADD COLUMN paraphrases TEXT")
except sqlite3.OperationalError:
    pass  # column already exists
```

A pragmatic in-place migration: try to add the column; if SQLite
complains it already exists, swallow the error. No version table, no
Alembic.

#### `_normalize_columns` (kb_service.py:103-130)

Handles the CSV/XLSX header variability:
- lowercase + strip + replace spaces with underscores
- alias `title` → `issue_title`, `category` → `issue_category`,
  `description` → `issue_description`
- enforce that `issue_title` and `resolution_steps` exist
- fill missing columns with `""` and drop rows with empty titles

**In plain words:** make the spreadsheet's headers match what the
database wants, even if the editor used different capitalisation.

#### `load_kb_into_db` (kb_service.py:152-185)

```python
if replace:
    c.execute("DELETE FROM kb_entries")
c.executemany("INSERT INTO kb_entries (...) VALUES (?, ?, ?, ?, ?)", [...])
c.execute("INSERT INTO meta (key, value) VALUES ('last_updated', ?) "
          "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (...))
```

`executemany` is the fast path for bulk inserts. The `ON CONFLICT … DO
UPDATE` is SQLite's UPSERT — sets `last_updated` whether or not the row
already exists. Note that `replace=True` deletes existing KB rows; this
also orphans `query_logs.matched_row_id` references from previous KB
versions, but since they're informational logs (no FK) that's
intentional.

#### Logging helpers (kb_service.py:285-344)

`log_query` returns the new `query_logs.id` so the chat can wire up the
matching feedback row later. `feedback_summary` reduces the table to
`{yes, no, total}` with one aggregate query.

**Python concepts used:** `@dataclass` + `field(default_factory=...)`
(§ 7.17), `@contextmanager` (§ 7.20), thread-safe SQLite (§ 7.12),
parameterised SQL (§ 7.21), `pandas.DataFrame.rename` / `.fillna` /
`.astype`, JSON column for nested data (§ 7.22), generator expression
inside `" | ".join(...)`, dict + list comprehensions (§ 7.9).

**Connections:**
- Imports: `pandas`, `utils.config`, `utils.logger`, stdlib `sqlite3`,
  `json`.
- Imported by: `app.chat_ui`, `services.search_service`,
  `services.groq_service`.

---

### 5.10 `services/__init__.py`

**One-line summary:** Empty package marker.

**Why it exists.** Same as `app/__init__.py` — it makes `services` a
Python package so `from services import kb_service` works. The file is
**0 bytes** by design: nothing should re-export from this layer because
some service modules (`groq_service`, `search_service`) import each
other and a non-empty `__init__.py` could create circular-import
headaches.

**Walkthrough:** there is nothing to walk through.

**Connections:** every `from services.<x>` references this package
implicitly.

---

### 5.11 `utils/config.py`

**One-line summary:** Loads `.env`, exposes paths and tunables as
module-level constants, and creates required directories on import.

**Why it exists.** A single source of truth for configuration. Every
other module imports the constants it needs from here.

**Public surface:** A flat collection of constants —

| Constant                  | Default                                              | Notes                                  |
| ------------------------- | ---------------------------------------------------- | -------------------------------------- |
| `PROJECT_ROOT`            | `<this file>/../..`                                  | `pathlib.Path`                         |
| `DATA_DIR`                | `PROJECT_ROOT/data`                                  | created if missing                     |
| `VECTOR_STORE_DIR`        | `DATA_DIR/vector_store`                              | created if missing                     |
| `DB_DIR`, `DB_PATH`       | `PROJECT_ROOT/db`, `…/database.db`                   | created if missing                     |
| `KB_XLSX_PATH`, `KB_CSV_PATH` | `DATA_DIR/knowledge_base.xlsx` / `.csv`          |                                        |
| `FAISS_INDEX_PATH`, `FAISS_META_PATH` | `…/kb.index`, `…/kb_meta.pkl`                |                                        |
| `EMBEDDING_MODEL_NAME`    | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |                          |
| `GROQ_MODEL`              | `llama-3.3-70b-versatile`                            |                                        |
| `GROQ_API_KEY`, `ADMIN_PASSWORD` | (empty)                                       | secrets                                |
| `SEMANTIC_WEIGHT`         | 0.7                                                  | hybrid blend                           |
| `FUZZY_WEIGHT`            | 0.3                                                  | hybrid blend                           |
| `CONFIDENCE_THRESHOLD`    | 0.45                                                 | gate to Groq fallback                  |
| `TOP_K`                   | 3                                                    | results returned to UI                 |
| `RERANKER_ENABLED`        | true                                                 | parsed from "1/true/yes/on"            |
| `RERANKER_MODEL_NAME`     | `BAAI/bge-reranker-v2-m3`                            |                                        |
| `RERANK_TOP_K`            | 15                                                   | size of stage-1 → stage-2 pool         |
| `KB_COLUMNS`              | `["issue_id", "issue_title", "issue_category", "issue_description", "resolution_steps"]` | canonical KB columns |

**Walkthrough:**

```python
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")
```

**In plain words:** find the project root, then read the `.env` file
into the environment.

**Technical note:** `python-dotenv` only sets variables that are not
already in `os.environ`, so an exported shell variable wins over the
file. `load_dotenv` is called at *module import time*, which means the
first import of `utils.config` (anywhere in the process) is what
materialises the env.

```python
def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default
```

**In plain words:** read a number from an env var; if it's missing or
not a valid number, use the default.

**Technical note:** typical `os.getenv("X", "0.7")` followed by
`float(...)` would crash on `"abc"`. This helper is gentler — bad input
is silently ignored. There is no logging of the bad value, so config
typos are easy to miss.

```python
for d in (DATA_DIR, VECTOR_STORE_DIR, DB_DIR):
    d.mkdir(parents=True, exist_ok=True)
```

**In plain words:** make sure the data and database folders exist on
import, creating them if needed.

**Technical note:** `parents=True` creates intermediate parents,
`exist_ok=True` makes the call idempotent. This is why the project
works on a fresh checkout with no manual `mkdir`.

**Python concepts used:** `pathlib.Path` (§ 7.1), `os.getenv` defaults,
side effects at import time, `python-dotenv` (§ 7.23), boolean parsing
from string (§ 7.24).

**Connections:**
- Imports: `os`, `pathlib`, `dotenv`.
- Imported by: every other project module.

---

### 5.12 `utils/logger.py`

**One-line summary:** One-time logging configuration; returns named
loggers.

**Why it exists.** Centralises log format and level so every module logs
identically, without each one configuring `logging` itself.

**Public surface:**

| Symbol             | Kind     | Notes                                  |
| ------------------ | -------- | -------------------------------------- |
| `get_logger(name)` | function | Returns a `logging.Logger`             |

**Walkthrough:**

```python
_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_configured = False

def get_logger(name: str = "dcs_chatbot") -> logging.Logger:
    global _configured
    if not _configured:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_FORMAT))
        root = logging.getLogger()
        root.handlers = [handler]
        root.setLevel(logging.INFO)
        _configured = True
    return logging.getLogger(name)
```

**In plain words:** the very first time anyone asks for a logger,
configure the global format and level. After that, just hand out named
loggers.

**Technical note:** the `_configured` flag avoids attaching duplicate
handlers when called repeatedly (a common gotcha that produces double
log lines). `root.handlers = [handler]` *replaces* whatever Streamlit /
huggingface might have installed. Format includes `%(levelname)-7s` —
the `-7` left-pads to 7 chars so columns line up across `INFO`,
`WARNING`, `ERROR`.

**Python concepts used:** `global` (§ 7.25), `logging` standard library,
module-level state.

**Connections:**
- Imports: `logging`, `sys`.
- Imported by: every service / app module.

---

### 5.13 `utils/__init__.py`

**One-line summary:** Empty package marker.

**Why it exists.** Same reasoning as the other `__init__.py` files —
makes `utils` importable as a package. **0 bytes** by design.

**Connections:** referenced implicitly by every `from utils.<x>`
import.

---

## 6. Non-Python Files

### `requirements.txt`

```
streamlit>=1.32.0
pandas>=2.1.0
openpyxl>=3.1.2
python-dotenv>=1.0.1
sentence-transformers>=2.7.0
faiss-cpu>=1.7.4
rapidfuzz>=3.6.0
numpy>=1.26.0
groq>=0.11.0
```

| Library                  | Role                                                     |
| ------------------------ | -------------------------------------------------------- |
| `streamlit`              | UI framework (chat, sidebar, session state)              |
| `pandas`                 | Read XLSX/CSV, clean column names, drop empty rows       |
| `openpyxl`               | Excel engine pandas uses for `.xlsx`                     |
| `python-dotenv`          | Loads `.env` into `os.environ`                           |
| `sentence-transformers`  | `SentenceTransformer` (bi-encoder) + `CrossEncoder`      |
| `faiss-cpu`              | Vector index `IndexFlatIP`                               |
| `rapidfuzz`              | Fuzzy string matching                                    |
| `numpy`                  | float32 vector handling                                  |
| `groq`                   | Groq LLM HTTP client                                     |

To change a version, edit this file then re-run
`pip install -r requirements.txt`.

### `.env` (you create this; not in git)

Keys consumed by `utils/config.py`. See [§ 4 Setup and Run](#4-setup-and-run)
for a worked example.

### `.gitignore`

```
.env
__pycache__/
*.pyc
.venv/
venv/
db/*.db
data/vector_store/*
!data/vector_store/.gitkeep
.streamlit/secrets.toml
```

The interesting line is the negation `!data/vector_store/.gitkeep`. Git
won't track empty folders, so a placeholder file keeps the directory in
the repo while everything else inside it (the actual generated index)
stays ignored.

### `data/knowledge_base.csv`

Starter KB with 25 DCS issues. Required columns:

| Column              | Required                | Description                                   |
| ------------------- | ----------------------- | --------------------------------------------- |
| `issue_id`          | optional                | Stable identifier ("DCS_001", …)              |
| `issue_category`    | optional                | Free-text category ("Authentication", …)      |
| `issue_title`       | **required**            | Short problem name                            |
| `issue_description` | optional                | One- or two-sentence symptom                  |
| `resolution_steps`  | **required**            | Markdown / numbered list of steps             |

Aliases `title`, `category`, `description` are accepted on load and
auto-renamed to the `issue_*` form. See `kb_service._normalize_columns`
([§ 5.9](#59-serviceskb_servicepy)).

### `data/vector_store/.gitkeep`

Empty file. Its only job is to keep the `vector_store/` directory in the
repo so the `.gitignore` negation works.

### `db/database.db` (created at runtime)

Single SQLite file holding the four tables (`kb_entries`, `meta`,
`query_logs`, `feedback`). To wipe state, stop Streamlit and delete this
file — the next run will recreate it from `data/knowledge_base.csv`.

---

## 7. Python Concepts Reference

This section explains every non-trivial Python concept used in the
project. Each entry has a 2-line definition, a tiny standalone example,
and a pointer to where the project uses it.

### 7.1 `pathlib.Path`

Object-oriented file paths. Replaces `os.path.join` and string
concatenation; works the same on Windows and Linux.

```python
from pathlib import Path

p = Path("/tmp") / "data" / "file.txt"   # use / to join
p.parent                                 # /tmp/data
p.suffix                                 # .txt
p.exists()                               # bool
p.write_bytes(b"hello")
```

Used in: `utils/config.py` (paths everywhere), `app/main.py:12`
(`Path(__file__).resolve().parent.parent`), `services/embedding_service.py`
(index save/load), `services/kb_service.py:188-199` (KB upload).

### 7.2 `from __future__ import annotations`

Postpones evaluation of all type hints in the module — they become
strings at import time, so you can reference types that aren't defined
yet (forward references) without quotes.

```python
from __future__ import annotations

class Tree:
    def __init__(self, parent: Tree | None = None):  # works without quotes
        self.parent = parent
```

Used in: every project module (top of file).

### 7.3 Built-in generic type hints (`list[str]`, `dict[int, float]`, `tuple[int, float]`)

Python 3.9+ lets you write generic types as subscripts of the builtins,
no `from typing import List` required.

```python
def first_words(sentences: list[str]) -> list[str]:
    return [s.split()[0] for s in sentences if s]
```

Used in: `services/search_service.py:104`, `services/embedding_service.py:41`,
`services/kb_service.py:35`, `services/fuzzy_service.py:21`.

### 7.4 `typing.Optional`

`Optional[T]` is shorthand for `T | None`. Used for return values that
may or may not be present.

```python
from typing import Optional

def find(name: str) -> Optional[int]:
    table = {"a": 1, "b": 2}
    return table.get(name)         # None if missing
```

Used in: `services/embedding_service.py:11,95`,
`services/search_service.py:14`, `services/kb_service.py:11`,
`services/groq_service.py:10`.

### 7.5 Decorators

A decorator is a function (or class) that wraps another function to add
behaviour. Syntax: `@decorator` above the def.

```python
def shouting(fn):
    def wrapped(*args, **kwargs):
        return fn(*args, **kwargs).upper()
    return wrapped

@shouting
def greet(name): return f"hi {name}"

greet("ana")    # "HI ANA"
```

Used in: `@st.cache_resource` (`app/chat_ui.py:16`), `@dataclass`
(`services/search_service.py:30`, `services/kb_service.py:27`),
`@property` and `@staticmethod` (`services/search_service.py:39,150`),
`@contextmanager` (`services/kb_service.py:80`), `@classmethod`
(`services/embedding_service.py:64,90`).

### 7.6 f-strings

`f"…"` lets you embed expressions in a string with `{}`. Format specifiers
follow a colon: `:.0%` is "percent with 0 decimals".

```python
name = "ana"
score = 0.873
f"hello {name}, you scored {score:.0%}"
# 'hello ana, you scored 87%'
```

Used heavily — e.g. `app/chat_ui.py:33,45,113-122`,
`services/groq_service.py:142,159,189-194`.

### 7.7 Type hints

Annotations on function arguments and return values. They're not
enforced at runtime — they document intent and let static checkers like
mypy catch bugs.

```python
def add(x: int, y: int = 0) -> int:
    return x + y
```

Used in: every function in the project.

### 7.8 `dict.get(key, default)`

Read a value if it's there, otherwise return a default. Avoids
`KeyError`.

```python
counts = {"apple": 3}
counts.get("banana", 0)   # 0
```

Used in: `services/search_service.py:117-118` (default 0.0 fuzzy/sem
scores), `services/rerank_service.py:79-80`,
`app/chat_ui.py:52,207,210-215`.

### 7.9 List / dict / set comprehensions

Compact syntax for building a collection by iterating + filtering.

```python
nums = [1, 2, 3, 4]
squares = [n * n for n in nums]              # [1, 4, 9, 16]
big = {n: n * n for n in nums if n > 2}      # {3: 9, 4: 16}
unique_sizes = {len(w) for w in ["a","bb","cc"]}  # {1, 2}
```

Used in:
- `services/fuzzy_service.py:23-25` (dict comp)
- `services/search_service.py:104,108-109` (dict + set)
- `services/kb_service.py:103-105,170-178,233,260` (dict + list)
- `services/groq_service.py:113-114,279-287`

### 7.10 Packages and `__init__.py`

A directory becomes a Python package when it contains an `__init__.py`
file. The file can be **empty** — its presence alone is the signal.
Without it, `from foldername import x` fails (with some exceptions for
PEP 420 namespace packages, which this project does not rely on).

```
mypkg/
├── __init__.py    # empty: 0 bytes
└── tools.py       # has def helper(): ...
```

Then `from mypkg.tools import helper` works.

Used in: `app/__init__.py`, `services/__init__.py`, `utils/__init__.py`
— all 0 bytes.

### 7.11 Graceful degradation pattern

A function that *can* fail returns a sentinel (None, empty dict/list)
instead of raising, so callers can carry on without the feature.

```python
def cached_api_call() -> dict:
    try:
        return remote_call()
    except RemoteError:
        return {}     # caller treats {} as "no data"
```

Used in: `services/rerank_service.py:71-84` (`{}` on failure → search
falls back to hybrid), `services/groq_service.py:125-127` (return
original query on rewrite failure), `services/embedding_service.py:105-107`
(return `None` on corrupted index → caller rebuilds).

### 7.12 `threading.Lock` and double-checked locking

A lock prevents two threads from running the same critical section at
once. The "double check" — checking the singleton both before and after
acquiring the lock — avoids paying the lock cost on every call once the
singleton exists.

```python
import threading
_obj = None
_lock = threading.Lock()

def get():
    global _obj
    if _obj is None:                # fast path
        with _lock:
            if _obj is None:        # avoid duplicate init in race
                _obj = ExpensiveThing()
    return _obj
```

Used in: `services/embedding_service.py:24-38`,
`services/rerank_service.py:21-52`, `services/groq_service.py:20-39`,
`services/kb_service.py:24,82` (single-lock per-process write
serialisation).

### 7.13 `pickle`

Serialises Python objects to bytes; useful for persisting things that
have no native file format. Pickle is **not safe** to load from
untrusted sources (it can execute arbitrary code), but for trusted
internal artefacts like our metadata it's fine.

```python
import pickle
with open("meta.pkl", "wb") as f:
    pickle.dump({"row_ids": [1, 2, 3], "dim": 384}, f)
```

Used in: `services/embedding_service.py:86-88,103-104` (saves and loads
the row-id ↔ FAISS-position mapping alongside `kb.index`).

### 7.14 `@classmethod`

A method that receives the class itself as the first argument (`cls`)
instead of an instance. Used for alternate constructors and factory
methods.

```python
class Vector:
    def __init__(self, x, y): self.x, self.y = x, y

    @classmethod
    def from_pair(cls, pair): return cls(pair[0], pair[1])

v = Vector.from_pair((3, 4))
```

Used in: `services/embedding_service.py:64-75` (`FaissIndex.build`),
`services/embedding_service.py:90-107` (`FaissIndex.load`).

### 7.15 Lazy / deferred imports

Putting `import heavy_lib` inside a function instead of at the top of
the module. The library is only imported the first time the function
runs — so users who never use that feature don't pay the import cost.

```python
def maybe_use_pandas():
    import pandas as pd      # only loaded when this fn is called
    return pd.DataFrame(...)
```

Used in: `services/embedding_service.py:34,66,82,96` (faiss,
sentence-transformers), `services/rerank_service.py:41`,
`services/groq_service.py:33`.

### 7.16 Module-level singletons vs Streamlit cache

Two layers in this project:
1. **Module singleton**: `services/search_service.py:166-173` keeps a
   single `HybridSearcher` instance per process via a module-level
   `_singleton` variable.
2. **Streamlit cache**: `@st.cache_resource` in `app/chat_ui.py:16`
   stores the same instance in Streamlit's resource cache so repeated
   reruns of the script (which Streamlit does on every interaction)
   don't re-instantiate.

Both are needed: Streamlit's cache survives reruns *within* a session;
the module-level singleton survives across all sessions in the same
process.

### 7.17 `@dataclass` with `field(default_factory=...)`

Auto-generates `__init__`, `__repr__`, `__eq__` from typed attributes.
For mutable defaults (lists, dicts), use `field(default_factory=list)`
instead of `= []` to avoid sharing the default across instances.

```python
from dataclasses import dataclass, field

@dataclass
class Cart:
    user: str
    items: list[str] = field(default_factory=list)

a = Cart("ana"); a.items.append("apple")
b = Cart("bo")
b.items     # [] — not contaminated by a's append
```

Used in: `services/kb_service.py:27-35`, `services/search_service.py:30-37`.

### 7.18 `@property` and `@staticmethod`

`@property` exposes a method as an attribute (no parentheses needed at
call site). `@staticmethod` is a function that lives inside a class but
takes no `self`/`cls` — purely for namespacing.

```python
class Box:
    def __init__(self, w, h): self.w, self.h = w, h
    @property
    def area(self): return self.w * self.h
    @staticmethod
    def from_square(side): return Box(side, side)

b = Box(3, 4)
b.area              # 12 — note: no ()
Box.from_square(5)
```

Used in: `services/search_service.py:39-41` (`@property confidence`),
`services/search_service.py:149-153` (`@staticmethod is_low_confidence`).

### 7.19 Sorting with `key=lambda`

`list.sort(key=fn)` sorts by the value `fn(item)` rather than the items
themselves. `lambda` makes a one-line function inline.

```python
people = [("ana", 30), ("bo", 25)]
people.sort(key=lambda p: p[1])     # by age
```

Used in: `services/search_service.py:131,144` (sort by `r.hybrid` then
by `r.score`).

### 7.20 `@contextmanager`

Lets you write a context manager (something usable with `with`) as a
generator instead of a class with `__enter__` / `__exit__`. The code
before `yield` is the setup; after is the teardown.

```python
from contextlib import contextmanager

@contextmanager
def open_locked(path):
    f = open(path); print("opened")
    try: yield f
    finally: f.close(); print("closed")

with open_locked("foo.txt") as f:
    data = f.read()
```

Used in: `services/kb_service.py:80-88` (`_conn` — yields a SQLite
connection inside a thread lock).

### 7.21 Parameterised SQL

Pass query parameters as a tuple to `cursor.execute(sql, params)`. The
DB driver escapes them — the **only** safe way to incorporate
user-controlled values into SQL.

```python
import sqlite3
c = sqlite3.connect(":memory:")
c.execute("CREATE TABLE u(name)")
c.execute("INSERT INTO u VALUES (?)", (user_input,))   # safe
# NEVER: c.execute(f"INSERT INTO u VALUES ('{user_input}')")
```

Used in: every SQL call in `services/kb_service.py`.

### 7.22 JSON column for nested data

When a column holds a small list/dict, encode it as a JSON string and
store as TEXT. SQLite has JSON1 functions but the project keeps it
simple with `json.dumps` / `json.loads`.

```python
import json
paraphrases = ["q1", "q2", "q3"]
encoded = json.dumps(paraphrases, ensure_ascii=False)   # safe for Unicode
# round-trip:
back = json.loads(encoded)
```

Used in: `services/kb_service.py:209-213` (decode), `269` (encode).
`ensure_ascii=False` matters because paraphrases contain Devanagari.

### 7.23 `python-dotenv`

`load_dotenv(path)` reads a `KEY=VALUE` file and sets each variable in
`os.environ` (without overwriting already-set values).

```python
# .env
GROQ_API_KEY=sk_xxx

# code
from dotenv import load_dotenv; load_dotenv()
import os; os.getenv("GROQ_API_KEY")
```

Used in: `utils/config.py:7-10`.

### 7.24 Boolean parsing from string

Env vars are always strings, so a truthy boolean is parsed by checking
membership in a known-true set.

```python
raw = os.getenv("MY_FLAG", "true")
flag = raw.strip().lower() in ("1", "true", "yes", "on")
```

Used in: `utils/config.py:57-59` (`RERANKER_ENABLED`).

### 7.25 `global` keyword

By default, assigning to a name inside a function creates a local. Use
`global` to declare you mean the module-level variable.

```python
counter = 0
def bump():
    global counter
    counter += 1
```

Used in: `services/embedding_service.py:30`,
`services/rerank_service.py:33`, `services/groq_service.py:25`,
`services/search_service.py:171`, `utils/logger.py:14`.

---

## 8. Glossary

| Term                       | Meaning                                                                                                                       |
| -------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| **DCS**                    | Digital Crop Survey — Government of India platform for surveying crops on cultivable land. Surveyors use the Android app.     |
| **Hinglish**               | Hindi vocabulary written in Latin (English) letters. e.g. *"app nahi chal raha"*.                                             |
| **Devanagari**             | The Hindi script. e.g. *"ऐप नहीं चल रहा"*.                                                                                    |
| **Bi-encoder**             | A model that encodes query and document **separately** to vectors. Fast at retrieval (one query encode + index lookup).        |
| **Cross-encoder**          | A model that takes `(query, document)` together and outputs a relevance score. Slow but more accurate; used as a reranker.    |
| **MiniLM**                 | A small (~118M-param) sentence-transformer family. We use the multilingual paraphrase variant — same vector space across 50+ langs. |
| **BGE reranker**           | "BAAI General Embedding" reranker. Cross-encoder trained for retrieval reranking; outputs raw logits → sigmoid to (0,1).      |
| **FAISS**                  | Facebook AI Similarity Search. C++ vector index library; we use `IndexFlatIP` — exact inner-product search, no quantisation.  |
| **`IndexFlatIP`**          | A FAISS index that stores raw float32 vectors and searches by exact inner product. With L2-normalised vectors, IP = cosine.   |
| **L2-normalisation**       | Scaling a vector so its Euclidean length is 1. Lets cosine similarity be computed via inner product.                          |
| **Token-set ratio**        | A `rapidfuzz` fuzzy metric that ignores word order and duplicates. Good for free-form queries.                                |
| **Confidence threshold**   | The minimum top-result score (post-rerank if used) below which the bot escalates to Groq.                                     |
| **Hybrid score**           | `SEMANTIC_WEIGHT * cosine + FUZZY_WEIGHT * fuzzy_ratio`. Default 0.7 / 0.3.                                                   |
| **Grounded LLM call**      | An LLM invocation where the prompt includes the candidates so the model has facts to ground its answer in.                    |
| **Paraphrase**             | A user-style rephrasing of an issue. Indexed alongside the title to boost recall.                                             |
| **`@st.cache_resource`**   | Streamlit decorator: compute once per process, share across sessions. Used for the search engine + DB bootstrap.              |
| **Session state**          | `st.session_state` — a dict-like object that persists across script reruns within one user's Streamlit session.               |
| **Logit**                  | A model's pre-sigmoid score, in (-∞, +∞). `sigmoid(logit)` is in (0, 1).                                                       |

---

## 9. How to Make Changes (Cookbook)

### 9.1 Add a new KB issue

**Easiest:** open the admin panel, upload a new XLSX/CSV with the same
columns, choose **Replace** → Apply.

**By hand:**
1. Edit `data/knowledge_base.csv`. Add a row with at least `issue_title`
   and `resolution_steps`.
2. Stop Streamlit (Ctrl+C).
3. Delete `db/database.db` and `data/vector_store/*`.
4. Restart `streamlit run app/main.py`. Bootstrap will reingest and
   rebuild the index.

### 9.2 Tune retrieval weights

Edit `.env`:
```
SEMANTIC_WEIGHT=0.6
FUZZY_WEIGHT=0.4
CONFIDENCE_THRESHOLD=0.50
```
Restart Streamlit. The values are read in `utils/config.py:51-53`.

> Caveat: when the reranker is enabled, the confidence threshold is
> applied to the *rerank* score, not the hybrid score. If you change
> `RERANKER_ENABLED`, retune `CONFIDENCE_THRESHOLD`.

### 9.3 Disable the reranker

Set `RERANKER_ENABLED=false` in `.env`. Search will use stage-1 hybrid
scores only. Faster, slightly less accurate. See `utils/config.py:57-59`,
`services/search_service.py:134-145`.

### 9.4 Switch the embedding model

Edit `.env`:
```
EMBEDDING_MODEL_NAME=sentence-transformers/distiluse-base-multilingual-cased-v2
```
Then **delete the FAISS index** at `data/vector_store/*` (a different
model has a different vector space). Restart. The bootstrap will rebuild
embeddings on first request.

### 9.5 Switch the LLM

Set `GROQ_MODEL` in `.env`. Any Groq-supported model id works. Code path:
`utils/config.py:44`, used at `services/groq_service.py:112,172,231,261`.

### 9.6 Tweak the system prompt

Edit `SYSTEM_PROMPT` in `services/groq_service.py:42-67`. The language
rules and the "never invent steps" rule are load-bearing — read the
existing wording before changing.

### 9.7 Add a new KB column (e.g. `severity`)

1. Add `"severity"` to `KB_COLUMNS` in `utils/config.py:68-74`.
2. Add `severity TEXT` to `SCHEMA` in `services/kb_service.py:45-77` and
   write a one-time `ALTER TABLE` migration like the existing
   `paraphrases` one.
3. Add `severity` to the `KBEntry` dataclass in `services/kb_service.py:27-42`.
4. Add it to the SELECT lists in `_row_to_entry`, `get_all_entries`,
   `get_entry`, `entries_missing_paraphrases`.
5. Decide if it should be in `search_text()` (boost retrieval) or only
   metadata (`_result_card` rendering in `app/chat_ui.py:31-45`).

### 9.8 Add a brand-new service module

1. Create `services/<your_service>.py`.
2. Import it where needed: `from services import <your_service>`.
3. No need to touch `services/__init__.py` — empty by design.
4. Read the lazy-import / lock / `_load_failed` pattern in
   `services/rerank_service.py` if your module has a heavy dependency
   that should be optional.

### 9.9 Inspect query / feedback logs

```powershell
sqlite3 db/database.db
> SELECT ts, query, confidence, source FROM query_logs ORDER BY id DESC LIMIT 20;
> SELECT s.source, AVG(f.helpful) AS helpfulness, COUNT(*) AS n
  FROM feedback f JOIN query_logs s ON s.id = f.query_log_id
  GROUP BY s.source;
```

`recent_queries(limit)` in `services/kb_service.py:335-344` returns the
same `query_logs` rows from Python.

### 9.10 Reset the bot to a clean state

```powershell
Stop-Process -Name streamlit -ErrorAction SilentlyContinue
Remove-Item db\database.db -ErrorAction SilentlyContinue
Remove-Item data\vector_store\* -Exclude .gitkeep
streamlit run app/main.py
```

### 9.11 Run / lint / typecheck

The repo has **no test suite, no linter config, no typechecker config**
checked in. To add them yourself:

```powershell
pip install pytest ruff mypy
ruff check .
mypy --strict services utils
pytest
```

(You will hit type-hint gaps because `from __future__ import annotations`
defers all hints to strings; mypy still type-checks them, but third-party
types like `streamlit`, `groq`, `faiss` may need `# type: ignore`.)

---

## 10. Known Limitations & Future Work

These are honest gaps. Some are easy fixes; others are design choices
that would need a real conversation before changing.

| Area | Limitation | Notes |
| ---- | ---------- | ----- |
| **Tests** | No tests at all. | A small pytest suite for `kb_service._normalize_columns` (CSV header tolerance), `fuzzy_service.fuzzy_score`, and `search_service.HybridSearcher.search` would catch most regressions. |
| **Confidence threshold semantics** | When the reranker is on, `CONFIDENCE_THRESHOLD` is implicitly applied to the cross-encoder score, not the hybrid score. Documented but easy to forget. | A cleaner design would have separate `HYBRID_THRESHOLD` and `RERANK_THRESHOLD` constants. |
| **KB stale-row detection** | `ensure_loaded` only checks that the row-id list is unchanged. Editing a row's text in SQLite does **not** trigger a rebuild. | Workaround: admin "Rebuild embeddings" button. Real fix: hash `search_text()` per row and persist the hash in `kb_meta.pkl`. |
| **No request-level locking** | `HybridSearcher` has a lock for build/rebuild but `search()` itself is not synchronised. FAISS `IndexFlatIP` is read-only at search time so this is fine, but if anyone swaps in a mutable index it will break. | |
| **`recent_queries` not exposed in UI** | Available in `services/kb_service.py:335-344` but not surfaced anywhere. | Easy admin-tab addition. |
| **Groq streaming not used** | `fallback_answer` waits for the full completion before showing. With Llama 3.3 70B at default temp, typical latency is 2–5 s. | Migrate to `client.chat.completions.create(..., stream=True)` and `st.write_stream`. |
| **Single-process, file-backed** | Two Streamlit workers would race on SQLite writes and on the FAISS index file. The project assumes one process. | Fine for the use case; flagged so future scale-up is informed. |
| **Pickle for index metadata** | `kb_meta.pkl` is `pickle.load`-ed; safe because we wrote it ourselves, but a tampered pickle would execute arbitrary code. | Replace with `json` since the metadata is just `{row_ids: list[int], dim: int}`. |
| **No rate limiting on Groq calls** | Admin "Generate paraphrases" can fire 25+ Groq calls back-to-back. | Add a small `time.sleep` between calls or use Groq's batch API. |
| **Confidence bar can show 0%** | Clamping to [0, 1] makes the displayed percentage exact, but a 0% bar with a card next to it looks odd to users. | UI polish: hide the bar below some floor, or label it "low confidence" instead. |
| **`explain_match` is dead code** | Kept for callers that want a conversational form of a confident match. The UI uses the structured cards exclusively. | Either wire it up behind an admin toggle ("Conversational mode") or remove. |

---

*End of `PROJECT_DOCUMENTATION.md`.*
