# DCS Chatbot — Beginner's Guide

A complete walkthrough of this project for someone learning Python by doing.
Read top-to-bottom. Each section explains a file, then explains the Python
concepts it uses with tiny standalone examples.

---

## Table of contents

1. [What this project does](#1-what-this-project-does)
2. [How the files are organized](#2-how-the-files-are-organized)
3. [The flow of one user question](#3-the-flow-of-one-user-question)
4. [File-by-file walkthrough](#4-file-by-file-walkthrough)
   - [`requirements.txt`](#41-requirementstxt)
   - [`.env.example`](#42-envexample)
   - [`app/main.py`](#43-appmainpy)
   - [`utils/config.py`](#44-utilsconfigpy)
   - [`utils/logger.py`](#45-utilsloggerpy)
   - [`services/kb_service.py`](#46-serviceskb_servicepy)
   - [`services/embedding_service.py`](#47-servicesembedding_servicepy)
   - [`services/fuzzy_service.py`](#48-servicesfuzzy_servicepy)
   - [`services/search_service.py`](#49-servicessearch_servicepy)
   - [`services/groq_service.py`](#410-servicesgroq_servicepy)
   - [`app/chat_ui.py`](#411-appchat_uipy)
5. [Python keywords & concepts cheat sheet](#5-python-keywords--concepts-cheat-sheet)
6. [Glossary of project-specific terms](#6-glossary-of-project-specific-terms)

---

## 1. What this project does

A web-based chatbot that helps users solve issues with the **Digital Crop
Survey (DCS)** mobile app. Users type their problem in **Hindi, Hinglish, or
English**, and the bot finds matching solutions from a knowledge base (a
spreadsheet of known issues + fixes).

Built with:
- **Streamlit** → turns Python code into a web app in the browser.
- **Sentence-transformers + FAISS** → AI search that understands meaning.
- **rapidfuzz** → letter-by-letter fuzzy matching.
- **Groq LLM** → optional AI fallback when nothing in the KB matches well.
- **SQLite** → built-in Python database for storing the KB and logs.

---

## 2. How the files are organized

```
dcs_chatbot/
├── app/                  ← What the user sees (Streamlit web UI)
│   ├── main.py           ← Entry point — `streamlit run app/main.py`
│   └── chat_ui.py        ← Chat panel, sidebar, admin upload
│
├── services/             ← The "brain" — search, AI, database
│   ├── kb_service.py         ← Knowledge base + SQLite + logs
│   ├── embedding_service.py  ← AI embeddings + FAISS index
│   ├── fuzzy_service.py      ← rapidfuzz scoring
│   ├── search_service.py     ← Combines semantic + fuzzy
│   └── groq_service.py       ← Calls Groq LLM
│
├── utils/                ← Shared helpers
│   ├── config.py         ← Reads .env, sets paths & tunable values
│   └── logger.py         ← Single logger used everywhere
│
├── data/
│   ├── knowledge_base.csv    ← Your spreadsheet of issues + fixes
│   └── vector_store/         ← Saved AI index (auto-generated)
│
├── db/database.db        ← SQLite database (auto-generated)
├── .env                  ← Your secret keys (you create this from .env.example)
├── requirements.txt      ← Python libraries to install
└── README.md
```

**The pattern:** `app/` (UI) calls `services/` (logic) which uses `utils/`
(helpers). UI never talks to the database directly — it always goes through a
service. This is a common Python project layout.

---

## 3. The flow of one user question

User types: *"fallow land option nahi aa raha"*

```
1. app/chat_ui.py       → catches the chat input
2. search_service.py    → searches the KB
   ├─ embedding_service.py → semantic search (meaning)
   └─ fuzzy_service.py     → fuzzy match (letters)
3. Combine the two scores → top 3 matches
4. Is the top score ≥ 0.45?
   ├─ YES → show the KB cards directly
   └─ NO  → call groq_service.py → ask the LLM for a clarifier
5. kb_service.py        → log the query into SQLite
6. UI shows 👍 / 👎 feedback buttons
```

---

## 4. File-by-file walkthrough

---

### 4.1 `requirements.txt`

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

This is the shopping list of external libraries the project needs. You install
them with:

```powershell
pip install -r requirements.txt
```

`>=1.32.0` means *"version 1.32.0 or newer"*.

| Library | What it does |
|---|---|
| `streamlit` | Builds the web UI from Python code. |
| `pandas` | Reads CSV/Excel files into tables (DataFrames). |
| `openpyxl` | Lets pandas read `.xlsx` files. |
| `python-dotenv` | Loads `.env` file into environment variables. |
| `sentence-transformers` | The AI model that turns text into numbers. |
| `faiss-cpu` | Super-fast similarity search over those numbers. |
| `rapidfuzz` | Fuzzy string matching (letter-level). |
| `numpy` | Math on arrays of numbers (used by FAISS). |
| `groq` | Talks to Groq's cloud-hosted LLM. |

---

### 4.2 `.env.example`

```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
ADMIN_PASSWORD=change_me
SEMANTIC_WEIGHT=0.7
FUZZY_WEIGHT=0.3
CONFIDENCE_THRESHOLD=0.45
```

A template. You copy it to `.env` and put your real secret keys in it. The
real `.env` is git-ignored so secrets don't end up on GitHub.

**Concept — environment variables:** values that live *outside* your code, in
the operating system or a `.env` file. Useful for secrets (API keys) and
settings that change between machines.

---

### 4.3 `app/main.py`

```python
"""Streamlit entry point."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.chat_ui import render  # noqa: E402

render()
```

**What it does:**
- Adds the project root to Python's import path so `from services...` works.
- Imports the `render()` function from `chat_ui.py` and calls it.

**Concepts:**

- **`from __future__ import annotations`** — modern type-hint syntax. Lets you
  write `list[int]` instead of `List[int]` even on older Pythons. Always safe to
  put at the top of a file.

- **`Path(__file__)`** — `__file__` is a special variable Python sets to the
  path of the current file. `Path()` wraps it in a friendly path object.
  `.resolve().parent.parent` walks up two levels (from `app/main.py` to the
  project root).

  ```python
  from pathlib import Path
  here = Path(__file__).resolve()       # /full/path/to/main.py
  parent = here.parent                  # /full/path/to/
  grandparent = here.parent.parent      # /full/path/
  ```

- **`sys.path`** — a list of folders Python searches for `import`. Adding the
  project root means `from services.kb_service import ...` will work.

- **`# noqa: E402`** — tells linters "I know this import isn't at the top of
  the file; that's intentional, don't warn me."

---

### 4.4 `utils/config.py`

```python
import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default

DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = PROJECT_ROOT / "db" / "database.db"

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
SEMANTIC_WEIGHT = _get_float("SEMANTIC_WEIGHT", 0.7)
CONFIDENCE_THRESHOLD = _get_float("CONFIDENCE_THRESHOLD", 0.45)
```

**What it does:** central place where all settings live. Reads the `.env`
file once, then exposes constants like `DB_PATH` and `CONFIDENCE_THRESHOLD`
that other files import.

**Concepts:**

- **`os.getenv("NAME", "default")`** — reads an environment variable. If it
  isn't set, returns the default.

  ```python
  import os
  api_key = os.getenv("GROQ_API_KEY", "")  # empty string if not set
  ```

- **`load_dotenv()`** — reads a `.env` file and copies its values into
  `os.environ` so `os.getenv` can find them.

- **Path arithmetic with `/`** — `pathlib.Path` overloads the `/` operator to
  join paths. Cleaner than string concatenation:

  ```python
  from pathlib import Path
  data = Path("/home/user") / "project" / "data.csv"
  # → /home/user/project/data.csv
  ```

- **Type hints** — `name: str` and `-> float` say what types the parameter
  and return value should be. Python doesn't enforce them at runtime; they're
  for tools and humans.

- **`try / except`** — handles errors so the program doesn't crash. Here, if
  the user puts a non-number in `.env`, fall back to the default.

  ```python
  try:
      n = float("not a number")
  except ValueError:
      n = 0.0
  ```

- **Underscore prefix** (`_get_float`) — convention meaning *"this is private,
  don't import it from outside this file."*

---

### 4.5 `utils/logger.py`

```python
import logging
import sys

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

**What it does:** sets up Python's built-in `logging` so every file can write
log messages with a consistent format. Used like:

```python
log = get_logger(__name__)
log.info("Loaded %d KB rows", count)
```

**Concepts:**

- **`logging`** — Python's built-in alternative to `print()`. Better because
  you can filter by level (DEBUG / INFO / WARNING / ERROR) and add timestamps.

- **`global`** — needed if you want to *reassign* a module-level variable from
  inside a function. Without `global _configured`, the line `_configured = True`
  would create a local variable instead.

- **`%s` / `%d` in log messages** — old-style string formatting. The logger
  fills them in lazily, which is faster than building the string upfront.

---

### 4.6 `services/kb_service.py`

The biggest file. Manages **three jobs**:
1. Reading the knowledge base CSV/Excel into a SQLite database.
2. Reading rows back out as `KBEntry` objects.
3. Logging every query and feedback click.

#### Key piece: the `KBEntry` dataclass

```python
from dataclasses import dataclass

@dataclass
class KBEntry:
    row_id: int
    issue_id: str
    title: str
    category: str
    description: str
    resolution_steps: str

    def search_text(self) -> str:
        parts = [self.title, self.category, self.description]
        return " | ".join(p for p in parts if p)
```

**Concept — `@dataclass`:** a decorator that automatically writes the
`__init__`, `__repr__`, and equality methods for you. Without it you'd have
to write:

```python
class KBEntry:
    def __init__(self, row_id, issue_id, title, ...):
        self.row_id = row_id
        self.issue_id = issue_id
        ...
```

With `@dataclass`, you just declare the fields. Use it whenever you have a
class that's mostly "a bag of attributes."

#### Key piece: the SQLite schema

```python
SCHEMA = """
CREATE TABLE IF NOT EXISTS kb_entries (
    row_id              INTEGER PRIMARY KEY AUTOINCREMENT,
    issue_id            TEXT,
    title               TEXT NOT NULL,
    ...
);
"""
```

**Concept — SQL inside a triple-quoted string:** Python's `"""..."""` lets
you write multi-line strings. SQLite executes the SQL when we call
`c.executescript(SCHEMA)`.

#### Key piece: a context manager for the DB connection

```python
from contextlib import contextmanager

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

Used like:

```python
with _conn() as c:
    rows = c.execute("SELECT * FROM kb_entries").fetchall()
```

**Concept — `with` statement:** guarantees cleanup. Whether the code inside
succeeds or crashes, the `finally` block runs and closes the connection.
Same idea as opening a file with `with open(...) as f:`.

**Concept — `@contextmanager`:** lets you turn a generator function (one
that uses `yield`) into something usable with `with`. Code before `yield` is
the "setup", code after `yield` is the "teardown".

#### Key piece: pandas for reading the spreadsheet

```python
import pandas as pd

def _read_kb_file(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".xlsx":
        df = pd.read_excel(path)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    return _normalize_columns(df)
```

**Concept — pandas DataFrame:** like an Excel sheet inside Python. You can
filter rows, rename columns, fill missing values:

```python
import pandas as pd
df = pd.read_csv("file.csv")
df = df.fillna("")           # replace NaN with ""
df = df[df["title"] != ""]   # keep only rows with a title
```

#### Key piece: parameterized SQL inserts

```python
c.executemany(
    "INSERT INTO kb_entries (issue_id, title, ...) VALUES (?, ?, ...)",
    [(r["issue_id"], r["issue_title"], ...) for _, r in df.iterrows()],
)
```

**Concept — `?` placeholders:** ALWAYS use these instead of f-strings when
inserting user data into SQL. Otherwise you're vulnerable to **SQL injection**
(an attacker could write a "title" that contains SQL commands).

```python
# BAD
c.execute(f"INSERT INTO t VALUES ('{user_input}')")
# GOOD
c.execute("INSERT INTO t VALUES (?)", (user_input,))
```

**Concept — list comprehension:** `[(r["x"], r["y"]) for _, r in df.iterrows()]`
builds a list in one line. Equivalent to:

```python
out = []
for _, r in df.iterrows():
    out.append((r["x"], r["y"]))
```

The `_` is a convention meaning *"I don't care about this value."*

---

### 4.7 `services/embedding_service.py`

Wraps two AI/ML libraries:
1. **sentence-transformers** — turns text into a list of ~384 numbers.
2. **FAISS** — finds the closest matches among millions of those lists, fast.

#### Key piece: lazy model loading

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

**Concept — singleton + lazy load:** the model is huge (~470 MB) and slow to
load. We load it once on first use, cache it in `_model`, and reuse it forever
after.

**Concept — `threading.Lock()`:** prevents two threads from both trying to
load the model at the same time. The "double-check" pattern (`if _model is
None` checked twice) is standard.

**Concept — local imports:** `from sentence_transformers import ...` is
inside the function instead of at the top. This delays the slow import until
it's actually needed.

#### Key piece: embeddings + FAISS

```python
def embed(texts: list[str]) -> np.ndarray:
    model = _load_model()
    vecs = model.encode(texts, normalize_embeddings=True)
    return vecs.astype("float32")

class FaissIndex:
    @classmethod
    def build(cls, row_ids, texts):
        import faiss
        vecs = embed(texts)
        index = faiss.IndexFlatIP(vecs.shape[1])
        index.add(vecs)
        return cls(index=index, row_ids=row_ids, dim=vecs.shape[1])
```

**Concept — embeddings:** an *embedding* is a fixed-length list of numbers
that captures the meaning of a piece of text. Texts with similar meanings
have similar embeddings (close together in space).

```
"app not working"        → [0.12, -0.04, 0.88, ...]  (384 numbers)
"app nahi chal raha"     → [0.11, -0.05, 0.86, ...]  ← very similar!
"how do I plant rice"    → [0.91,  0.42, 0.03, ...]  ← very different
```

**Concept — cosine similarity:** the standard way to measure how close two
embeddings are. By normalizing vectors first (`normalize_embeddings=True`),
we can use the faster *inner product* and get the same answer.

**Concept — `@classmethod`:** a method that receives the class itself as
its first argument (`cls`) instead of an instance (`self`). Used here to make
`FaissIndex.build(...)` a constructor-like factory.

```python
class Foo:
    @classmethod
    def from_string(cls, s):
        return cls(value=int(s))
foo = Foo.from_string("42")
```

#### Key piece: pickle for saving Python objects

```python
import pickle
with open(meta_path, "wb") as f:
    pickle.dump({"row_ids": self.row_ids, "dim": self.dim}, f)
```

**Concept — `pickle`:** serializes any Python object to bytes so you can
save it to disk and load it back later. `"wb"` = "write binary".

> ⚠️ Never `pickle.load()` data from an untrusted source — it can run arbitrary code.

---

### 4.8 `services/fuzzy_service.py`

Tiny file, two functions:

```python
from rapidfuzz import fuzz

def fuzzy_score(query: str, title: str, category: str) -> float:
    title_score = fuzz.token_set_ratio(query, title or "") / 100.0
    cat_score = fuzz.token_set_ratio(query, category or "") / 100.0
    return 0.8 * title_score + 0.2 * cat_score

def score_all(query: str, entries) -> dict[int, float]:
    return {e.row_id: fuzzy_score(query, e.title, e.category) for e in entries}
```

**Concept — fuzzy matching:** measures how similar two strings are
*letter-by-letter*. Useful for typos and exact tokens like error codes.

```python
from rapidfuzz import fuzz
fuzz.token_set_ratio("error 503 aa raha hai", "503 server error")  # → 81
```

**Concept — `or` as a default:** `title or ""` returns `title` if truthy,
otherwise `""`. Handy for replacing `None` with a safe default.

```python
name = None
greeting = "Hello, " + (name or "stranger")  # "Hello, stranger"
```

**Concept — dict comprehension:** like list comprehension but builds a dict.

```python
squares = {n: n*n for n in range(5)}  # {0:0, 1:1, 2:4, 3:9, 4:16}
```

---

### 4.9 `services/search_service.py`

The conductor — it asks the embedding service AND the fuzzy service, then
blends their scores.

```python
@dataclass
class SearchResult:
    entry: kb_service.KBEntry
    semantic: float
    fuzzy: float
    score: float

class HybridSearcher:
    def search(self, query, top_k=TOP_K) -> list[SearchResult]:
        query_norm = _preprocess(query)

        sem_pairs = self._index.search(query_norm, top_k=max(top_k * 4, 10))
        sem_scores = {row_id: s for row_id, s in sem_pairs}

        fuz_scores = fuzzy_service.score_all(query_norm, self._entries)

        candidate_ids = set(sem_scores) | {rid for rid, s in fuz_scores.items() if s >= 0.5}
        results = []
        for rid in candidate_ids:
            entry = self._by_id.get(rid)
            sem = sem_scores.get(rid, 0.0)
            fuz = fuz_scores.get(rid, 0.0)
            score = SEMANTIC_WEIGHT * sem + FUZZY_WEIGHT * fuz
            results.append(SearchResult(entry, sem, fuz, score))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]
```

**Concepts:**

- **Set union with `|`:** combines two sets, no duplicates.
  ```python
  {1, 2, 3} | {3, 4, 5}  # → {1, 2, 3, 4, 5}
  ```

- **`dict.get(key, default)`:** returns `default` if the key is missing
  (instead of raising `KeyError`).
  ```python
  d = {"a": 1}
  d.get("b", 0)  # → 0
  d["b"]         # → KeyError!
  ```

- **`sorted` / `.sort` with `key=lambda`:** sort by a computed value.
  `lambda r: r.score` is a tiny anonymous function — same as
  `def f(r): return r.score`. `reverse=True` sorts highest first.

- **List slicing `results[:top_k]`:** take the first `top_k` items.
  ```python
  [10, 20, 30, 40][:2]  # → [10, 20]
  ```

- **The hybrid blend** itself:
  ```python
  score = 0.7 * semantic + 0.3 * fuzzy
  ```
  Semantic search wins ties, fuzzy adds a boost when literal tokens overlap.

---

### 4.10 `services/groq_service.py`

Calls Groq's cloud LLM as a fallback when search confidence is low.

```python
SYSTEM_PROMPT = (
    "You are the DCS support assistant. Users may write in Hindi, Hinglish, or English. "
    "Reply in the same language and script the user used. "
    "You are given the closest matches from a curated knowledge base. "
    "..."
)

def fallback_answer(query, candidates, chat_history=None) -> str:
    client = _get_client()
    if client is None:
        return _offline_fallback(query, candidates)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if chat_history:
        messages.extend(chat_history[-6:])
    messages.append({"role": "user", "content": user_block})

    completion = client.chat.completions.create(
        model=GROQ_MODEL, messages=messages, temperature=0.2, max_tokens=512,
    )
    return completion.choices[0].message.content.strip()
```

**Concepts:**

- **System / user / assistant messages:** standard LLM chat format. The
  *system* message tells the model how to behave; *user* messages are what
  the human said; *assistant* messages are previous bot replies.

- **`temperature`:** 0.0 = deterministic, 1.0 = creative. We use 0.2 because
  we want consistent support answers.

- **`max_tokens=512`:** caps the reply length (~400 words).

- **Negative list slicing** — `chat_history[-6:]` = "the last 6 items".
  ```python
  [1, 2, 3, 4, 5][-2:]  # → [4, 5]
  ```

- **Offline fallback:** if the API key is missing or Groq is down, return a
  hand-written reply instead of crashing. **Always have a graceful fallback
  for external services.**

---

### 4.11 `app/chat_ui.py`

The Streamlit UI. Three big sections: bootstrap, sidebar, chat.

#### Streamlit basics

Streamlit re-runs your whole script every time the user clicks anything. To
avoid re-loading the AI model on every click, we use:

```python
@st.cache_resource(show_spinner="Loading retrieval engine…")
def _bootstrap():
    kb_service.init_db()
    searcher = get_searcher()
    searcher.ensure_loaded()
    return searcher
```

**Concept — `@st.cache_resource`:** Streamlit caches the return value across
re-runs. Use it for "expensive things you only want to do once" (DB
connections, ML models).

#### Session state — remembering things across re-runs

```python
if "messages" not in st.session_state:
    st.session_state.messages = []
st.session_state.messages.append({"role": "user", "content": prompt})
```

**Concept — `st.session_state`:** a dict-like object that persists across
Streamlit's re-runs (within one user's browser tab). This is how chat
history survives clicks.

#### Common Streamlit widgets used here

| Widget | Purpose |
|---|---|
| `st.title("...")` | Big page title |
| `st.markdown("**bold**")` | Render markdown text |
| `st.chat_input("placeholder")` | The chat box at the bottom |
| `st.chat_message("user")` | A speech bubble (use as `with` block) |
| `st.button("Click me")` | Returns `True` the run after it's clicked |
| `st.text_input(label, type="password")` | Password field |
| `st.file_uploader(...)` | File upload widget |
| `st.radio(label, options)` | Radio button group |
| `st.spinner("Loading...")` | A loading spinner (use as `with` block) |
| `st.progress(value, text=...)` | Progress bar (used for confidence) |
| `st.success(...)`, `st.error(...)`, `st.warning(...)` | Colored boxes |
| `st.rerun()` | Force-restart the script (after writing state) |

#### The main rendering pattern

```python
with st.chat_message("assistant"):
    with st.spinner("Searching knowledge base…"):
        results = searcher.search(prompt)
        low_conf = searcher.is_low_confidence(results)

    if not low_conf:
        for r in results:
            _result_card(r)
    else:
        reply = groq_service.fallback_answer(query=prompt, candidates=results)
        st.markdown(reply)
```

The `with` blocks wrap UI elements so anything inside is rendered "in" them.

---

## 5. Python keywords & concepts cheat sheet

Quick reference for the patterns used throughout this project. Each entry is
a tiny standalone example.

### Imports

```python
import os                              # whole module
from pathlib import Path               # one name from a module
from services import kb_service        # submodule
from services.kb_service import KBEntry  # specific class
import pandas as pd                    # alias
```

### Functions and type hints

```python
def add(a: int, b: int = 0) -> int:
    return a + b
add(5)        # 5
add(5, 3)     # 8
add(b=3, a=5) # 8 — keyword args
```

### `if __name__ == "__main__"`

```python
def main(): ...
if __name__ == "__main__":
    main()
```
Runs `main()` only when this file is executed directly, not when imported.
(Not used in this project because Streamlit launches `main.py` differently.)

### Classes

```python
class Dog:
    def __init__(self, name: str):
        self.name = name
    def bark(self):
        return f"{self.name} says woof"

d = Dog("Rex")
print(d.bark())  # Rex says woof
```

### `@dataclass`

```python
from dataclasses import dataclass

@dataclass
class Point:
    x: float
    y: float

p = Point(1.0, 2.0)
print(p)        # Point(x=1.0, y=2.0)
print(p == Point(1.0, 2.0))  # True
```

### `@classmethod` vs `@staticmethod` vs regular method

```python
class Counter:
    total = 0

    def __init__(self, n): self.n = n
    def value(self):       return self.n            # uses self
    @classmethod
    def from_string(cls, s): return cls(int(s))      # uses cls
    @staticmethod
    def add(a, b):           return a + b           # uses neither
```

### Decorators

A decorator wraps a function. `@my_decorator` is shorthand for
`f = my_decorator(f)`.

```python
def shout(func):
    def wrapper(*args):
        result = func(*args)
        return result.upper()
    return wrapper

@shout
def greet(name): return f"hello {name}"

greet("ana")  # "HELLO ANA"
```

You see decorators in this project:
- `@dataclass`
- `@classmethod`
- `@contextmanager`
- `@st.cache_resource`

### Context managers (`with`)

```python
with open("file.txt") as f:
    data = f.read()
# file is auto-closed, even if an error occurred
```

### List / dict / set comprehensions

```python
[x*2 for x in range(5)]          # [0, 2, 4, 6, 8]
{x: x*2 for x in range(3)}       # {0: 0, 1: 2, 2: 4}
{x % 3 for x in range(10)}       # {0, 1, 2}

# With a filter
[x for x in range(10) if x % 2]  # [1, 3, 5, 7, 9]
```

### Generator expressions

Same syntax with `(...)` instead of `[...]` — produces values lazily.

```python
total = sum(x*x for x in range(1000000))  # no big list created
```

### `*args` and `**kwargs`

```python
def f(*args, **kwargs):
    print(args, kwargs)
f(1, 2, 3, name="ana")  # (1, 2, 3) {'name': 'ana'}
```

### f-strings (formatted string literals)

```python
name, age = "Ana", 25
f"Hello {name}, you are {age}"          # "Hello Ana, you are 25"
f"Pi is {3.14159:.2f}"                  # "Pi is 3.14"
f"{42:>5}"                              # "   42" (right-aligned in 5 chars)
```

### Exception handling

```python
try:
    x = 1 / 0
except ZeroDivisionError as e:
    print(f"oops: {e}")
except (ValueError, TypeError):
    print("bad input")
else:
    print("no error")
finally:
    print("always runs")
```

### `None` and truthiness

```python
x = None
if x is None: ...      # check for None — use 'is', not '=='
if not x: ...          # True for None, "", 0, [], {}
```

### Mutable default argument trap

```python
# BAD — the list is shared across calls!
def append_to(item, lst=[]):
    lst.append(item)
    return lst

append_to(1)  # [1]
append_to(2)  # [1, 2]  ← surprise!

# GOOD
def append_to(item, lst=None):
    if lst is None: lst = []
    lst.append(item)
    return lst
```

### `global` keyword

```python
counter = 0
def bump():
    global counter      # without this, `counter = ...` would create a local
    counter += 1
```

### Slicing

```python
s = [10, 20, 30, 40, 50]
s[0]      # 10
s[-1]     # 50 (last)
s[1:3]    # [20, 30]
s[:2]     # [10, 20]
s[-2:]    # [40, 50] (last two)
s[::-1]   # [50, 40, 30, 20, 10] (reversed)
```

### `enumerate` and `zip`

```python
for i, name in enumerate(["ana", "bob"]):
    print(i, name)        # 0 ana / 1 bob

for a, b in zip([1, 2, 3], ["x", "y", "z"]):
    print(a, b)           # 1 x / 2 y / 3 z
```

### Lambda

```python
square = lambda x: x * x
square(5)  # 25

# Most common use: as a key= argument
sorted([{"n": 3}, {"n": 1}], key=lambda d: d["n"])
```

### Threading lock (used in this project for the model singleton)

```python
import threading
lock = threading.Lock()
with lock:
    # only one thread at a time can be in here
    ...
```

---

## 6. Glossary of project-specific terms

| Term | Meaning |
|---|---|
| **KB** | Knowledge Base — the spreadsheet of issues + fixes. |
| **Embedding** | A list of ~384 numbers representing the meaning of a text. |
| **Semantic search** | Finding texts by *meaning* (using embeddings). |
| **Fuzzy matching** | Finding texts by *letters* (using rapidfuzz). |
| **FAISS** | Facebook's library for fast similarity search over embeddings. |
| **Cosine similarity** | A score from -1 to 1 measuring how close two vectors are. |
| **LLM** | Large Language Model (like GPT). Here: Groq's hosted Llama model. |
| **Confidence threshold** | If top match score < this (default 0.45), call Groq instead. |
| **Hybrid retrieval** | Mixing semantic + fuzzy scores: `0.7 * sem + 0.3 * fuz`. |
| **Streamlit** | Python framework for building data web apps. |
| **Session state** | Streamlit's per-tab memory (`st.session_state`). |
| **Cache resource** | Streamlit's "load this once" decorator (`@st.cache_resource`). |
| **SQLite** | Tiny file-based database, built into Python. |
| **Dataclass** | A class made just to hold data, written with `@dataclass`. |
| **Context manager** | An object you use with `with` — guarantees cleanup. |

---

## What to try next

1. **Read** `app/chat_ui.py` end-to-end — it's the file you'll edit most.
2. **Add a row** to `data/knowledge_base.csv`, restart Streamlit, ask the bot
   about it.
3. **Tweak** `CONFIDENCE_THRESHOLD` in `.env` and watch when Groq kicks in.
4. **Change a label** in the sidebar (`_render_sidebar` in `chat_ui.py`) so
   you see the edit-test loop.

When you're ready to make a real change, tell me what you want to do and
I'll walk you through which file to edit.
