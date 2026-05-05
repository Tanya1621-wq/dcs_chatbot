# DCS Chatbot — Process Flow & Performance Guide

How the bot processes a question, where time is spent, what's saved/cached,
and what a "reranker" is.

---

## 1. The full process — what happens for ONE question

When you type a question and press Enter, here's the journey, in order:

```
You type: "My crop is missing what to do"
       │
       ▼
┌────────────────────────────────────────────────────────┐
│ STEP 1: UI catches the input                           │
│   File: app/chat_ui.py                                 │
│   Function: _render_chat()                             │
│   Time: instant (< 1 ms)                               │
└────────────────────────────────────────────────────────┘
       │
       ▼
┌────────────────────────────────────────────────────────┐
│ STEP 2: Query rewriting (LLM call to Groq)             │
│   File: services/groq_service.py                       │
│   Function: rewrite_query()                            │
│   What: turns "My crop is missing what to do"          │
│         into "crop missing"                            │
│   Time: ~300-700 ms (network call to Groq)             │
└────────────────────────────────────────────────────────┘
       │
       ▼
┌────────────────────────────────────────────────────────┐
│ STEP 3: Embed the query into numbers                   │
│   File: services/embedding_service.py                  │
│   Function: embed()                                    │
│   What: text → list of 384 numbers                     │
│   Time: ~20-50 ms (model already loaded)               │
│         or 5-30 SECONDS the very first time            │
└────────────────────────────────────────────────────────┘
       │
       ▼
┌────────────────────────────────────────────────────────┐
│ STEP 4a: FAISS semantic search                         │
│   File: services/embedding_service.py                  │
│   Function: FaissIndex.search()                        │
│   What: find top-K closest KB embeddings               │
│   Time: ~1-5 ms                                        │
└────────────────────────────────────────────────────────┘
       │
       ▼
┌────────────────────────────────────────────────────────┐
│ STEP 4b: Fuzzy match every KB row                      │
│   File: services/fuzzy_service.py                      │
│   Function: score_all()                                │
│   Time: ~5-15 ms (for 25 rows)                         │
└────────────────────────────────────────────────────────┘
       │
       ▼
┌────────────────────────────────────────────────────────┐
│ STEP 5: Combine scores (0.7 × sem + 0.3 × fuz)         │
│   File: services/search_service.py                     │
│   Function: HybridSearcher.search()                    │
│   Time: < 1 ms                                         │
└────────────────────────────────────────────────────────┘
       │
       ▼
       ┌───────────────────┴────────────────────┐
       │  Is top score ≥ confidence threshold?  │
       └────────────────────┬───────────────────┘
              YES                    NO
               │                      │
               ▼                      ▼
   ┌─────────────────────┐   ┌─────────────────────┐
   │ STEP 6a: show cards │   │ STEP 6b: call Groq  │
   │ instant render      │   │ ~500-2000 ms        │
   └─────────────────────┘   └─────────────────────┘
               │                      │
               └──────────┬───────────┘
                          ▼
              ┌─────────────────────────┐
              │ STEP 7: log to SQLite   │
              │ File: kb_service.py     │
              │ Time: < 5 ms            │
              └─────────────────────────┘
                          ▼
              ┌─────────────────────────┐
              │ STEP 8: render message  │
              │ + 👍 / 👎 buttons       │
              │ Time: < 10 ms           │
              └─────────────────────────┘
```

---

## 2. Total time per question — typical numbers

| Scenario | Total time |
|---|---|
| **First question after starting the app** | **5–30 seconds** (loading the AI model) |
| Question with confident KB match | **~400–800 ms** |
| Question that triggers Groq fallback | **~1–3 seconds** |
| Same question asked again | **~400–800 ms** (no special "memory") |

> Note: most of the per-query time is now the **Groq query rewriter** (~300-700 ms). FAISS + fuzzy themselves are blink-of-an-eye fast.

---

## 3. Why does timing vary? — every reason listed

### Reason A — Cold start (first question)

The first question of a session is **MUCH slower** because:

1. **Loading the embedding model (~470 MB)** — this takes 5-30 seconds. It happens **once**, then stays in memory until you stop the app.
2. **Loading the FAISS index from disk** — small but non-zero.
3. **Connecting to SQLite** for the first time.

After this, every subsequent question is fast.

This is handled by `@st.cache_resource` in `_bootstrap()` (`app/chat_ui.py:16`) — Streamlit keeps the loaded model in memory across re-runs.

### Reason B — Did Groq need to be called?

- **Confident match** → no Groq call → fast (~400 ms total).
- **Low confidence** → Groq fallback → adds ~500-2000 ms.

### Reason C — Query rewriting always runs

Every question now triggers `rewrite_query()`, which is one Groq call (~300-700 ms). This is a **constant overhead** we accepted to improve search quality.

> **Optional optimization:** skip the rewrite if the query is already short (e.g. < 5 words). Could shave ~400ms off short queries.

### Reason D — Network latency to Groq

Groq calls go over the internet. If your wifi is slow or Groq's servers are busy → slower. If you're offline → the rewrite/fallback gracefully skips and returns the original query.

### Reason E — How many KB rows you have

- 25 rows: fuzzy scoring takes ~10 ms.
- 1,000 rows: fuzzy takes ~200 ms.
- 100,000 rows: fuzzy becomes a bottleneck (~20 sec). At that scale you'd swap rapidfuzz for a real keyword index like BM25.

### Reason F — Streamlit re-runs the whole script on every interaction

Every click, every keystroke in chat_input → Streamlit re-runs `chat_ui.py` from the top. Anything **not cached** (with `@st.cache_resource`) recomputes. That's why we cache the heavy stuff (model, FAISS, searcher singleton).

---

## 4. "Why is asking the same question again faster?" — explained

This is a common confusion. **The bot does NOT save query results.** It re-runs the full pipeline every time.

But it FEELS faster because:

| What's already warmed up | Saves time |
|---|---|
| Embedding model loaded in RAM | huge (saves model load) |
| FAISS index in RAM | small |
| SQLite connection cached | small |
| Operating system file cache | small |
| Python imports done | small |

**Important:** the bot **never short-circuits** based on past questions. It does NOT do this:

```
if user_already_asked_this:
    return previous_answer  ← THIS IS NOT HAPPENING
```

So if you ask "crop missing" twice, it goes through embedding + FAISS + fuzzy + (maybe Groq) BOTH times. The second time is just faster because nothing is cold.

### What IS saved across questions?

Two different things:

**1. Chat history (per browser tab):**
- Stored in `st.session_state.messages`.
- Used to display the chat scroll-back AND sent to Groq for context.
- Cleared when you click "Clear chat" or refresh the page.
- NOT used to skip recomputation.

**2. Query log (permanent, in SQLite):**
- Every question is saved in `db/database.db` → table `query_logs`.
- Used for analytics ("what % of questions did we answer well?"), not for speeding up repeated queries.

---

## 5. What's saved where? — full inventory

| Data | Where | When written | Persists across restarts? |
|---|---|---|---|
| Knowledge base (KB) rows | `db/database.db` → `kb_entries` table | When you upload/load CSV | ✅ |
| LLM-generated paraphrases | `db/database.db` → `kb_entries.paraphrases` column | When you click "Generate paraphrases" | ✅ |
| FAISS embeddings index | `data/vector_store/kb.index` + `kb_meta.pkl` | After any rebuild | ✅ |
| Query logs (every question) | `db/database.db` → `query_logs` table | Every question | ✅ |
| Feedback (👍 / 👎 clicks) | `db/database.db` → `feedback` table | Every click | ✅ |
| Chat history (visible scroll-back) | `st.session_state.messages` (RAM) | Every message | ❌ (cleared on refresh) |
| Loaded embedding model | RAM (`_model` global) | First query | ❌ (re-downloaded if cache cleared) |
| Loaded searcher singleton | RAM (`_singleton` global) | First query | ❌ |
| Settings | `.env` file | When you edit it | ✅ |

---

## 6. Other scenarios that affect timing — full list

### Scenario 1 — KB just got rebuilt
After clicking "Rebuild embeddings", the next query is normal-speed because the index is already loaded into the searcher. But if you also restarted Streamlit, the next query becomes a cold start again.

### Scenario 2 — Groq is offline / API key missing
- `rewrite_query()` returns the original query immediately (no network) → **faster**.
- `fallback_answer()` returns a hand-written reply immediately → **faster**.
- But quality drops (no rewriting, no smart fallback).

### Scenario 3 — Network is slow
Both Groq calls (rewrite + fallback) wait on network. A slow connection can turn a 400ms query into a 5-second query.

### Scenario 4 — Very long user message
Embedding a 500-word message takes longer than a 5-word one (still milliseconds, but noticeable). Also, longer rewrite output = more tokens = slower Groq call.

### Scenario 5 — KB grew from 25 rows to 25,000
- FAISS still fast (designed for millions of vectors).
- Fuzzy slows linearly — 25k rows = ~250 ms.
- Index rebuild slows (embedding 25k rows takes minutes).
- Storage grows but stays small (each row's embedding = ~1.5 KB).

### Scenario 6 — Multiple users at the same time
Streamlit is single-process by default. Two users = requests queue up. The model + FAISS are shared, but each request still runs the pipeline. For high traffic, you'd run Streamlit behind multiple workers OR move the search/Groq pipeline to a separate FastAPI service.

### Scenario 7 — Disk cache cleared
If you delete `data/vector_store/*` → next startup rebuilds the FAISS index from scratch (~1 sec for 25 rows, ~minutes for 25k).
If you delete the HuggingFace model cache → next startup re-downloads ~470 MB.

### Scenario 8 — Hot-reload during development
When you edit a `.py` file and Streamlit auto-reloads, the `@st.cache_resource` may invalidate, forcing a model reload (cold start again).

---

## 7. What is a "Reranker"? — and how would it help?

### The current setup (called a "bi-encoder")

We embed the **query** and the **KB rows** *separately* into vectors, then compare them with cosine similarity:

```
Query: "crop missing"  ──→  embed  ──→  [0.12, -0.04, ...]  ┐
                                                            │ → cosine similarity
KB row: "Crop not visible" ──→ embed ──→ [0.15, -0.06, ...] ┘
```

**Pros:** super fast — KB rows are pre-embedded once, query is embedded once, comparison is instant.
**Cons:** less accurate — the model never sees query + KB row together. It has to encode both into one fixed vector and hope they end up close.

### A reranker (called a "cross-encoder")

A reranker takes the **query AND a candidate KB row TOGETHER** as one input, and outputs a single relevance score:

```
Input:  "crop missing  [SEP]  Crop not visible"  ──→  cross-encoder  ──→  0.93
Input:  "crop missing  [SEP]  Mobile device change" ──→ cross-encoder ──→  0.04
```

Because the model sees both texts together, it can reason about how they relate — much more accurately than the bi-encoder.

### The two-stage pipeline (modern RAG standard)

You don't replace the bi-encoder — you **add** the reranker as a second stage:

```
Query
  ↓
[Stage 1: bi-encoder + FAISS]  ← gets top-20 candidates fast (~5 ms)
  ↓
[Stage 2: cross-encoder rerank] ← re-scores those 20 (~50-200 ms)
  ↓
Show top-3
```

Why two stages?
- **Cross-encoders are slow** — you can't run them over millions of KB rows. So bi-encoder narrows down first.
- **Cross-encoders are accurate** — only run on 10-50 candidates → big quality boost on the final ranking.

### How it would help THIS bot

Real example you hit:
- Query: *"My crop is missing what to do"*
- Bi-encoder ranks: `Mobile device change (0.42)` first by mistake → not confident → falls back to Groq.
- Cross-encoder would re-look at all top-20 candidates with the query and likely promote `Crop type/crop not visible (0.91)` to #1 → confident match → no Groq needed → user gets the right card.

### Trade-offs

| | Bi-encoder (current) | Cross-encoder (reranker) |
|---|---|---|
| Speed | Blazing (~5 ms) | Slower (~50-200 ms per query) |
| Accuracy | Good | Excellent |
| Pre-computable? | ✅ Yes | ❌ No (needs query + candidate together) |
| Memory | Small | Medium |

### Popular reranker models

- **`BAAI/bge-reranker-v2-m3`** — multilingual, strong on Hindi/English (~570 MB)
- **`cross-encoder/ms-marco-MiniLM-L-12-v2`** — English-only, smaller (~120 MB)
- **`jinaai/jina-reranker-v2-base-multilingual`** — multilingual, fast

### When NOT to add a reranker

- KB is very small (< 50 rows): the bi-encoder + paraphrases is usually enough.
- Latency budget is tight (< 500 ms total): adding 100-200 ms might hurt UX.
- LLM-as-judge is also a reranker option — use Groq to pick the best of 5 candidates. More flexible but more expensive per query.

---

## 8. TL;DR — the mental model

1. **The bot runs the full pipeline EVERY time.** No magical query memory.
2. **The first question is always slow** (model loads ~5-30 sec).
3. **All subsequent questions are fast** because the model stays in memory.
4. **Same question = same time** (just feels faster because everything's warm).
5. **Groq calls add network time** — these are the slowest parts of a normal query.
6. **What's persisted:** KB rows, paraphrases, FAISS index, query logs, feedback. **What's NOT persisted:** chat history (per-tab), loaded model (RAM only).
7. **A reranker** = a second pass that re-scores the top candidates with a more accurate model. Slower per query but big accuracy gain. Worth adding when KB grows and bi-encoder ranking gets noisy.
