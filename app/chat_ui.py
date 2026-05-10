"""Streamlit chat UI: chat panel + admin sidebar."""
from __future__ import annotations

import streamlit as st

from services import groq_service, kb_service
from services.search_service import SearchResult, get_searcher
from utils.config import ADMIN_PASSWORD, CONFIDENCE_THRESHOLD
from utils.logger import get_logger

log = get_logger(__name__)


# --- cached singletons -----------------------------------------------------

@st.cache_resource(show_spinner="Loading retrieval engine…")
def _bootstrap():
    """Initialize DB + searcher once per process."""
    kb_service.init_db()
    if kb_service.kb_count() == 0:
        # First-run bootstrap: ingest the file under data/ if present.
        loaded = kb_service.load_kb_into_db()
        log.info("Bootstrapped KB with %d rows", loaded)
    searcher = get_searcher()
    searcher.ensure_loaded()
    return searcher


# --- helpers ---------------------------------------------------------------

def _result_card(result: SearchResult) -> None:
    e = result.entry
    st.markdown(f"### {e.title}")
    meta_bits = []
    if e.issue_id:
        meta_bits.append(f"`{e.issue_id}`")
    if e.category:
        meta_bits.append(f"_{e.category}_")
    if meta_bits:
        st.markdown(" · ".join(meta_bits))
    if e.description:
        st.markdown(f"**Issue:** {e.description}")
    st.markdown("**Resolution steps:**")
    st.markdown(e.resolution_steps)
    # st.progress(min(max(result.score, 0.0), 1.0), text=f"Confidence: {result.score:.0%}")


def _chat_history_for_groq() -> list[dict]:
    """Convert session messages → OpenAI-style chat history."""
    out: list[dict] = []
    for m in st.session_state.messages:
        if m["role"] in ("user", "assistant") and isinstance(m.get("content"), str):
            out.append({"role": m["role"], "content": m["content"]})
    return out


# --- main render -----------------------------------------------------------

def render() -> None:
    st.set_page_config(
        page_title="DCS Issue Resolution Chatbot",
        page_icon="🌾",
        layout="wide",
    )
    searcher = _bootstrap()

    _render_sidebar(searcher)
    _render_chat(searcher)


def _render_sidebar(searcher) -> None:
    with st.sidebar:
        st.title("DCS Support Bot")
        # st.caption("Hindi · Hinglish · English")

        if st.button("Clear chat"):
            st.session_state.messages = []
            st.rerun()

        st.divider()
        _render_admin(searcher)

        if st.session_state.get("is_admin"):
            st.divider()
            _render_feedback_summary()


def _render_admin(searcher) -> None:
    st.subheader("Admin: Knowledge base")
    if not ADMIN_PASSWORD:
        st.info("Set `ADMIN_PASSWORD` in `.env` to enable admin actions.")
        return

    if "is_admin" not in st.session_state:
        st.session_state.is_admin = False

    if not st.session_state.is_admin:
        pw = st.text_input("Admin password", type="password", key="admin_pw")
        if st.button("Unlock"):
            if pw and pw == ADMIN_PASSWORD:
                st.session_state.is_admin = True
                st.success("Admin unlocked")
                st.rerun()
            else:
                st.error("Wrong password")
        return

    st.success("Admin mode")

    last = kb_service.get_last_updated()
    st.markdown(
        f"**KB entries:** {kb_service.kb_count()}  \n"
        f"**Last updated:** {last or '—'}"
    )
    st.markdown(
        "**Confidence threshold:** "
        f"{CONFIDENCE_THRESHOLD:.2f}  \n"
        "Below this, the assistant asks Groq for a fallback reply."
    )
    st.markdown(
        "**Groq fallback:** "
        + ("✅ enabled" if groq_service.is_available() else "⚠️ offline (no API key)")
    )

    st.divider()

    st.markdown("**Download knowledge base**")
    if kb_service.kb_count() > 0:
        col_csv, col_xlsx = st.columns(2)
        col_csv.download_button(
            "⬇️ CSV",
            data=kb_service.export_kb_csv(),
            file_name="knowledge_base.csv",
            mime="text/csv",
            use_container_width=True,
        )
        col_xlsx.download_button(
            "⬇️ Excel",
            data=kb_service.export_kb_xlsx(),
            file_name="knowledge_base.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    else:
        st.caption("KB is empty — nothing to download.")

    st.divider()

    upload = st.file_uploader(
        "Upload knowledge base (.xlsx or .csv)",
        type=["xlsx", "csv"],
        accept_multiple_files=False,
    )
    mode = st.radio("Update mode", ["Replace", "Append"], horizontal=True)

    if upload is not None and st.button("Apply update"):
        try:
            saved_path = kb_service.save_uploaded_kb(upload.getvalue(), upload.name)
            n = kb_service.load_kb_into_db(
                path=saved_path,
                replace=(mode == "Replace"),
            )
            ok = searcher.rebuild()
            if ok:
                st.success(f"Loaded {n} rows and rebuilt the index.")
            else:
                st.warning("Rows loaded, but the index could not be built.")
            st.rerun()
        except Exception as exc:
            st.error(f"Update failed: {exc}")

    if st.button("Rebuild embeddings"):
        if searcher.rebuild():
            st.success("Index rebuilt.")
        else:
            st.error("Could not rebuild the index — is the KB empty?")

    if st.button("Generate query paraphrases"):
        pending = kb_service.entries_missing_paraphrases()
        if not pending:
            st.info("All entries already have paraphrases.")
        elif not groq_service.is_available():
            st.error("Groq API key is required for paraphrase generation.")
        else:
            progress = st.progress(0.0, text="Generating paraphrases…")
            generated = 0
            for i, entry in enumerate(pending):
                paraphrases = groq_service.generate_paraphrases(entry)
                if paraphrases:
                    kb_service.set_paraphrases(entry.row_id, paraphrases)
                    generated += 1
                progress.progress(
                    (i + 1) / len(pending),
                    text=f"{i + 1}/{len(pending)} done",
                )
            st.success(f"Generated paraphrases for {generated} of {len(pending)} entries.")
            if searcher.rebuild():
                st.success("Index rebuilt with paraphrases.")
            st.rerun()

    if st.button("Lock admin"):
        st.session_state.is_admin = False
        st.rerun()


def _render_feedback_summary() -> None:
    st.subheader("Feedback")
    s = kb_service.feedback_summary()
    if s["total"] == 0:
        st.caption("No feedback yet.")
        return
    pct = (s["yes"] / s["total"]) * 100 if s["total"] else 0
    st.markdown(
        f"👍 {s['yes']} · 👎 {s['no']} · **{pct:.0f}%** helpful "
        f"(across {s['total']} responses)"
    )


def _render_chat(searcher) -> None:
    st.title("How may I help you?")
    

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Replay history.
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg.get("kind") == "result_card":
                for r in msg["results"]:
                    _result_card(r)
                if msg.get("query_log_id"):
                    _render_feedback_buttons(msg["query_log_id"], msg["id"])
            else:
                st.markdown(msg["content"])
                if msg.get("query_log_id"):
                    _render_feedback_buttons(msg["query_log_id"], msg["id"])

    prompt = st.chat_input("Write your issue here..")
    if not prompt:
        return

    msg_id = f"u{len(st.session_state.messages)}"
    st.session_state.messages.append({"role": "user", "content": prompt, "id": msg_id})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base…"):
            search_query = groq_service.rewrite_query(prompt)
            if search_query != prompt:
                log.info("Rewrote query: %r → %r", prompt, search_query)
            results = searcher.search(search_query)
            low_conf = searcher.is_low_confidence(results)

        if not low_conf:
            top = results[0]
            query_log_id = kb_service.log_query(
                query=prompt,
                matched_row_id=top.entry.row_id,
                confidence=top.score,
                source="hybrid",
            )
            for r in results:
                _result_card(r)
            assistant_id = f"a{len(st.session_state.messages)}"
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "kind": "result_card",
                    "results": results,
                    "query_log_id": query_log_id,
                    "id": assistant_id,
                }
            )
            _render_feedback_buttons(query_log_id, assistant_id)
        else:
            with st.spinner("Thinking…"):
                reply = groq_service.fallback_answer(
                    query=prompt,
                    candidates=results,
                    chat_history=_chat_history_for_groq(),
                )
            top_id = results[0].entry.row_id if results else None
            top_score = results[0].score if results else None
            query_log_id = kb_service.log_query(
                query=prompt,
                matched_row_id=top_id,
                confidence=top_score,
                source="groq" if groq_service.is_available() else "offline_fallback",
            )
            st.markdown(reply)
            assistant_id = f"a{len(st.session_state.messages)}"
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": reply,
                    "query_log_id": query_log_id,
                    "id": assistant_id,
                }
            )
            _render_feedback_buttons(query_log_id, assistant_id)


def _render_feedback_buttons(query_log_id: int, msg_id: str) -> None:
    key_yes = f"fb_yes_{query_log_id}_{msg_id}"
    key_no = f"fb_no_{query_log_id}_{msg_id}"
    state_key = f"fb_done_{query_log_id}"

    if st.session_state.get(state_key):
        st.caption("Thanks for your feedback ✓")
        return

    col1, col2, _ = st.columns([1, 1, 6])
    if col1.button("👍 Helpful", key=key_yes):
        kb_service.log_feedback(query_log_id, True)
        st.session_state[state_key] = True
        st.rerun()
    if col2.button("👎 Not helpful", key=key_no):
        kb_service.log_feedback(query_log_id, False)
        st.session_state[state_key] = True
        st.rerun()
