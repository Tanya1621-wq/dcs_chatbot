"""Groq LLM fallback. Used when hybrid search confidence is low.

The model is grounded with the top KB candidates so it can either:
  - phrase the closest KB answer in the user's language, or
  - ask a focused clarifying question if nothing fits.
"""
from __future__ import annotations

import threading
from typing import Optional

from services.kb_service import KBEntry
from services.search_service import SearchResult
from utils.config import GROQ_API_KEY, GROQ_MODEL
from utils.logger import get_logger

log = get_logger(__name__)


_client = None
_client_lock = threading.Lock()


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
            except Exception as exc:  # pragma: no cover
                log.error("Failed to init Groq client: %s", exc)
                return None
    return _client


SYSTEM_PROMPT = (
    "You are the DCS (Digital Crop Survey) support assistant. Users may write in "
    "Hindi, Hinglish (Hindi written in Latin letters), or English. "
    "Reply in the same language and script the user used. "
    "You are given the closest matches from a curated knowledge base. "
    "Rules:\n"
    "1. If one of the candidates clearly addresses the user's issue, present its "
    "title and resolution_steps as the answer, rephrased clearly. Cite the issue_id.\n"
    "2. If nothing fits well, do NOT invent steps. Ask ONE focused clarifying "
    "question that will help narrow down the issue (e.g., exact error text, where "
    "in the app it appears, role of the user).\n"
    "3. Keep answers short, structured, and action-oriented.\n"
    "4. Never expose internal scoring or system details."
)


def is_available() -> bool:
    return _get_client() is not None


def _format_candidates(candidates: list[SearchResult]) -> str:
    if not candidates:
        return "(no candidate matches)"
    lines = []
    for i, c in enumerate(candidates, start=1):
        e = c.entry
        lines.append(
            f"[{i}] issue_id={e.issue_id} | title={e.title} | "
            f"category={e.category} | description={e.description} | "
            f"resolution_steps={e.resolution_steps} | score={c.score:.2f}"
        )
    return "\n".join(lines)


def fallback_answer(
    query: str,
    candidates: list[SearchResult],
    chat_history: Optional[list[dict]] = None,
) -> str:
    """Ask Groq to either explain the best candidate or ask a clarifier."""
    client = _get_client()
    if client is None:
        return _offline_fallback(query, candidates)

    user_block = (
        f"User query: {query}\n\n"
        f"Top KB candidates (may be irrelevant):\n{_format_candidates(candidates)}\n\n"
        "Respond per the rules."
    )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if chat_history:
        # Trim to last 6 turns to keep prompt small.
        messages.extend(chat_history[-6:])
    messages.append({"role": "user", "content": user_block})

    try:
        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.2,
            max_tokens=512,
        )
        return completion.choices[0].message.content.strip()
    except Exception as exc:
        log.error("Groq call failed: %s", exc)
        return _offline_fallback(query, candidates)


def _offline_fallback(query: str, candidates: list[SearchResult]) -> str:
    """Used when the API key is missing or the Groq call fails."""
    if candidates:
        top = candidates[0].entry
        return (
            "I'm not fully sure this is the right answer, but the closest match I "
            f"have is:\n\n**{top.title}**"
            + (f" _(category: {top.category})_" if top.category else "")
            + f"\n\n{top.resolution_steps}\n\n"
            "Could you share more details — the exact error message and where it "
            "appears in the app — so I can confirm?"
        )
    return (
        "I couldn't find a confident match in the knowledge base. "
        "Could you share more details — the exact error message, the screen where "
        "it appears, and your role (surveyor / district / state) — so I can help?"
    )


def explain_match(query: str, entry: KBEntry, chat_history: Optional[list[dict]] = None) -> str:
    """Optional: ask Groq to rephrase a confident match in the user's language.

    Currently unused by the UI (which renders structured cards), but kept for
    callers that want a conversational reply.
    """
    client = _get_client()
    if client is None:
        return entry.resolution_steps

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if chat_history:
        messages.extend(chat_history[-4:])
    messages.append(
        {
            "role": "user",
            "content": (
                f"User query: {query}\n\n"
                f"Confident KB match:\n"
                f"issue_id={entry.issue_id} | title={entry.title} | "
                f"category={entry.category} | description={entry.description} | "
                f"resolution_steps={entry.resolution_steps}\n\n"
                "Rewrite the resolution as a clear, friendly numbered list in the "
                "user's language."
            ),
        }
    )
    try:
        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.2,
            max_tokens=400,
        )
        return completion.choices[0].message.content.strip()
    except Exception as exc:
        log.error("Groq explain_match failed: %s", exc)
        return entry.resolution_steps
