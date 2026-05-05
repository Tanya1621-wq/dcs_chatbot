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
    "You are the DCS (Digital Crop Survey) support assistant.\n\n"
    "=== LANGUAGE RULE — STRICT, NON-NEGOTIABLE ===\n"
    "Detect the language of the user's MOST RECENT message and reply in EXACTLY "
    "that language and script. Never mix languages. Do NOT default to Hinglish.\n"
    "- English message (Latin script + English words like 'what', 'is', 'error', "
    "'how') → reply in English ONLY. No Hindi/Hinglish words.\n"
    "- Hinglish message (Latin script + Hindi words like 'kaise', 'nahi', 'kya', "
    "'aa raha') → reply in Hinglish.\n"
    "- Hindi message (Devanagari script) → reply in Hindi (Devanagari).\n"
    "Examples:\n"
    "  User: 'error 404 is coming, what to do?' → reply in English.\n"
    "  User: 'error 404 aa raha hai, kya karein?' → reply in Hinglish.\n"
    "  User: 'एरर 404 आ रहा है' → reply in Hindi.\n\n"
    "=== TASK ===\n"
    "You are given the closest matches from a curated knowledge base.\n"
    "Rules:\n"
    "1. If one of the candidates clearly addresses the user's issue, present its "
    "title and resolution_steps as the answer, rephrased clearly in the user's "
    "language. Cite the issue_id.\n"
    "2. If nothing fits well, do NOT invent steps. Ask ONE focused clarifying "
    "question (in the user's language) — e.g., exact error text, where in the "
    "app it appears, role of the user.\n"
    "3. Keep answers short, structured, and action-oriented.\n"
    "4. Never expose internal scoring or system details."
)


REWRITE_PROMPT = (
    "You rewrite user messages into focused search queries for a DCS "
    "(Digital Crop Survey) support knowledge base. Users write in Hindi, "
    "Hinglish, or English.\n\n"
    "Rules:\n"
    "1. Keep the meaning. Strip filler words ('please', 'kya kare', 'what to do', "
    "'I have', 'my', 'the', 'kaise', etc.).\n"
    "2. Preserve technical tokens verbatim: error codes, screen names, feature "
    "names (e.g. 'fallow land', '503', 'Aadhaar', 'OTP').\n"
    "3. Keep the user's language — do NOT translate Hindi/Hinglish to English.\n"
    "4. Output ONLY the rewritten query. No quotes, no explanation, no prefix."
)


PARAPHRASE_PROMPT = (
    "You generate diverse example user queries for a multilingual support "
    "chatbot. Given an issue from the DCS (Digital Crop Survey) knowledge "
    "base, generate 6 to 8 short queries that real users (farmers, "
    "surveyors, field officers) would actually type when they hit this issue.\n\n"
    "Requirements:\n"
    "- Mix languages: 2-3 English, 2-3 Hinglish (Hindi in Latin letters), "
    "1-2 Hindi (Devanagari).\n"
    "- Include both short ('crop missing') and verbose ('my crops are not "
    "showing what should I do') variants.\n"
    "- Use informal, real-world phrasing — not formal documentation.\n"
    "- Preserve key technical terms (error codes, screen names) verbatim.\n"
    "- Output ONLY the queries, one per line. No numbering, no quotes, no "
    "headers, no explanations."
)


def rewrite_query(query: str) -> str:
    """Rewrite a verbose user query into a focused search query.

    Falls back to the original query if Groq is unavailable or fails.
    """
    client = _get_client()
    if client is None or not query.strip():
        return query

    try:
        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": REWRITE_PROMPT},
                {"role": "user", "content": query},
            ],
            temperature=0.0,
            max_tokens=60,
        )
        rewritten = completion.choices[0].message.content.strip()
        # Sanity check: empty or absurdly long output → keep original.
        if not rewritten or len(rewritten) > 4 * len(query) + 50:
            return query
        return rewritten
    except Exception as exc:
        log.warning("Query rewrite failed (%s); using original.", exc)
        return query


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


def generate_paraphrases(entry: KBEntry) -> list[str]:
    """Ask Groq for diverse user-query paraphrases for one KB entry.

    Returns an empty list if Groq is unavailable or the call fails.
    """
    client = _get_client()
    if client is None:
        return []

    user_block = (
        f"Issue title: {entry.title}\n"
        f"Category: {entry.category}\n"
        f"Description: {entry.description}\n"
        f"Resolution: {entry.resolution_steps}\n\n"
        "Generate 6-8 user queries per the rules."
    )

    try:
        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": PARAPHRASE_PROMPT},
                {"role": "user", "content": user_block},
            ],
            temperature=0.7,
            max_tokens=400,
        )
        text = completion.choices[0].message.content.strip()
        return _parse_paraphrases(text)
    except Exception as exc:
        log.error("Paraphrase generation failed: %s", exc)
        return []


def _parse_paraphrases(text: str) -> list[str]:
    """Clean up the LLM's free-form output into a list of plain queries."""
    out: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        # Strip leading numbering/bullets and surrounding quotes.
        line = line.lstrip("0123456789.)- *•·\"'")
        line = line.rstrip("\"'")
        line = line.strip()
        if line:
            out.append(line)
    return out
