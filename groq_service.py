"""Groq-powered extraction of university admission requirements.

The extractor asks ``llama-3.3-70b-versatile`` for one strict JSON object and is
cached with Streamlit so repeated lookups do not spend extra tokens. When no
``GROQ_API_KEY`` is configured (or the request fails), the bundled
``fallback_requirements.json`` database answers instantly for known universities.
"""

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_SOURCE = "groq"
FALLBACK_SOURCE = "fallback"
REQUEST_TIMEOUT_SECONDS = 30

FALLBACK_PATH = Path(__file__).with_name("fallback_requirements.json")

REQUIREMENT_FIELDS = (
    "university",
    "country",
    "sat_min",
    "sat_recommended",
    "ielts_min_overall",
    "ielts_min_listening",
    "ielts_min_reading",
    "ielts_min_writing",
    "ielts_min_speaking",
    "annual_tuition_usd",
    "acceptance_rate",
)

SYSTEM_PROMPT = (
    "You extract undergraduate international admission requirements for one "
    "university. Reply with a single JSON object and no markdown. Use null for any "
    "value you cannot verify; never invent numbers. The object must have exactly "
    "these keys: "
    '"university" (string), "country" (string), '
    '"sat_min" (integer 400-1600), "sat_recommended" (integer 400-1600), '
    '"ielts_min_overall" (number 0-9), "ielts_min_listening" (number 0-9), '
    '"ielts_min_reading" (number 0-9), "ielts_min_writing" (number 0-9), '
    '"ielts_min_speaking" (number 0-9), '
    '"annual_tuition_usd" (integer: one year of tuition for international '
    "undergraduates, in US dollars), "
    '"acceptance_rate" (number: percent from 0 to 100).'
)


def get_api_key() -> str:
    """Read GROQ_API_KEY from Streamlit secrets, then the environment.

    Accessing ``st.secrets`` raises when no secrets file exists, so the lookup is
    guarded and always falls back to the environment.
    """
    secret = ""
    try:
        secret = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        secret = ""
    if not secret:
        secret = os.environ.get("GROQ_API_KEY", "")
    return str(secret).strip()


def _as_number(value: Any, minimum: float, maximum: float) -> Optional[float]:
    """Coerce a JSON value to a number inside a sane range, or return None."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number < minimum or number > maximum:
        return None
    return number


def _as_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = " ".join(str(value).split())
    return text[:200] or default


def _normalise(payload: Dict[str, Any], requested_name: str) -> Dict[str, Any]:
    """Validate the model's JSON and collapse it to the fields the UI expects."""
    if not isinstance(payload, dict):
        raise ValueError("Groq returned a non-object JSON value")

    acceptance = _as_number(payload.get("acceptance_rate"), 0, 100)
    if acceptance is not None:
        # The prompt asks for a percentage (e.g. 4 for 4%).
        acceptance = min(max(acceptance / 100.0, 0.0), 1.0)

    return {
        "university": _as_text(payload.get("university"), requested_name) or requested_name,
        "country": _as_text(payload.get("country")),
        "sat_min": _as_number(payload.get("sat_min"), 400, 1600),
        "sat_recommended": _as_number(payload.get("sat_recommended"), 400, 1600),
        "ielts_min_overall": _as_number(payload.get("ielts_min_overall"), 0, 9),
        "ielts_min_listening": _as_number(payload.get("ielts_min_listening"), 0, 9),
        "ielts_min_reading": _as_number(payload.get("ielts_min_reading"), 0, 9),
        "ielts_min_writing": _as_number(payload.get("ielts_min_writing"), 0, 9),
        "ielts_min_speaking": _as_number(payload.get("ielts_min_speaking"), 0, 9),
        "annual_tuition_usd": _as_number(
            payload.get("annual_tuition_usd"), 0, 250_000),
        "acceptance_rate": acceptance,
    }


@st.cache_data(ttl=60 * 60 * 24, show_spinner="Extracting requirements with Groq...")
def extract_requirements(university_name: str) -> Dict[str, Any]:
    """Ask Groq for one university's requirements as JSON (cached for a day).

    Raises on a missing key, SDK error, or malformed response so the caller can
    fall back to the local database instead of caching a broken result.
    """
    api_key = get_api_key()
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not configured")

    try:
        from groq import Groq
    except ImportError as exc:  # pragma: no cover - dependency is in requirements
        raise RuntimeError("The groq package is not installed") from exc

    client = Groq(api_key=api_key, timeout=REQUEST_TIMEOUT_SECONDS)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"University: {university_name}\n"
                    "Use the most recent admissions cycle (2025-2026). If a value "
                    "is not published by the university, use null."
                ),
            },
        ],
        temperature=0,
        max_tokens=600,
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content or ""
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError("Groq returned invalid JSON") from exc
    return _normalise(payload, university_name)


@st.cache_data(show_spinner=False)
def load_fallback_requirements() -> List[Dict[str, Any]]:
    """Load the bundled local database (cached; independent of Streamlit state)."""
    if not FALLBACK_PATH.exists():
        return []
    with FALLBACK_PATH.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict):
        entries = data.get("universities", [])
    else:
        entries = data
    return entries if isinstance(entries, list) else []


def _match_score(entry: Dict[str, Any], query: str) -> int:
    name = _as_text(entry.get("name")).lower()
    aliases = [_as_text(alias).lower() for alias in entry.get("aliases", [])]
    if not name:
        return 0
    if query == name or query in aliases:
        return 3
    if query in name or any(query in alias for alias in aliases):
        return 2
    haystack = " ".join([name] + aliases)
    if all(token in haystack for token in query.split()):
        return 1
    return 0


def find_fallback_requirements(university_name: str) -> Optional[Dict[str, Any]]:
    """Find the best local match for a name (exact, alias, substring, tokens)."""
    query = " ".join(str(university_name or "").lower().split())
    if not query:
        return None

    best: Optional[Dict[str, Any]] = None
    best_score = 0
    for entry in load_fallback_requirements():
        score = _match_score(entry, query)
        if score > best_score:
            best, best_score = entry, score
            if score == 3:
                break
    if best is None:
        return None

    record = {field: best.get(field) for field in REQUIREMENT_FIELDS}
    record["university"] = _as_text(best.get("name"), university_name)
    record["country"] = _as_text(best.get("country"))
    return record


def lookup_requirements(university_name: str) -> Tuple[
        Optional[Dict[str, Any]], str, Optional[str]]:
    """Return ``(record, source, error_type)`` for a university name.

    Groq is used when a key is configured; any failure is reported via the error
    type and the local database is used instead.
    """
    name = " ".join(str(university_name or "").split())
    if not name:
        return None, FALLBACK_SOURCE, None

    if get_api_key():
        try:
            return extract_requirements(name), GROQ_SOURCE, None
        except Exception as exc:  # network, auth, rate limit, malformed JSON
            return find_fallback_requirements(name), FALLBACK_SOURCE, type(exc).__name__

    return find_fallback_requirements(name), FALLBACK_SOURCE, None
