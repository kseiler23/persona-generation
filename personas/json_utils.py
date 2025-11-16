from __future__ import annotations

import json
import re
from typing import Any, Dict


def _strip_code_fences(text: str) -> str:
    """
    If the model wrapped JSON in triple backticks (``` or ```json), extract the inner block.
    """
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE)
    if fence:
        return fence.group(1).strip()
    return text


def _slice_curly_braces(text: str) -> str:
    """
    Find the longest {...} span as a best-effort JSON object candidate.
    """
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


def _normalize_quotes(text: str) -> str:
    """
    Replace smart quotes with ASCII quotes to avoid JSONDecodeError.
    """
    return (
        text.replace("“", '"')
        .replace("”", '"')
        .replace("‘", "'")
        .replace("’", "'")
    )


def _strip_trailing_commas(text: str) -> str:
    """
    Remove trailing commas before closing } or ] which can break strict JSON.
    """
    text = re.sub(r",\s*([}\]])", r"\1", text)
    return text


def _balance_brackets(text: str) -> str:
    """
    If output was truncated, add missing closing } or ] to balance counts.
    This is a heuristic and may over-correct if braces appear in strings.
    """
    opens_curly = text.count("{")
    closes_curly = text.count("}")
    opens_square = text.count("[")
    closes_square = text.count("]")
    fix = ""
    if closes_curly < opens_curly:
        fix += "}" * (opens_curly - closes_curly)
    if closes_square < opens_square:
        fix += "]" * (opens_square - closes_square)
    return text + fix


def coerce_json_object(text: str) -> Dict[str, Any]:
    """
    Best-effort conversion of LLM output to a JSON object.
    - Strips code fences
    - Extracts the outermost {...}
    - Normalizes quotes
    - Removes trailing commas
    - Balances missing closing braces/brackets
    Raises ValueError if parsing ultimately fails.
    """
    candidates = []
    t = text.strip()
    candidates.append(t)

    t1 = _strip_code_fences(t)
    if t1 != t:
        candidates.append(t1)

    t2 = _slice_curly_braces(t1)
    if t2 != t1:
        candidates.append(t2)

    t3 = _normalize_quotes(t2)
    if t3 != t2:
        candidates.append(t3)

    t4 = _strip_trailing_commas(t3)
    if t4 != t3:
        candidates.append(t4)

    t5 = _balance_brackets(t4)
    if t5 != t4:
        candidates.append(t5)

    tried = set()
    for s in candidates:
        if s in tried:
            continue
        tried.add(s)
        try:
            data = json.loads(s)
            if isinstance(data, dict):
                return data
        except Exception:
            continue

    raise ValueError("Failed to parse JSON object from model output.")


