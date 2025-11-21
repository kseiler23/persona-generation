from __future__ import annotations

from typing import Any, Dict, List, Optional

import litellm
import os
from dotenv import load_dotenv

load_dotenv()

# Normalize common aliases so stray un-prefixed model names still resolve.
litellm.model_alias_map = {
    "gemini-3-pro-preview": "gemini/gemini-3-pro-preview",
}

def chat_completion(
    *,
    model: str,
    messages: List[Dict[str, str]],
    max_tokens: Optional[int] = None,
    response_format: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Thin wrapper around LiteLLM's completion API.

    This lets us swap underlying providers (OpenAI, Anthropic, Azure, etc.)
    via environment variables or LiteLLM config, without touching the rest of
    the pipeline code.
    """

    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if response_format is not None:
        # Pass response_format both at the top-level (OpenAI) and via extra_body
        # to maximize compatibility across providers.
        kwargs["response_format"] = response_format
        kwargs.setdefault("extra_body", {})
        kwargs["extra_body"]["response_format"] = response_format

    # Allow explicit API key override for Gemini/Google providers.
    gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if gemini_key and model.startswith("gemini"):
        kwargs["api_key"] = gemini_key
    try:
        completion = litellm.completion(**kwargs)
    except Exception:
        raise

    choice = completion.choices[0]
    message = getattr(choice, "message", None)

    if isinstance(message, dict):
        content = message.get("content", "")
    else:
        content = getattr(message, "content", "")

    # Gemini can occasionally return an empty content string when asked for
    # JSON. If that happens, retry once without response_format to collect a
    # usable payload.
    if not content:
        retry_kwargs = dict(kwargs)
        retry_kwargs.pop("response_format", None)
        if "extra_body" in retry_kwargs:
            retry_kwargs["extra_body"].pop("response_format", None)
        completion = litellm.completion(**retry_kwargs)
        choice = completion.choices[0]
        message = getattr(choice, "message", None)
        if isinstance(message, dict):
            content = message.get("content", "")
        else:
            content = getattr(message, "content", "")

    return content or ""
