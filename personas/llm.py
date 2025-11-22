from __future__ import annotations

from typing import Any, Dict, List, Optional

import litellm
import os
from dotenv import load_dotenv

load_dotenv()

# Normalize common aliases so stray un-prefixed model names still resolve.
litellm.model_alias_map = {
    "gemini-3-pro-preview": "gemini/gemini-3-pro-preview",
    "gpt-5.1": "openai/gpt-5.1",
    "gpt-5.1-2025-11-13": "openai/gpt-5.1-2025-11-13",
}

def chat_completion(
    *,
    model: str,
    messages: List[Dict[str, str]],
    max_tokens: Optional[int] = None,
    response_format: Optional[Dict[str, Any]] = None,
    api_key: Optional[str] = None,
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

    # Allow explicit API key override for providers if env vars are set or passed explicitly.
    gemini_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    openai_key = api_key or os.environ.get("OPENAI_API_KEY")

    if gemini_key and model.startswith("gemini"):
        kwargs["api_key"] = gemini_key
        # Disable safety settings for Gemini to prevent empty responses on clinical content
        kwargs["safety_settings"] = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
    if openai_key and (model.startswith("openai") or model.startswith("gpt-")):
        kwargs["api_key"] = openai_key

    try:
        completion = litellm.completion(**kwargs)
    except Exception:
        # If it's a 400 error from Gemini about response_format, retry without it immediately
        if "response_format" in kwargs:
            kwargs.pop("response_format", None)
            if "extra_body" in kwargs:
                kwargs.pop("extra_body", None)
            completion = litellm.completion(**kwargs)
        else:
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
