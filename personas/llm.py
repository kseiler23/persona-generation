from __future__ import annotations

from typing import Any, Dict, List, Optional

import litellm
from dotenv import load_dotenv

# Load environment variables from a local .env file if present.
# This is where OPENAI_API_KEY (and other provider keys) should live.
load_dotenv()


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
        # LiteLLM exposes provider-specific extras via extra_body
        kwargs.setdefault("extra_body", {})
        kwargs["extra_body"]["response_format"] = response_format

    completion = litellm.completion(**kwargs)

    # LiteLLM mirrors OpenAI's response format
    choice = completion.choices[0]
    message = getattr(choice, "message", None)

    if isinstance(message, dict):
        content = message.get("content", "")
    else:
        # pydantic object with .content attribute
        content = getattr(message, "content", "")

    return content or ""


