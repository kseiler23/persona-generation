from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml


DEFAULT_CONFIG_PATH: Path = Path(__file__).with_name("config.yaml")
_CONFIG_CACHE: Dict[str, Any] | None = None


def _config_path() -> Path:
    """
    Resolve config.yaml path from env or default beside this module.
    """
    env_path = os.environ.get("PERSONA_CONFIG_PATH")
    return Path(env_path) if env_path else DEFAULT_CONFIG_PATH


def load_config(path: Path | None = None) -> Dict[str, Any]:
    """
    Load the YAML config. Returns {} if missing/empty.
    """
    p = path or _config_path()
    if not p.exists():
        return {}
    text = p.read_text(encoding="utf-8")
    if not text.strip():
        return {}
    data = yaml.safe_load(text)
    return data or {}


def get_config() -> Dict[str, Any]:
    """
    Cached access to config to avoid repeated disk reads.
    """
    global _CONFIG_CACHE
    if _CONFIG_CACHE is None:
        try:
            _CONFIG_CACHE = load_config()
        except Exception:
            _CONFIG_CACHE = {}
    return _CONFIG_CACHE


def get_value(section: str, key: str, default: Any) -> Any:
    """
    Generic getter for a top-level config section.
    """
    cfg = get_config()
    sec = cfg.get(section, {}) or {}
    value = sec.get(key, default)
    return value if value is not None else default


def get_agent_config(agent_name: str) -> Dict[str, Any]:
    """
    Return the config dict for a named agent under agents.* with {} fallback.
    """
    cfg = get_config()
    agents = cfg.get("agents", {}) or {}
    sec = agents.get(agent_name, {}) or {}
    return sec


def get_model_for_agent(agent_name: str, default_model: str) -> str:
    """
    Resolve the model for a given agent with fallbacks:
    agents[agent_name].model -> defaults.model -> provided default_model
    """
    agent = get_agent_config(agent_name)
    chosen = default_model
    source = "code_default"
    if isinstance(agent.get("model"), str) and agent["model"].strip():
        chosen = str(agent["model"]).strip()
        source = f"agents.{agent_name}.model"
    else:
        defaults = get_value("defaults", "model", None)
        if isinstance(defaults, str) and defaults.strip():
            chosen = defaults.strip()
            source = "defaults.model"
    return chosen


def get_max_tokens_for_agent(agent_name: str, default_max_tokens: int) -> int:
    """
    Resolve max_tokens for a given agent with a safe int fallback.
    """
    agent = get_agent_config(agent_name)
    value = agent.get("max_tokens", default_max_tokens)
    try:
        resolved = int(value) if value is not None else default_max_tokens
    except Exception:
        resolved = default_max_tokens
    return resolved


