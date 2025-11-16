from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml


DEFAULT_PROMPTS_PATH: Path = Path(__file__).with_name("prompts.yaml")


def _prompts_path() -> Path:
    """
    Resolve prompts.yaml path from env or default beside this module.
    """
    env_path = os.environ.get("PERSONA_PROMPTS_PATH")
    return Path(env_path) if env_path else DEFAULT_PROMPTS_PATH


def load_prompts(path: Path | None = None) -> Dict[str, Any]:
    """
    Load the YAML file containing prompts. Returns {} if missing/empty.
    """
    p = path or _prompts_path()
    if not p.exists():
        return {}
    text = p.read_text(encoding="utf-8")
    if not text.strip():
        return {}
    data = yaml.safe_load(text)
    return data or {}


def read_prompt(section: str, key: str, default: str = "") -> str:
    """
    Read a single prompt entry as text with a default.
    """
    p = _prompts_path()
    data = load_prompts(p)
    sec = data.get(section, {}) or {}
    found = key in sec
    value = sec.get(key, default)
    preview = (str(value)[:60] + "...") if isinstance(value, str) and len(str(value)) > 60 else str(value)
    return str(value) if value is not None else default


def write_prompt(section: str, key: str, text: str, path: Path | None = None) -> None:
    """
    Update a single prompt entry and write the YAML back to disk.
    Creates the file/section/key if missing.
    """
    p = path or _prompts_path()
    data = load_prompts(p)
    if section not in data or data.get(section) is None:
        data[section] = {}
    data[section][key] = text
    yaml_text = yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
    p.write_text(yaml_text, encoding="utf-8")
    


def read_value(section: str, key: str, default: Any) -> Any:
    """
    Generic reader that returns any YAML value type (str/int/bool/etc).
    """
    p = _prompts_path()
    data = load_prompts(p)
    sec = data.get(section, {}) or {}
    found = key in sec
    value = sec.get(key, default)
    
    return value


def read_int(section: str, key: str, default: int) -> int:
    """
    Read an integer value with fallback/default and logging.
    """
    value = read_value(section, key, default)
    try:
        return int(value)
    except Exception:
        
        return default


