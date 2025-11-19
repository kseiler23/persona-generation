from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List

from .llm import chat_completion
from .models import PatientProfile
from .prompts import read_prompt, load_prompts
from .config import get_model_for_agent, get_max_tokens_for_agent
from .json_utils import coerce_json_object


PATIENT_PROFILE_SYSTEM_PROMPT = read_prompt(
    "patient_profile",
    "system_prompt",
    """
You are a clinician structuring raw case material into a patient profile that
can drive a simulated-patient agent.

Input:
- A mix of structured data (e.g., demographics, diagnoses) and unstructured notes
  describing a patient's history and current presentation.

Transform this into a concise but information-rich Patient Profile.

Rules (mirror the 'Structure Data' box in the architecture):
- Extract patient-level facts and context of the visit.
- Omit clinician teaching notes, meta-commentary, or speculation that the patient
  would not have direct access to.
- Omit or generalize identifiers (names, exact locations, specific employers, etc.).
- Prefer information that is relevant to how the patient will show up in a live
  interaction with a clinician.

Output:
- Return a single JSON object with the exact keys expected by the schema:
  label, age, gender_identity, cultural_context, primary_diagnoses,
  other_relevant_conditions, presenting_problems, history_of_present_illness,
  psychosocial_history, medical_history, risk_factors, protective_factors,
  current_functioning, goals_for_care, constraints_on_disclosure, session_context.
""",
)
_pp_data = load_prompts()
_pp_found = bool((_pp_data.get("patient_profile", {}) or {}).get("system_prompt"))
 


@dataclass
class PatientProfileBuilder:
    """
    Bottom branch of the architecture:

    Raw case (structured + unstructured) → structured Patient Profile.
    """

    model: str = get_model_for_agent("patient_profile", "gpt-4.1-mini")
    max_tokens: int = get_max_tokens_for_agent("patient_profile", 1024)

    def build_from_case(self, raw_case: str) -> PatientProfile:
        """
        Build a PatientProfile from raw case material.
        """

        if not raw_case.strip():
            raise ValueError("Raw case text is empty; cannot build patient profile.")

        
        content = chat_completion(
            model=self.model,
            messages=[
                {"role": "system", "content": PATIENT_PROFILE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "Here is the raw case description. "
                        "Return ONLY a JSON object with the requested fields.\n\n"
                        f"{raw_case}"
                    ),
                },
            ],
            response_format={"type": "json_object"},
            max_tokens=self.max_tokens,
        )
        if not content:
            raise RuntimeError("Model returned empty content while building patient profile.")

        try:
            data = json.loads(content)
        except Exception:
            data = coerce_json_object(content)
        

        # Normalize fields so they match the PatientProfile schema even if the
        # model outputs None or string blobs instead of lists.
        string_fields: List[str] = [
            "history_of_present_illness",
            "psychosocial_history",
            "medical_history",
            "current_functioning",
            "session_context",
        ]
        for field in string_fields:
            value = data.get(field, "")
            if value is None:
                data[field] = ""
            elif not isinstance(value, str):
                data[field] = str(value)

        list_fields: List[str] = [
            "primary_diagnoses",
            "other_relevant_conditions",
            "presenting_problems",
            "risk_factors",
            "protective_factors",
            "goals_for_care",
            "constraints_on_disclosure",
        ]
        for field in list_fields:
            value = data.get(field)
            if value is None:
                data[field] = []
            elif isinstance(value, str):
                lines = [
                    line.strip(" -•\t")
                    for line in value.replace("\r", "").split("\n")
                    if line.strip(" -•\t")
                ]
                data[field] = lines

        def _coerce_to_text(value: Any) -> str:
            if value is None:
                return ""
            if isinstance(value, str):
                return value.strip()
            if isinstance(value, (int, float, bool)):
                return str(value)
            if isinstance(value, list):
                parts = [_coerce_to_text(item) for item in value]
                parts = [part for part in parts if part]
                return "; ".join(parts)
            if isinstance(value, dict):
                parts = []
                for k, v in value.items():
                    text = _coerce_to_text(v)
                    if not text:
                        continue
                    key = str(k).strip()
                    parts.append(f"{key}: {text}" if key else text)
                return "; ".join(parts)
            return str(value).strip()

        def _record_extra(key: Any, value: Any, extra: Dict[str, str]) -> None:
            key_str = str(key).strip() if key is not None else ""
            text = _coerce_to_text(value)
            if not text:
                return
            if not key_str:
                key_str = "note"
            if key_str in extra:
                existing = extra[key_str]
                if text in existing:
                    return
                extra[key_str] = f"{existing}; {text}"
            else:
                extra[key_str] = text

        def _ingest_extra_container(container: Any, extra: Dict[str, str]) -> None:
            if container is None:
                return
            if isinstance(container, dict):
                for key, value in container.items():
                    _record_extra(key, value, extra)
                return
            if isinstance(container, list):
                for item in container:
                    if isinstance(item, dict):
                        for key, value in item.items():
                            _record_extra(key, value, extra)
                    else:
                        _record_extra(None, item, extra)
                return
            if isinstance(container, str):
                lines = [line.strip() for line in container.splitlines() if line.strip()]
                for line in lines:
                    if ":" in line:
                        key, value = line.split(":", 1)
                        _record_extra(key, value, extra)
                    else:
                        _record_extra(None, line, extra)
                return
            _record_extra(None, container, extra)

        allowed_fields = set(PatientProfile.model_fields.keys())
        extra_attributes: Dict[str, str] = {}
        _ingest_extra_container(data.get("extra_attributes"), extra_attributes)

        for key in list(data.keys()):
            if key in allowed_fields:
                continue
            _record_extra(key, data.pop(key), extra_attributes)

        data["extra_attributes"] = extra_attributes

        return PatientProfile.model_validate(data)
