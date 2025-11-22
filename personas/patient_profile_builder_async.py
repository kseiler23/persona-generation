"""Async version of PatientProfileBuilder for concurrent processing."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List

from .llm_async import achat_completion
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

PATIENT_PROFILE_REVIEW_PROMPT = read_prompt(
    "patient_profile",
    "review_prompt",
    """
You are auditing a Patient Profile to catch missing or under-specified details.

You will be given:
- The raw clinical case text.
- The current Patient Profile as JSON.

Identify concrete facts from the raw case that are absent or too thin in the current profile. Think especially about:
- Presenting problems, diagnoses, comorbidities, medications, substance use.
- Course/timeline details (onset, duration, recent changes).
- Psychosocial context (housing, work/school, legal issues, relationships, finances).
- Risks and safety factors (self-harm, violence, medical red flags).
- Barriers/constraints (access, disclosure hesitations, adherence issues).
- Functional impact (sleep, appetite, energy, cognition, pain, mobility).
- Protective factors and strengths.

Output:
- Return ONLY the additions/clarifications as a JSON object with Patient Profile keys.
- Do NOT delete or rewrite existing content; only add missing details.
- For lists, include only new items not already covered.
- For strings, provide concise additions (not full rewrites). Leave out keys that do not need changes.
- Put any details that do not fit a named field into `extra_attributes` as { key: value } entries with short snake_case keys.
- Return {} if there is nothing to add.
""",
)


@dataclass
class AsyncPatientProfileBuilder:
    """
    Async version of PatientProfileBuilder for concurrent processing.
    """

    model: str = get_model_for_agent("patient_profile", "gemini/gemini-3-pro-preview")
    max_tokens: int = get_max_tokens_for_agent("patient_profile", 1024)
    review_passes: int = 1

    async def build_from_case(self, raw_case: str) -> PatientProfile:
        """
        Build a PatientProfile from raw case material.
        """

        if not raw_case.strip():
            raise ValueError("Raw case text is empty; cannot build patient profile.")

        string_fields: List[str] = [
            "history_of_present_illness",
            "psychosocial_history",
            "medical_history",
            "current_functioning",
            "session_context",
            "label",
            "gender_identity",
            "cultural_context",
        ]

        list_fields: List[str] = [
            "primary_diagnoses",
            "other_relevant_conditions",
            "presenting_problems",
            "risk_factors",
            "protective_factors",
            "goals_for_care",
            "constraints_on_disclosure",
            "medications",
            "substance_use",
        ]

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

        def _normalize_profile_data(raw: Dict[str, Any]) -> Dict[str, Any]:
            """
            Normalize raw model output into a dict ready for PatientProfile validation.
            """
            data = dict(raw) if raw is not None else {}

            for field in string_fields:
                value = data.get(field, "")
                if value is None:
                    data[field] = ""
                elif not isinstance(value, str):
                    data[field] = str(value)

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

            allowed_fields = set(PatientProfile.model_fields.keys())
            extra_attributes: Dict[str, str] = {}
            _ingest_extra_container(data.get("extra_attributes"), extra_attributes)

            for key in list(data.keys()):
                if key in allowed_fields:
                    continue
                _record_extra(key, data.pop(key), extra_attributes)

            data["extra_attributes"] = extra_attributes
            return data

        content = await achat_completion(
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

        normalized = _normalize_profile_data(data)
        profile = PatientProfile.model_validate(normalized)

        async def _request_review(current_profile: PatientProfile) -> Dict[str, Any]:
            """
            Ask the model to spot missing details and return a partial payload.
            """
            current_json = current_profile.model_dump_json(indent=2)
            content = await achat_completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": PATIENT_PROFILE_REVIEW_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            "Raw case text:\n"
                            f"{raw_case}\n\n"
                            "Current Patient Profile (JSON):\n"
                            f"{current_json}\n\n"
                            "Return ONLY a JSON object with additions/clarifications. "
                            "Leave out keys that do not need changes."
                        ),
                    },
                ],
                response_format={"type": "json_object"},
                max_tokens=self.max_tokens,
            )
            if not content:
                return {}
            try:
                review_data = json.loads(content)
            except Exception:
                try:
                    review_data = coerce_json_object(content)
                except Exception:
                    return {}
            return review_data if isinstance(review_data, dict) else {}

        def _merge_profile(base: PatientProfile, delta: Dict[str, Any]) -> PatientProfile:
            """
            Merge partial additions into the base profile without overwriting existing details.
            """
            if not delta:
                return base

            merged: Dict[str, Any] = base.model_dump()
            merged_extras: Dict[str, str] = dict(merged.get("extra_attributes", {}) or {})

            allowed_fields = set(PatientProfile.model_fields.keys())

            delta_norm = _normalize_profile_data(delta)
            for key, value in list(delta_norm.items()):
                if key == "extra_attributes":
                    _ingest_extra_container(value, merged_extras)
                    continue
                if key not in allowed_fields:
                    _record_extra(key, value, merged_extras)
                    continue
                if value is None:
                    continue

                if key in list_fields:
                    existing_list = list(merged.get(key, []) or [])
                    candidates = value if isinstance(value, list) else [value]
                    for item in candidates:
                        text = item if isinstance(item, str) else str(item)
                        text = text.strip() if isinstance(text, str) else str(text)
                        if text and text not in existing_list:
                            existing_list.append(text)
                    merged[key] = existing_list
                    continue

                # Numeric fields (age) should only be set if empty.
                if isinstance(value, (int, float)) and merged.get(key) in (None, "", []):
                    merged[key] = value
                    continue

                text_val = ""
                if isinstance(value, str):
                    text_val = value.strip()
                else:
                    text_val = str(value).strip()
                if not text_val:
                    continue

                existing_text = merged.get(key)
                existing_norm = ""
                if isinstance(existing_text, str):
                    existing_norm = existing_text.strip()
                elif existing_text is not None:
                    existing_norm = str(existing_text).strip()
                if existing_norm:
                    if text_val not in existing_norm:
                        merged[key] = f"{existing_norm}; {text_val}"
                else:
                    merged[key] = text_val

            merged["extra_attributes"] = merged_extras
            return PatientProfile.model_validate(merged)

        passes = max(0, int(self.review_passes)) if self.review_passes is not None else 0
        current = profile
        for _ in range(passes):
            additions = await _request_review(current)
            if not additions:
                break
            updated = _merge_profile(current, additions)
            # If nothing changed, stop looping
            if updated == current:
                break
            current = updated

        return current
