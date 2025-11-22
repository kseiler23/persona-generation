from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List

from .llm import chat_completion
from .models import BehavioralLinguisticProfile
from .prompts import read_prompt
from .config import get_model_for_agent, get_max_tokens_for_agent
from .json_utils import coerce_json_object


BLP_EXTRACTION_SYSTEM_PROMPT = read_prompt(
    "blp_extraction",
    "system_prompt",
    """
You are a clinical psychologist and conversation analyst.

Your task is to read an anonymized clinical interview transcript and produce a
behavioral and linguistic profile (BLP) of the patient.

The goal is to support a simulated-patient agent used for clinician training.
Your profile must stick closely to what is actually supported by the transcript.

Requirements:
- Work ONLY from patient speech, not from the clinician's assumptions.
- Focus on stable patterns in behavior, affect, thinking, and language use.
- Identify concrete language signatures (recurrent phrases, tics, constructions)
  that a simulator can mimic.
- Include both strengths and vulnerabilities.
- Include any apparent risk markers, but do not over-pathologize.

Output:
- Return a single JSON object with the exact keys expected by the schema:
  summary, communication_style, emotional_tone, cognitive_patterns,
  interpersonal_patterns, coping_strategies, strengths, vulnerabilities,
  risk_markers, language_signatures, evidence_quotes.
""",
)


@dataclass
class BLPExtractor:
    """
    Second-stage agent in the top branch of the architecture:

    Anonymized transcript → Behavioral & Linguistic Profile (BLP).
    """

    model: str = get_model_for_agent("blp_extraction", "gemini/gemini-3-pro-preview")
    max_tokens: int = get_max_tokens_for_agent("blp_extraction", 2048)

    def extract(self, anonymized_transcript: str) -> BehavioralLinguisticProfile:
        """
        Extract a Behavioral & Linguistic Profile from an anonymized transcript.
        """

        if not anonymized_transcript.strip():
            raise ValueError("Transcript is empty; cannot extract BLP.")

        content = chat_completion(
            model=self.model,
            messages=[
                {"role": "system", "content": BLP_EXTRACTION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "Here is the anonymized transcript. "
                        "Return ONLY a JSON object with the requested fields.\n\n"
                        f"{anonymized_transcript}"
                    ),
                },
            ],
            response_format={"type": "json_object"},
            max_tokens=self.max_tokens,
        )
        if not content:
            raise RuntimeError("Model returned empty content while extracting BLP.")

        try:
            data = json.loads(content)
        except Exception:
            # Be robust to minor format issues by coercing to a JSON object.
            data = coerce_json_object(content)

        # --- Fix for "Russian Doll" JSON (nested in summary or response) ---
        if isinstance(data, dict) and len(data) < 3:
            # Check if 'summary' or 'response' holds the real JSON as a string
            nested_key = None
            if "summary" in data and isinstance(data["summary"], str) and "{" in data["summary"]:
                nested_key = "summary"
            elif "response" in data and isinstance(data["response"], str) and "{" in data["response"]:
                nested_key = "response"
            
            if nested_key:
                try:
                    nested_json = data[nested_key]
                    # Find the first { and last } to isolate JSON
                    start = nested_json.find("{")
                    end = nested_json.rfind("}")
                    if start != -1 and end != -1:
                        nested_content = nested_json[start : end + 1]
                        try:
                            inner_data = json.loads(nested_content)
                        except Exception:
                            inner_data = coerce_json_object(nested_content)
                        
                        # If successful, merge it into data (overwriting the malformed field)
                        if isinstance(inner_data, dict) and len(inner_data) > 1:
                            data.update(inner_data)
                            # If it was 'response', remove it. If 'summary', we overwritten it or need to check.
                            # Actually, if inner_data has 'summary', it overwrote the string summary.
                            # If inner_data doesn't have 'summary', we might want to keep the original string as summary?
                            # But usually the string IS the JSON, so it's not a valid summary text.
                            # Ideally we prefer the inner 'summary' if present.
                            if nested_key == "response":
                                data.pop("response", None)
                except Exception:
                    pass # Fallback to original data
        # -------------------------------------------------------------------

        # Normalize required string fields even if model emits dicts/objects.
        string_fields: List[str] = [
            "summary",
            "communication_style",
            "emotional_tone",
            "cognitive_patterns",
            "interpersonal_patterns",
        ]
        for field in string_fields:
            value = data.get(field, "")
            if value is None:
                data[field] = ""
            elif not isinstance(value, str):
                data[field] = str(value)

        # The model sometimes returns string blobs or dicts instead of lists for certain fields.
        # Normalize those so they always become List[str] before validation.
        list_fields: List[str] = [
            "coping_strategies",
            "strengths",
            "vulnerabilities",
            "risk_markers",
            "language_signatures",
            "evidence_quotes",
        ]
        def _normalize_to_str_list(value) -> List[str]:
            if value is None:
                return []
            # Already a list: coerce each element to a clean string
            if isinstance(value, list):
                out: List[str] = []
                for item in value:
                    if isinstance(item, str):
                        s = item.strip()
                        if s:
                            out.append(s)
                    elif isinstance(item, dict):
                        # Flatten dict values; ignore keys for cleanliness
                        for v in item.values():
                            if isinstance(v, str):
                                s = v.strip()
                                if s:
                                    out.append(s)
                            elif v is not None:
                                out.append(str(v))
                    elif item is not None:
                        out.append(str(item))
                return out
            # Dict: flatten values (which may be lists or scalars)
            if isinstance(value, dict):
                out: List[str] = []
                for v in value.values():
                    if isinstance(v, list):
                        for sub in v:
                            if isinstance(sub, str):
                                s = sub.strip()
                                if s:
                                    out.append(s)
                            elif sub is not None:
                                out.append(str(sub))
                    elif isinstance(v, str):
                        s = v.strip()
                        if s:
                            out.append(s)
                    elif v is not None:
                        out.append(str(v))
                return out
            # String: split on newlines/bullets
            if isinstance(value, str):
                lines = [
                    line.strip(" -•\t")
                    for line in value.replace("\r", "").split("\n")
                    if line.strip(" -•\t")
                ]
                return lines or [value.strip()]
            # Fallback: best-effort stringification
            return [str(value)]

        for field in list_fields:
            data[field] = _normalize_to_str_list(data.get(field))

        return BehavioralLinguisticProfile.model_validate(data)
