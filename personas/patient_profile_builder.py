from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List

from .llm import chat_completion
from .models import PatientProfile


PATIENT_PROFILE_SYSTEM_PROMPT = """
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
"""


@dataclass
class PatientProfileBuilder:
    """
    Bottom branch of the architecture:

    Raw case (structured + unstructured) → structured Patient Profile.
    """

    model: str = "gpt-4.1-mini"
    max_tokens: int = 1_024

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

        data = json.loads(content)

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

        return PatientProfile.model_validate(data)


