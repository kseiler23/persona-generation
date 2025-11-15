from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List

from .llm import chat_completion
from .models import BehavioralLinguisticProfile


BLP_EXTRACTION_SYSTEM_PROMPT = """
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
"""


@dataclass
class BLPExtractor:
    """
    Second-stage agent in the top branch of the architecture:

    Anonymized transcript → Behavioral & Linguistic Profile (BLP).
    """

    model: str = "gpt-4.1-mini"
    max_tokens: int = 1_024

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

        data = json.loads(content)

        # The model sometimes returns string blobs instead of lists for certain fields.
        # Normalize those so they always become List[str] before validation.
        list_fields: List[str] = [
            "coping_strategies",
            "strengths",
            "vulnerabilities",
            "risk_markers",
            "language_signatures",
            "evidence_quotes",
        ]
        for field in list_fields:
            value = data.get(field)
            if isinstance(value, str):
                # Try to split on newlines / bullets; fall back to a single-element list.
                lines = [
                    line.strip(" -•\t")
                    for line in value.replace("\r", "").split("\n")
                    if line.strip(" -•\t")
                ]
                data[field] = lines or [value.strip()]

        return BehavioralLinguisticProfile.model_validate(data)


