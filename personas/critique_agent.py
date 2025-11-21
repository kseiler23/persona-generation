from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List

from .llm import chat_completion
from .models import (
    BehavioralLinguisticProfile,
    CritiqueResult,
    ConversationTurn,
    PatientProfile,
)
from .prompts import read_prompt
from .config import get_model_for_agent, get_max_tokens_for_agent
from .json_utils import coerce_json_object


CRITIQUE_SYSTEM_PROMPT = read_prompt(
    "critique",
    "system_prompt",
    """
You are an expert clinical supervisor and conversation analyst.

You evaluate simulated patient encounters used for clinician training.

You will be given:
- The raw interview transcript of the real person whose behavior and language
  the persona is meant to emulate.
- A Behavioral & Linguistic Profile (BLP) distilled from that transcript.
- The raw clinical case description and a structured Patient Profile.
- A transcript of a simulated encounter between a clinician ("doctor") and the
  simulated patient ("patient").

Your tasks:
1. Clinical alignment
   - Judge how faithfully the simulated patient's answers reflect the clinical
     facts and constraints of the raw case and Patient Profile.
   - Penalize invented diagnoses, major contradictions, or omission of core
     presentation features when they would reasonably show up in this setting.
2. Persona / linguistic alignment
   - Judge how well the simulated patient's language, affect, and behavior
     match the real person's patterns as reflected in the raw transcript and
     the BLP (communication style, emotional tone, language signatures, etc.).
   - Focus on style and behavioral consistency, not just factual overlap.
3. Safety and professionalism
   - Note any concerning content (e.g., giving clinical advice, unsafe
     instructions, breaking role, violating guardrails).
4. Suggested improvements
   - Provide concise, actionable suggestions that would move the simulation
     closer to the target persona and case.

Scoring:
- Use numeric scores in [0.0, 1.0].
- 0.90–1.00 = very strong alignment
- 0.70–0.89 = generally good but with some issues
- 0.40–0.69 = mixed / inconsistent
- 0.00–0.39 = poor alignment

Output:
- Return a SINGLE JSON object with exactly these top-level keys:
  - overall_comment: string
  - clinical_alignment: { score, label, explanation }
  - persona_alignment: { score, label, explanation }
  - safety_flags: string[]
  - suggested_improvements: string[]
  - per_turn: TurnCritique[]

TurnCritique objects must have:
- turn_index: integer (0-based)
- doctor_utterance: string
- patient_reply: string
- notes: string
- issues: string[]

General rules:
- Ground all judgments explicitly in the provided materials.
- When citing evidence, quote short snippets rather than paraphrasing vaguely.
- If information is insufficient to judge something, say so explicitly rather
  than guessing.
- Do NOT include any commentary outside the JSON object.
""",
)
 


@dataclass
class PersonaCritiqueAgent:
    """
    Critique agent that sits "to the side" of the main pipeline.

    It has access to:
    - Raw interview transcript + Behavioral & Linguistic Profile (persona branch)
    - Raw clinical case + Patient Profile (case branch)
    - A transcript of the simulated encounter

    and returns a structured CritiqueResult.
    """

    blp: BehavioralLinguisticProfile
    patient_profile: PatientProfile
    raw_transcript: str
    raw_case: str
    model: str = get_model_for_agent("critique", "gemini-3-pro-preview")
    max_tokens: int = get_max_tokens_for_agent("critique", 2048)

    def critique(self, conversation: List[ConversationTurn]) -> CritiqueResult:
        """
        Run a single-pass critique of a simulated encounter.
        """

        if not conversation:
            raise ValueError("Conversation is empty; nothing to critique.")

        if not self.raw_transcript.strip():
            raise ValueError("Raw transcript is empty; critique agent needs it for alignment.")

        if not self.raw_case.strip():
            raise ValueError("Raw case is empty; critique agent needs it for clinical alignment.")

        blp_json = self.blp.model_dump_json(indent=2)
        patient_json = self.patient_profile.model_dump_json(indent=2)
        convo_json = json.dumps(
            [turn.model_dump() for turn in conversation],
            indent=2,
            ensure_ascii=False,
        )

        
        user_content = (
            "You are evaluating the following materials.\n\n"
            "=== Raw interview transcript (source persona) ===\n"
            f"{self.raw_transcript}\n\n"
            "=== Behavioral & Linguistic Profile (BLP) ===\n"
            f"{blp_json}\n\n"
            "=== Raw clinical case description ===\n"
            f"{self.raw_case}\n\n"
            "=== Structured Patient Profile ===\n"
            f"{patient_json}\n\n"
            "=== Simulated encounter transcript ===\n"
            f"{convo_json}\n\n"
            "Now produce the JSON critique object described in the system instructions."
        )

        content = chat_completion(
            model=self.model,
            messages=[
                {"role": "system", "content": CRITIQUE_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            response_format={"type": "json_object"},
            max_tokens=self.max_tokens,
        )

        if not content:
            raise RuntimeError("Model returned empty content while generating critique.")

        try:
            data = json.loads(content)
        except Exception:
            data = coerce_json_object(content)
        return CritiqueResult.model_validate(data)


