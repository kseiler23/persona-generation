from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List

from .config import get_value
from .llm import chat_completion
from .models import BehavioralLinguisticProfile, ConversationTurn, PatientProfile
from .json_utils import coerce_json_object


_DEFAULT_JUDGE_MODEL = str(
    get_value("defaults", "model", "gemini/gemini-3-pro-preview") or "gemini/gemini-3-pro-preview"
)


JUDGE_PROMPT = """
You are grading a simulated patient conversation for rule violations.

You will be given:
- Raw clinical case (ground truth facts)
- Behavioral & Linguistic Profile (BLP)
- Structured Patient Profile
- Conversation turns: doctor questions + patient answers

Score the patient responses. Count errors (integers) for:
- question_anticipation: Patient offers unasked info that anticipates future questions.
- clinical_fact_fidelity: Patient fabricates or alters clinical facts not in the case/profile.
- lay_terminology: Patient uses clinical jargon instead of lay words.
- question_restatement: Patient restates or parrots the doctor's question.
- persona_integrity: Patient invents clinically relevant details outside the BLP/profile.
- binary_answer_padding: For yes/no questions, patient adds extra info beyond a direct answer.

Output JSON with:
- per_category: { question_anticipation, clinical_fact_fidelity, lay_terminology, question_restatement, persona_integrity, binary_answer_padding } as integers >= 0
- total_errors: integer sum of the above
- notes: short strings list highlighting notable issues (optional)

Only return JSON. Be strict but fair; do not hallucinate missing case facts.
"""


@dataclass
class ConversationJudge:
    model: str = _DEFAULT_JUDGE_MODEL
    max_tokens: int = 512

    def evaluate(
        self,
        *,
        conversation: List[ConversationTurn],
        raw_case: str,
        blp: BehavioralLinguisticProfile,
        patient_profile: PatientProfile,
    ) -> Dict[str, Any]:
        convo_json = json.dumps([turn.model_dump() for turn in conversation], ensure_ascii=False, indent=2)
        payload = (
            "Raw clinical case:\n"
            f"{raw_case}\n\n"
            "Behavioral & Linguistic Profile (BLP):\n"
            f"{blp.model_dump_json(indent=2)}\n\n"
            "Structured Patient Profile:\n"
            f"{patient_profile.model_dump_json(indent=2)}\n\n"
            "Conversation turns (doctor/patient):\n"
            f"{convo_json}\n\n"
            "Grade and return the JSON described."
        )

        content = chat_completion(
            model=self.model,
            messages=[
                {"role": "system", "content": JUDGE_PROMPT},
                {"role": "user", "content": payload},
            ],
            response_format={"type": "json_object"},
            max_tokens=self.max_tokens,
        )
        if not content:
            # Fall back to a safe zeroed report instead of blowing up the pipeline
            return {
                "per_category": {
                    "question_anticipation": 0,
                    "clinical_fact_fidelity": 0,
                    "lay_terminology": 0,
                    "question_restatement": 0,
                    "persona_integrity": 0,
                    "binary_answer_padding": 0,
                },
                "total_errors": 0,
                "notes": ["Judge model returned empty content."],
            }

        try:
            data = json.loads(content)
        except Exception:
            data = coerce_json_object(content)

        per_cat = data.get("per_category", {}) or {}
        total = data.get("total_errors", 0)
        # Ensure integers, non-negative
        clean_per: Dict[str, int] = {}
        for key in [
            "question_anticipation",
            "clinical_fact_fidelity",
            "lay_terminology",
            "question_restatement",
            "persona_integrity",
            "binary_answer_padding",
        ]:
            try:
                val = int(per_cat.get(key, 0))
            except Exception:
                val = 0
            clean_per[key] = max(0, val)
        try:
            total_int = int(total)
            if total_int < 0:
                total_int = 0
        except Exception:
            total_int = sum(clean_per.values())

        return {
            "per_category": clean_per,
            "total_errors": total_int if total_int is not None else sum(clean_per.values()),
            "notes": data.get("notes", []),
        }
