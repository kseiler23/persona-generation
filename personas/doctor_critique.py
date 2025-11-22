from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List

from .llm import chat_completion
from .models import (
    SimulationTrace, 
    PatientProfile, 
    DoctorCritiqueResult,
    ConversationTurn
)
from .prompts import read_prompt
from .config import get_model_for_agent, get_max_tokens_for_agent
from .json_utils import coerce_json_object


@dataclass
class DoctorCritiqueAgent:
    """
    Evaluates the performance of the Simulated Doctor.
    Serves as the reward model for the optimization loop.
    """
    
    model: str = get_model_for_agent("doctor_critique", "gemini/gemini-3-pro-preview")
    max_tokens: int = get_max_tokens_for_agent("doctor_critique", 2048)
    system_prompt: str = ""

    def __post_init__(self) -> None:
        self.system_prompt = read_prompt("doctor_critique", "system_prompt", "")
        if not self.system_prompt:
            self.system_prompt = "Evaluate the doctor's performance."

    def critique(self, trace: SimulationTrace, patient_profile: PatientProfile) -> DoctorCritiqueResult:
        """
        Compare the simulation trace against the ground truth profile.
        """
        
        # Prepare context for the LLM Judge
        transcript_text = "\n".join([f"{t.role.upper()}: {t.content}" for t in trace.transcript])
        ground_truth_dx = ", ".join(patient_profile.primary_diagnoses)
        
        user_content = (
            f"=== DOCTOR'S FINAL DIAGNOSIS ===\n"
            f"{trace.doctor_diagnosis}\n\n"
            f"=== GROUND TRUTH DIAGNOSIS (From Patient Profile) ===\n"
            f"{ground_truth_dx}\n\n"
            f"=== FULL TRANSCRIPT ===\n"
            f"{transcript_text}\n\n"
            f"Please evaluate the doctor's performance based on the system instructions."
        )
        
        content = chat_completion(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_content},
            ],
            response_format={"type": "json_object"},
            max_tokens=self.max_tokens,
        )
        
        if not content:
            raise RuntimeError("Critique model returned empty content.")

        try:
            data = json.loads(content)
        except Exception:
            data = coerce_json_object(content)

        # Post-process: Handle critique_text as list (LLM sometimes returns array)
        if "critique_text" in data and isinstance(data["critique_text"], list):
            data["critique_text"] = "\n".join(data["critique_text"])

        # Validate and return
        return DoctorCritiqueResult.model_validate(data)

