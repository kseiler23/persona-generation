from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .llm import chat_completion
from .prompts import read_prompt
from .config import get_model_for_agent, get_max_tokens_for_agent


@dataclass
class SimulatedDoctorAgent:
    """
    LLM-based agent acting as the clinician.
    
    It is "blind" to the patient profile and must gather information via dialogue.
    It signals completion by outputting a diagnosis in a specific format.
    """
    
    # Corrected key to match config.yaml "simulated_doctor"
    model: str = get_model_for_agent("simulated_doctor", "gemini/gemini-3-pro-preview")
    max_tokens: int = get_max_tokens_for_agent("simulated_doctor", 512)
    system_prompt: str = field(init=False)

    def __post_init__(self) -> None:
        self.system_prompt = read_prompt("simulated_doctor", "system_prompt", "")
        if not self.system_prompt:
            # Fallback if yaml is missing
            self.system_prompt = (
                "You are a doctor. Interview the patient to find a diagnosis. "
                "Output [DIAGNOSIS]: <diagnosis> when done."
            )

    def next_turn(self, history: List[Dict[str, str]]) -> str:
        """
        Generate the next doctor utterance given the conversation history.
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            *history
        ]

        content = chat_completion(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
        )
        
        return content
