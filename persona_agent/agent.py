from typing import List, Dict, Optional
from dataclasses import dataclass, field

from litellm import completion


@dataclass
class PersonaAgent:
    """
    Simple litellm-based chat agent that accepts instructions and a context,
    and allows interactive Q&A.
    """
    model: str
    instructions: str
    context: str
    messages: List[Dict[str, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        system_msg = {
            "role": "system",
            "content": self.instructions.strip(),
        }
        ctx = self.context.strip()
        self.messages = [system_msg]
        if ctx:
            self.messages.append({"role": "system", "content": ctx})

    def ask(self, user_input: str) -> str:
        self.messages.append({"role": "user", "content": user_input})
        resp = completion(model=self.model, messages=self.messages)
        try:
            content = resp.choices[0].message["content"]
        except Exception:
            content = ""
        self.messages.append({"role": "assistant", "content": content})
        return content


