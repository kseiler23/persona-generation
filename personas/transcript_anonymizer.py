from __future__ import annotations

from dataclasses import dataclass

from .llm import chat_completion
from .config import get_model_for_agent


ANONYMIZATION_SYSTEM_PROMPT = """
You are an expert in de-identifying clinical interview transcripts.

Your job is to take raw interview transcripts and produce an anonymized version
that can safely be used for downstream modeling.

Rules:
- Replace all clinician names with "Interviewer".
- Replace all patient names with "Respondent".
- Remove or generalize any personally identifying information (exact names, addresses,
  workplaces, phone numbers, email addresses, exact birth dates, specific hospitals,
  or other obvious identifiers).
- Preserve the conversational flow, meaning, and emotional content.
- Preserve turn-taking structure where possible (e.g., labels like 'Interviewer:' /
  'Respondent:' on separate lines).
- Do not summarize. Return a turn-by-turn transcript with the same level of detail.
"""


@dataclass
class TranscriptAnonymizer:
    """
    First-stage agent in the pipeline:

    Raw interview transcript → anonymized transcript that is safe to feed into the BLP extractor.
    """

    model: str = get_model_for_agent("anonymizer", "gemini/gemini-3-pro-preview")

    def anonymize(self, transcript: str) -> str:
        """
        Anonymize a raw transcript according to the architecture's first step.
        """

        if not transcript.strip():
            raise ValueError("Transcript is empty; nothing to anonymize.")

        content = chat_completion(
            model=self.model,
            messages=[
                {"role": "system", "content": ANONYMIZATION_SYSTEM_PROMPT},
                {"role": "user", "content": transcript},
            ],
        )
        return content
