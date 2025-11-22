"""Case filtering to determine if a case should be included in preprocessing."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict

from .llm_async import achat_completion


CASE_FILTER_PROMPT = """You are a medical case quality assessor for a clinical training system.

Your task is to evaluate whether a clinical case is suitable for training simulated patient interactions.

Evaluate the case based on these criteria:

1. **Clinical Content Quality**:
   - Does the case contain meaningful clinical information?
   - Is there sufficient detail about patient presentation, history, or diagnosis?
   - Are the clinical details coherent and realistic?

2. **Suitability for Simulation**:
   - Can this case be used to train clinicians in patient interactions?
   - Does it provide enough context for realistic patient simulation?
   - Would a simulated patient be able to portray this case?

3. **Completeness**:
   - Is there enough information about the patient's condition?
   - Are key clinical elements present (symptoms, history, or diagnosis)?

**Exclude cases that:**
- Are purely academic/theoretical without patient details
- Contain only laboratory results or imaging without clinical context
- Are literature reviews or meta-analyses
- Lack sufficient detail for patient simulation
- Are primarily about surgical techniques or procedures without patient context
- Are case series summaries without individual patient narratives

**Include cases that:**
- Have clear patient presentation and history
- Contain realistic clinical scenarios
- Provide enough detail for simulated patient interactions
- Include behavioral, psychological, or social aspects
- Have diagnostic or treatment decision-making elements

Return a JSON object with:
{
  "include": true/false,
  "reason": "Brief explanation (1-2 sentences) of why this case should or should not be included",
  "confidence": "high/medium/low"
}
"""


@dataclass
class CaseFilter:
    """
    Filters clinical cases to determine suitability for preprocessing.
    """

    model: str = "gpt-4o-mini"  # Use faster/cheaper model for filtering

    async def should_include(self, case_text: str, abstract: str = "") -> Dict[str, any]:
        """
        Determine if a case should be included in preprocessing.

        Args:
            case_text: The main case text
            abstract: Optional abstract text (used as fallback)

        Returns:
            Dictionary with keys: include (bool), reason (str), confidence (str)
        """
        # Combine case_text and abstract for evaluation
        full_text = case_text.strip()
        if not full_text and abstract:
            full_text = abstract.strip()
        elif abstract:
            full_text = f"{case_text}\n\n{abstract}"

        if not full_text:
            return {
                "include": False,
                "reason": "Empty case text and abstract",
                "confidence": "high"
            }

        # Truncate very long texts to save on API costs
        max_chars = 3000
        if len(full_text) > max_chars:
            full_text = full_text[:max_chars] + "..."

        try:
            content = await achat_completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": CASE_FILTER_PROMPT},
                    {
                        "role": "user",
                        "content": f"Evaluate this clinical case:\n\n{full_text}"
                    },
                ],
                response_format={"type": "json_object"},
                max_tokens=200,
            )

            result = json.loads(content)

            # Validate the response structure
            if "include" not in result:
                return {
                    "include": True,  # Default to including if filter fails
                    "reason": "Filter evaluation failed, including by default",
                    "confidence": "low"
                }

            return result

        except Exception as e:
            # If filtering fails, default to including the case
            return {
                "include": True,
                "reason": f"Filter error: {str(e)}, including by default",
                "confidence": "low"
            }
