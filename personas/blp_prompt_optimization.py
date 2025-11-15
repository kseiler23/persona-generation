from __future__ import annotations

"""
Offline DSPy loop to optimize the BLP extraction prompt using the critique agent.

This does NOT change the runtime API. Instead, it lets you:

1. Define a small training set of examples (raw transcript + raw case).
2. For each candidate BLP extractor prompt, run:
   transcript -> BLP -> simulated patient conversation -> critique.
3. Use the critique scores (clinical + persona alignment) as a reward signal.
4. Let DSPy search over better instructions / few-shot patterns for the
   BLP extraction step.

You can then manually copy the best-found prompt back into
`BLP_EXTRACTION_SYSTEM_PROMPT` in `blp_extractor.py`, or load it from disk.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Tuple

import dspy

from .critique_agent import PersonaCritiqueAgent
from .models import (
    BehavioralLinguisticProfile,
    ConversationTurn,
    PatientProfile,
)
from .patient_profile_builder import PatientProfileBuilder
from .simulated_patient_agent import SimulatedPatientAgent


# --- DSPy setup ----------------------------------------------------------------


def configure_dspy(model_name: str = "gpt-4.1-mini") -> None:
    """
    Point DSPy at the same underlying chat model we use elsewhere via LiteLLM.

    You can also configure this to talk directly to OpenAI/Anthropic if you
    prefer, but reusing the same base model keeps behavior consistent.
    """

    lm = dspy.LM(
        model_name,
        # DSPy expects an OpenAI-style endpoint by default; you can configure
        # keys / base URLs via environment variables, same as LiteLLM.
    )
    dspy.settings.configure(lm=lm)


class BLPExtractionSignature(dspy.Signature):
    """
    Signature for the BLP extraction step, framed as JSON-producing text.
    """

    transcript: str = dspy.InputField(
        desc="An anonymized or raw clinical interview transcript.",
    )
    blp_json: str = dspy.OutputField(
        desc=(
            "A JSON object with the Behavioral & Linguistic Profile fields "
            "expected by the `BehavioralLinguisticProfile` schema."
        ),
    )


class DSPyBLPExtractor(dspy.Module):
    """
    Drop-in BLP extractor whose prompt will be optimized by DSPy.
    """

    def __init__(self) -> None:
        super().__init__()
        self.extract = dspy.Predict(BLPExtractionSignature)

    def forward(self, transcript: str) -> str:
        pred = self.extract(transcript=transcript)
        return pred.blp_json


# --- Data container for optimization ------------------------------------------


@dataclass
class BLPLearningExample:
    """
    Single training example for prompt optimization.

    You can construct these from your existing datasets or synthetic cases.
    """

    raw_transcript: str
    raw_case: str
    # Optional scripted doctor utterances to probe the simulation.
    doctor_script: List[str]


# --- Reward / metric based on critique agent ----------------------------------


def run_single_simulation(
    *,
    blp_json: str,
    example: BLPLearningExample,
    builder: PatientProfileBuilder,
    convo_turns: int = 4,
) -> Tuple[List[ConversationTurn], BehavioralLinguisticProfile, PatientProfile]:
    """
    Given a candidate BLP (as JSON text) and a training example, run a short
    simulated conversation that we will feed into the critique agent.
    """

    blp = BehavioralLinguisticProfile.model_validate_json(blp_json)
    patient_profile = builder.build_from_case(example.raw_case)

    agent = SimulatedPatientAgent(blp=blp, patient_profile=patient_profile)

    conversation: List[ConversationTurn] = []
    for idx, doctor_utt in enumerate(example.doctor_script[:convo_turns]):
        reply = agent.reply(doctor_utt)
        conversation.append(ConversationTurn(role="doctor", content=doctor_utt))
        conversation.append(ConversationTurn(role="patient", content=reply))

    return conversation, blp, patient_profile


def critique_reward(
    *,
    example: BLPLearningExample,
    blp_json: str,
    builder: PatientProfileBuilder,
    w_clinical: float = 0.5,
    w_persona: float = 0.5,
) -> float:
    """
    Compute a scalar reward from the critique agent for a single example.
    """

    conversation, blp, patient_profile = run_single_simulation(
        blp_json=blp_json,
        example=example,
        builder=builder,
    )

    critique_agent = PersonaCritiqueAgent(
        blp=blp,
        patient_profile=patient_profile,
        raw_transcript=example.raw_transcript,
        raw_case=example.raw_case,
    )
    critique = critique_agent.critique(conversation)

    clinical = float(critique.clinical_alignment.score)
    persona = float(critique.persona_alignment.score)
    return w_clinical * clinical + w_persona * persona


def make_metric(
    *,
    examples: List[BLPLearningExample],
    builder: PatientProfileBuilder,
    w_clinical: float = 0.5,
    w_persona: float = 0.5,
) -> Callable[[dspy.Example, dspy.Prediction, dspy.Trace], float]:
    """
    Wrap the critique-based reward into a DSPy metric function.
    """

    # Map DSPy examples back to our richer BLPLearningExample objects.
    index = {i: ex for i, ex in enumerate(examples)}

    def metric(example: dspy.Example, prediction: dspy.Prediction, trace: dspy.Trace) -> float:  # type: ignore[override]
        # DSPy passes the fields from the signature; we need to know which
        # BLPLearningExample this corresponds to. We stash its index in the
        # example metadata.
        ex_id = getattr(example, "ex_id", None)
        if ex_id is None or ex_id not in index:
            return 0.0

        blp_json = getattr(prediction, "blp_json", "")
        if not blp_json:
            return 0.0

        return critique_reward(
            example=index[ex_id],
            blp_json=blp_json,
            builder=builder,
            w_clinical=w_clinical,
            w_persona=w_persona,
        )

    return metric


# --- Top-level optimization routine ------------------------------------------


def optimize_blp_prompt(
    *,
    examples: Iterable[BLPLearningExample],
    model_name: str = "gpt-4.1-mini",
    w_clinical: float = 0.5,
    w_persona: float = 0.5,
    num_candidates: int = 8,
    num_iterations: int = 3,
    output_path: Path | None = None,
) -> DSPyBLPExtractor:
    """
    Run a DSPy teleprompter loop to improve the BLP extraction prompt.

    - `examples`: small set of (raw_transcript, raw_case, doctor_script)
    - `w_clinical`, `w_persona`: mix weights for the reward
    - `num_candidates`, `num_iterations`: search budget
    - `output_path`: optional path to save the resulting prompt/program
    """

    configure_dspy(model_name)

    builder = PatientProfileBuilder(model=model_name)
    module = DSPyBLPExtractor()

    # Convert to DSPy examples; we attach ex_id so the metric can find the
    # corresponding BLPLearningExample.
    dspy_examples: List[dspy.Example] = []
    raw_examples: List[BLPLearningExample] = list(examples)

    for idx, ex in enumerate(raw_examples):
        d_ex = dspy.Example(transcript=ex.raw_transcript)
        setattr(d_ex, "ex_id", idx)
        dspy_examples.append(d_ex)

    metric = make_metric(
        examples=raw_examples,
        builder=builder,
        w_clinical=w_clinical,
        w_persona=w_persona,
    )

    teleprompter = dspy.teleprompt.BootstrapFewShotWithRandomSearch(
        metric=metric,
        max_train_iterations=num_iterations,
        num_candidates=num_candidates,
    )

    optimized_module = teleprompter.compile(
        module=module,
        trainset=dspy_examples,
    )

    if output_path is not None:
        program = optimized_module.to_dict()
        output_path.write_text(json.dumps(program, indent=2))

    return optimized_module


def example_usage() -> None:
    """
    Minimal sketch of how you might call `optimize_blp_prompt` from a script.

    Replace the inline examples with your own transcripts / cases.
    """

    examples = [
        BLPLearningExample(
            raw_transcript="... raw interview transcript 1 ...",
            raw_case="... raw case material 1 ...",
            doctor_script=[
                "Hi, I'm Dr. X. What brought you in today?",
                "How has this been affecting your work or school?",
                "Have you noticed any changes in your sleep or appetite?",
            ],
        ),
        BLPLearningExample(
            raw_transcript="... raw interview transcript 2 ...",
            raw_case="... raw case material 2 ...",
            doctor_script=[
                "Can you tell me a bit about what's been hardest lately?",
                "How do you usually cope when things feel overwhelming?",
            ],
        ),
    ]

    out_path = Path("optimized_blp_extractor.json")
    optimize_blp_prompt(
        examples=examples,
        model_name="gpt-4.1-mini",
        w_clinical=0.5,
        w_persona=0.5,
        num_candidates=6,
        num_iterations=2,
        output_path=out_path,
    )
    print(f"Saved optimized BLP program to {out_path.resolve()}")


if __name__ == "__main__":
    example_usage()



