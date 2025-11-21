from __future__ import annotations

"""
Offline DSPy loop to optimize the BLP extraction prompt using the critique agent.

This does NOT change the runtime API. Instead, it lets you:

1. Define a small training set of examples (raw transcript + raw case).
2. For each candidate BLP extractor prompt, run:
   transcript -> BLP -> simulated patient conversation -> critique.
3. Use the critique scores (e.g., persona alignment, clinical alignment) as a reward signal.
4. Let DSPy GEPA perform reflective instruction optimization over the
   BLP extraction step (no few-shot demo bootstrapping).

You can then manually copy the best-found prompt back into
`BLP_EXTRACTION_SYSTEM_PROMPT` in `blp_extractor.py`, or load it from disk.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import dspy

from .critique_agent import PersonaCritiqueAgent
from .prompts import read_prompt, write_prompt
from .config import get_value
from .models import (
    BehavioralLinguisticProfile,
    ConversationTurn,
    PatientProfile,
)
from .patient_profile_builder import PatientProfileBuilder
from .simulated_patient_agent import SimulatedPatientAgent


# --- DSPy setup ----------------------------------------------------------------


def configure_dspy(model_name: str = "gemini-3-pro-preview") -> None:
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


# --- Cancellation ---------------------------------------------------------------
class OptimizationCancelled(Exception):
    pass


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
    w_clinical: float = 0.0,
    w_persona: float = 1.0,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    total_budget: Optional[int] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> Callable[[dspy.Example, dspy.Prediction, dspy.Trace], float]:
    """
    Wrap the critique-based reward into a DSPy metric function.

    Note: This returns a scalar score. GEPA can also consume textual feedback,
    but a scalar works fine if you prefer simplicity.
    """

    # Map DSPy examples back to our richer BLPLearningExample objects.
    index = {i: ex for i, ex in enumerate(examples)}
    calls = 0
    best_score = float("-inf")

    def metric(
        example: dspy.Example,
        prediction: dspy.Prediction,
        trace: dspy.Trace,
        pred_name=None,
        pred_trace=None,
    ) -> float:  # type: ignore[override]
        # Check cooperative cancellation before doing any heavy work
        if should_cancel is not None:
            try:
                if should_cancel():
                    raise OptimizationCancelled("GEPA optimization cancelled by user")
            except OptimizationCancelled:
                raise
            except Exception:
                # If the predicate itself fails, treat as cancelled to be safe.
                raise OptimizationCancelled("GEPA optimization cancelled")
        # DSPy passes the fields from the signature; we need to know which
        # BLPLearningExample this corresponds to. We stash its index in the
        # example metadata.
        ex_id = getattr(example, "ex_id", None)
        if ex_id is None or ex_id not in index:
            return 0.0

        blp_json = getattr(prediction, "blp_json", "")
        if not blp_json:
            return 0.0

        score = critique_reward(
            example=index[ex_id],
            blp_json=blp_json,
            builder=builder,
            w_clinical=w_clinical,
            w_persona=w_persona,
        )
        nonlocal calls, best_score
        calls += 1
        if score > best_score:
            best_score = score
        if progress_callback is not None:
            try:
                pct = 0
                if isinstance(total_budget, int) and total_budget > 0:
                    pct = int(min(100, max(0, round(100 * calls / total_budget))))
                progress_callback(
                    {
                        "status": "running",
                        "metric_calls": calls,
                        "max_metric_calls": total_budget,
                        "percent": pct,
                        "latest_score": float(score),
                        "best_score": float(best_score if best_score != float('-inf') else 0.0),
                    }
                )
            except Exception:
                # Progress is best-effort; ignore callback errors
                pass
        return score

    return metric


# --- Top-level optimization routine ------------------------------------------


def _extract_predict_instructions(module: DSPyBLPExtractor) -> str | None:
    """
    Best-effort retrieval of the evolved instruction text from a compiled module.
    Works across different DSPy versions by trying multiple access paths.
    """
    # Attempt direct access via signature attributes
    try:
        sig = getattr(module.extract, "signature", None)
        if sig is not None:
            maybe = getattr(sig, "instructions", None)
            if isinstance(maybe, str) and maybe.strip():
                return maybe
            maybe = getattr(sig, "__doc__", None)
            if isinstance(maybe, str) and maybe.strip():
                return maybe
    except Exception:
        pass

    # Fallback: scan the serialized program dict for a plausible instruction blob
    try:
        program = module.to_dict()  # type: ignore[attr-defined]
    except Exception:
        program = None

    def _search(obj) -> str | None:
        target_markers = [
            "behavioral and linguistic profile (BLP)",
            "Return a single JSON object",
            "communication_style, emotional_tone, cognitive_patterns",
        ]
        if isinstance(obj, str):
            s = obj.strip()
            if len(s) > 200 and any(m in s for m in target_markers):
                return s
            return None
        if isinstance(obj, dict):
            for v in obj.values():
                hit = _search(v)
                if hit:
                    return hit
        if isinstance(obj, list):
            for v in obj:
                hit = _search(v)
                if hit:
                    return hit
        return None

    if program is not None:
        return _search(program)
    return None


def optimize_blp_prompt(
    *,
    examples: Iterable[BLPLearningExample],
    model_name: str = "gemini-3-pro-preview",
    w_clinical: float = 0.0,
    w_persona: float = 1.0,
    # GEPA budgets (defaults tuned for light runs; adjust per need)
    reflection_minibatch_size: int = 3,
    candidate_selection_strategy: str = "pareto",
    max_metric_calls: int = 200,
    failure_score: float = 0.0,
    perfect_score: float = 1.0,
    use_merge: bool = True,
    track_stats: bool = True,
    output_path: Path | None = None,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
    skip_configure: bool = False,
) -> DSPyBLPExtractor:
    """
    Run a DSPy GEPA loop to improve the BLP extraction prompt.

    - `examples`: small set of (raw_transcript, raw_case, doctor_script)
    - `w_clinical`, `w_persona`: mix weights for the reward (set `w_clinical=0.0` for persona-only)
    - GEPA budget knobs: reflection_minibatch_size, max_metric_calls, etc.
    - `output_path`: optional path to save the resulting prompt/program
    """

    if not skip_configure:
        try:
            configure_dspy(model_name)
        except Exception:
            # If DSPy settings were configured by another thread, ignore reconfigure errors.
            pass

    # Configure reflection LM for GEPA (required by DSPy).
    try:
        reflection_model_any = get_value("gepa", "reflection_model", model_name)
        reflection_model = str(reflection_model_any) if reflection_model_any is not None else model_name
        reflection_max_tokens_any = get_value("gepa", "reflection_max_tokens", 32000)
        try:
            reflection_max_tokens = int(reflection_max_tokens_any) if reflection_max_tokens_any is not None else 32000
        except Exception:
            reflection_max_tokens = 32000
    except Exception:
        reflection_model = model_name
        reflection_max_tokens = 32000
    reflection_lm = dspy.LM(
        model=reflection_model,
        temperature=1.0,
        max_tokens=reflection_max_tokens,
    )

    # Seed the DSPy Signature instruction with the real runtime system prompt from YAML.
    # This ensures GEPA edits the exact instruction used in production.
    seed_instruction = read_prompt("blp_extraction", "system_prompt", "")
    if seed_instruction.strip():
        try:
            BLPExtractionSignature.__doc__ = seed_instruction  # type: ignore[attr-defined]
        except Exception:
            # If for some reason doc assignment fails, proceed with Signature default.
            pass

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
        progress_callback=progress_callback,
        total_budget=max_metric_calls,
        should_cancel=should_cancel,
    )

    # GEPA reflective optimizer (no demos; edits instruction text of the predictor)
    teleprompter = dspy.GEPA(
        metric=metric,
        reflection_lm=reflection_lm,
        reflection_minibatch_size=reflection_minibatch_size,
        candidate_selection_strategy=candidate_selection_strategy,
        max_metric_calls=max_metric_calls,
        failure_score=failure_score,
        perfect_score=perfect_score,
        use_merge=use_merge,
        track_stats=track_stats,
    )

    optimized_module = teleprompter.compile(module, trainset=dspy_examples)

    # Try to write back the evolved instruction to prompts.yaml so runtime uses it.
    evolved = _extract_predict_instructions(optimized_module)
    if isinstance(evolved, str) and evolved.strip():
        try:
            write_prompt("blp_extraction", "system_prompt", evolved)
        except Exception:
            # Non-fatal: still return the optimized module.
            pass

    if output_path is not None:
        program = optimized_module.to_dict()
        output_path.write_text(json.dumps(program, indent=2))

    if progress_callback is not None:
        try:
            progress_callback(
                {
                    "status": "complete",
                    "metric_calls": None,
                    "max_metric_calls": max_metric_calls,
                    "percent": 100,
                }
            )
        except Exception:
            pass

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
        model_name="gemini-3-pro-preview",
        w_clinical=0.0,
        w_persona=1.0,
        reflection_minibatch_size=3,
        candidate_selection_strategy="pareto",
        max_metric_calls=200,
        use_merge=True,
        track_stats=True,
        output_path=out_path,
    )
    


if __name__ == "__main__":
    example_usage()
