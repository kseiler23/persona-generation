from __future__ import annotations

"""
Offline DSPy loop to optimize the BLP extraction prompt using the critique agent.

This does NOT change the runtime API. Instead, it lets you:

1. Define a small training set of examples (raw transcript + raw case).
2. For each candidate BLP extractor prompt, run:
   transcript -> BLP -> simulated patient conversation -> critique.
3. Use the critique scores (e.g., persona alignment, clinical alignment) as a reward signal.
4. Let DSPy GEPA (or COPRO) perform reflective instruction optimization over the
   BLP extraction step (no few-shot demo bootstrapping).

You can then manually copy the best-found prompt back into
`BLP_EXTRACTION_SYSTEM_PROMPT` in `blp_extractor.py`, or load it from disk.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import dspy

# Attempt to import GEPA (newer) or fallback to COPRO (standard)
try:
    from dspy.teleprompt import GEPA
except ImportError:
    try:
        from dspy.optimizers import GEPA
    except ImportError:
        GEPA = None

try:
    from dspy.teleprompt import COPRO
except ImportError:
    from dspy.optimizers import COPRO

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


def configure_dspy(model_name: str = "gemini/gemini-3-pro-preview") -> None:
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
    model_name: str = "gemini/gemini-3-pro-preview",
) -> Tuple[List[ConversationTurn], BehavioralLinguisticProfile, PatientProfile]:
    """
    Given a candidate BLP (as JSON text) and a training example, run a short
    simulated conversation that we will feed into the critique agent.
    """

    blp = BehavioralLinguisticProfile.model_validate_json(blp_json)
    patient_profile = builder.build_from_case(example.raw_case)

    agent = SimulatedPatientAgent(
        blp=blp, 
        patient_profile=patient_profile,
        model=model_name
    )

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
    model_name: str = "gemini/gemini-3-pro-preview",
) -> float:
    """
    Compute a scalar reward from the critique agent for a single example.
    """

    conversation, blp, patient_profile = run_single_simulation(
        blp_json=blp_json,
        example=example,
        builder=builder,
        model_name=model_name,
    )

    critique_agent = PersonaCritiqueAgent(
        blp=blp,
        patient_profile=patient_profile,
        raw_transcript=example.raw_transcript,
        raw_case=example.raw_case,
        model=model_name,
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
    model_name: str = "gemini/gemini-3-pro-preview",
) -> Callable[[dspy.Example, dspy.Prediction, Optional[dspy.Trace]], float]:
    """
    Wrap the critique-based reward into a DSPy metric function.
    """

    # Map DSPy examples back to our richer BLPLearningExample objects.
    index = {i: ex for i, ex in enumerate(examples)}
    calls = 0
    best_score = float("-inf")

    def metric(
        example: dspy.Example,
        prediction: dspy.Prediction,
        trace: Optional[dspy.Trace] = None,
        pred_name: Optional[str] = None,
        pred_trace: Optional[dspy.Trace] = None,
    ) -> float:
        # Check cooperative cancellation before doing any heavy work
        if should_cancel is not None:
            try:
                if should_cancel():
                    raise OptimizationCancelled("Optimization cancelled by user")
            except OptimizationCancelled:
                raise
            except Exception:
                # If the predicate itself fails, treat as cancelled to be safe.
                raise OptimizationCancelled("Optimization cancelled")
        
        # DSPy passes the fields from the signature; we need to know which
        # BLPLearningExample this corresponds to. We stash its index in the
        # example metadata.
        ex_id = getattr(example, "ex_id", None)
        if ex_id is None or ex_id not in index:
            return 0.0

        blp_json = getattr(prediction, "blp_json", "")
        if not blp_json:
            return 0.0

        try:
            score = critique_reward(
                example=index[ex_id],
                blp_json=blp_json,
                builder=builder,
                w_clinical=w_clinical,
                w_persona=w_persona,
                model_name=model_name,
            )
        except Exception:
            # If simulation or critique fails, return 0 reward
            return 0.0

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
    model_name: str = "gemini/gemini-3-pro-preview",
    w_clinical: float = 0.0,
    w_persona: float = 1.0,
    # COPRO/GEPA budget knobs
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
    Run a DSPy optimization loop (GEPA if available, else COPRO) to improve the BLP extraction prompt.

    - `examples`: small set of (raw_transcript, raw_case, doctor_script)
    - `reflection_minibatch_size`: maps to COPRO `breadth` or GEPA `reflection_minibatch_size`
    - `max_metric_calls`: used to estimate COPRO `depth` or GEPA budget
    """

    if not skip_configure:
        try:
            configure_dspy(model_name)
        except Exception:
            # If DSPy settings were configured by another thread, ignore reconfigure errors.
            pass

    # Configure reflection LM (required by DSPy optimizers).
    # We use the passed model_name directly. The caller (API or script) is responsible
    # for determining the correct model (from config or user input).
    try:
        reflection_model = model_name
        
        # Only look up max tokens from config, as that's a model-specific property
        # that might not be passed in.
        reflection_max_tokens_any = get_value("gepa", "reflection_max_tokens", 32000)
        try:
            reflection_max_tokens = int(reflection_max_tokens_any) if reflection_max_tokens_any is not None else 32000
        except Exception:
            reflection_max_tokens = 32000
    except Exception:
        reflection_model = model_name
        reflection_max_tokens = 32000
    
    # If model is standard OpenAI GPT-4/4o, clamp max_tokens safely if it seems too high
    # (Standard GPT-4 output limit is 4k-16k depending on version)
    if "gpt-4" in reflection_model and reflection_max_tokens > 16000 and "gpt-5" not in reflection_model:
        reflection_max_tokens = 4096  # Sane default for older models

    reflection_lm = dspy.LM(
        model=reflection_model,
        temperature=1.0,
        max_tokens=reflection_max_tokens,
        # Force JSON object to prevent "list object has no attribute items" error
        # when COPRO parses the proposed instructions.
        response_format={"type": "json_object"},
    )

    # Seed the DSPy Signature instruction with the real runtime system prompt from YAML.
    # This ensures the optimizer edits the exact instruction used in production.
    seed_instruction = read_prompt("blp_extraction", "system_prompt", "")
    if seed_instruction.strip():
        try:
            BLPExtractionSignature.__doc__ = seed_instruction  # type: ignore[attr-defined]
        except Exception:
            pass

    builder = PatientProfileBuilder(model=model_name)
    module = DSPyBLPExtractor()

    # Convert to DSPy examples
    dspy_examples: List[dspy.Example] = []
    raw_examples: List[BLPLearningExample] = list(examples)

    for idx, ex in enumerate(raw_examples):
        d_ex = dspy.Example(transcript=ex.raw_transcript).with_inputs("transcript")
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
        model_name=model_name, # Pass model_name to the metric factory
    )

    # Decide whether to use GEPA (preferred, if installed) or COPRO (fallback)
    if GEPA is not None:
        # GEPA optimization
        teleprompter = GEPA(
            metric=metric,
            reflection_lm=reflection_lm,
            reflection_minibatch_size=max(2, reflection_minibatch_size),
            candidate_selection_strategy=candidate_selection_strategy,
            max_metric_calls=max_metric_calls,
            track_stats=track_stats,
            # Pass other params if GEPA supports them, but these are the core ones
        )
        # GEPA compile signature usually doesn't require eval_kwargs
        optimized_module = teleprompter.compile(module, trainset=dspy_examples)
    
    else:
        # COPRO Fallback
        # Calculate depth from budget
        breadth = max(2, reflection_minibatch_size)
        train_size = len(dspy_examples)
        if train_size > 0:
            depth = max(1, int(max_metric_calls / (max(1, breadth * train_size))))
        else:
            depth = 3

        teleprompter = COPRO(
            metric=metric,
            prompt_model=reflection_lm,
            breadth=breadth,
            depth=depth,
            track_last_step=track_stats,
        )
        
        # COPRO often needs eval_kwargs={} to avoid crashes in certain versions
        optimized_module = teleprompter.compile(
            module, 
            trainset=dspy_examples, 
            eval_kwargs={}
        )

    # Try to write back the evolved instruction to prompts.yaml so runtime uses it.
    evolved = _extract_predict_instructions(optimized_module)
    if isinstance(evolved, str) and evolved.strip():
        try:
            write_prompt("blp_extraction", "system_prompt", evolved)
        except Exception:
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
    """

    examples = [
        BLPLearningExample(
            raw_transcript="... raw interview transcript 1 ...",
            raw_case="... raw case material 1 ...",
            doctor_script=[
                "Hi, I'm Dr. X. What brought you in today?",
            ],
        ),
    ]

    out_path = Path("optimized_blp_extractor.json")
    optimize_blp_prompt(
        examples=examples,
        model_name="gemini/gemini-3-pro-preview",
        w_clinical=0.0,
        w_persona=1.0,
        max_metric_calls=200,
        output_path=out_path,
    )


if __name__ == "__main__":
    example_usage()
