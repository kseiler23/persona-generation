from __future__ import annotations

import json
import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

import dspy

from .models import (
    SimulationTrace, 
    DoctorCritiqueResult, 
    PatientProfile, 
    BehavioralLinguisticProfile,
    ConversationTurn
)
from .prompts import read_prompt, write_prompt
from .config import get_value
from .simulation_controller import SimulationController
from .simulated_patient_agent import SimulatedPatientAgent
from .simulated_doctor import SimulatedDoctorAgent
from .doctor_critique import DoctorCritiqueAgent

# --- Data Logging for GRPO ---
TRACE_LOG_DIR = Path("data/traces")
TRACE_LOG_DIR.mkdir(parents=True, exist_ok=True)

def log_trace_for_grpo(
    trace: SimulationTrace, 
    reward: DoctorCritiqueResult, 
    prompt_snapshot: str
) -> None:
    """
    Append the simulation trace to a JSONL file for future training.
    """
    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "session_id": trace.session_id,
        "case_id": trace.case_id,
        "prompt": prompt_snapshot,
        "transcript": [t.model_dump() for t in trace.transcript],
        "doctor_diagnosis": trace.doctor_diagnosis,
        "ground_truth": trace.ground_truth_diagnosis,
        "reward_score": reward.overall_score,
        "accuracy": reward.diagnostic_accuracy,
        "process": reward.process_score,
        "critique": reward.critique_text
    }
    
    filename = f"doctor_traces_{datetime.date.today()}.jsonl"
    with open(TRACE_LOG_DIR / filename, "a") as f:
        f.write(json.dumps(log_entry) + "\n")


# --- DSPy Components ---

class DoctorSignature(dspy.Signature):
    """
    Signature for the Doctor Agent's turn generation.
    """
    conversation_history: str = dspy.InputField(
        desc="The dialogue history so far."
    )
    doctor_utterance: str = dspy.OutputField(
        desc="The doctor's next question or statement. If ready to diagnose, output [DIAGNOSIS]: <diagnosis>."
    )


class DSPyDoctorModule(dspy.Module):
    """
    The module we want to optimize. 
    """
    def __init__(self):
        super().__init__()
        self.generate = dspy.Predict(DoctorSignature)

    def forward(self, conversation_history: str) -> str:
        return self.generate(conversation_history=conversation_history).doctor_utterance


class DSPyDoctorWrapper(SimulatedDoctorAgent):
    """
    Wraps the DSPy module to look like our standard SimulatedDoctorAgent
    so the Controller can use it.
    """
    def __init__(self, dspy_module: DSPyDoctorModule):
        # We don't call super().__post_init__ because we don't need the yaml prompt here
        # The prompt is managed by DSPy's signature/program
        self.dspy_module = dspy_module

    def next_turn(self, history: List[Dict[str, str]]) -> str:
        # Convert list-of-dicts history to string for DSPy
        history_str = ""
        for msg in history:
            role = msg["role"]
            content = msg["content"]
            if role == "system": continue # Skip system prompt in history text
            history_str += f"{role.upper()}: {content}\n"
            
        return self.dspy_module(conversation_history=history_str)


@dataclass
class DoctorTrainingCase:
    case_id: str
    patient_profile: PatientProfile
    blp: BehavioralLinguisticProfile


def configure_dspy(model_name: str = "gemini/gemini-3-pro-preview") -> None:
    lm = dspy.LM(model_name)
    dspy.settings.configure(lm=lm)


class DSPyDoctorLoop(dspy.Module):
    """
    The full program: Takes a case, runs the controller loop, returns the trace.
    """
    def __init__(self, model_name: str):
        super().__init__()
        self.doctor_inner = DSPyDoctorModule()
        self.controller = SimulationController()
        self.model_name = model_name

    def forward(self, case_obj: DoctorTrainingCase) -> dspy.Prediction:
        # Create wrappers
        doc_agent = DSPyDoctorWrapper(self.doctor_inner)
        pat_agent = SimulatedPatientAgent(
            blp=case_obj.blp, 
            patient_profile=case_obj.patient_profile,
            model=self.model_name
        )
        
        trace = self.controller.run_simulation(
            doctor_agent=doc_agent,
            patient_agent=pat_agent,
            case_id=case_obj.case_id,
            ground_truth_diagnosis=case_obj.patient_profile.primary_diagnoses,
            max_turns=8 # Keep short for training speed
        )
        
        # We return the trace wrapped in a Prediction
        return dspy.Prediction(trace=trace, diagnosis=trace.doctor_diagnosis)


def _extract_instruction(module: DSPyDoctorModule) -> str | None:
    """
    Helper to extract instructions from the optimized module.
    """
    try:
        sig = module.generate.signature
        return sig.instructions
    except Exception:
        return None


def optimize_doctor_agent(
    cases: List[DoctorTrainingCase],
    model_name: str = "gemini/gemini-3-pro-preview",
    max_metric_calls: int = 20
) -> None:
    
    configure_dspy(model_name)
    
    # Seed the prompt
    original_prompt = read_prompt("simulated_doctor", "system_prompt", "")
    if original_prompt:
        DoctorSignature.__doc__ = original_prompt

    # Define Metric
    critique_agent = DoctorCritiqueAgent(model=model_name)
    
    def trace_metric(example, prediction, ctx):
        trace: SimulationTrace = prediction.trace
        # Find matching case
        case = next((c for c in cases if c.case_id == trace.case_id), None)
        if not case: return 0.0
        
        # Run Critique
        result = critique_agent.critique(trace, case.patient_profile)
        
        # Log for GRPO
        current_instruction = DoctorSignature.__doc__
        log_trace_for_grpo(trace, result, current_instruction)
        
        return result.overall_score

    # Define Training Set
    trainset = []
    for c in cases:
        ex = dspy.Example(case_obj=c).with_inputs("case_obj")
        trainset.append(ex)

    # Optimizer: MIPROv2
    # We optimize the full loop program
    program = DSPyDoctorLoop(model_name=model_name)
    
    # Use MIPROv2 if available, else BootstrapFewShot
    # For 1-day implementation and agent loops, BootstrapFewShot is safer/faster
    # but MIPRO is requested. Let's try MIPROv2 with a low budget.
    
    try:
        # Note: MIPROv2 requires a metric that takes (example, prediction, trace)
        # Our trace_metric matches that signature.
        optimizer = dspy.MIPROv2(
            metric=trace_metric,
            auto="light", # Light optimization mode
            num_candidates=3,
            init_temperature=0.5,
            verbose=True
        )
        
        optimized_program = optimizer.compile(
            program,
            trainset=trainset,
            max_bootstrapped_demos=2,
            max_labeled_demos=2,
            requires_permission_to_run=False,
            minibatch=False # run sequentially to avoid rate limits
        )
        
        # Extract and save the optimized prompt
        # The optimized_program is an instance of DSPyDoctorLoop
        # We want the instruction from its internal doctor_inner module
        best_instr = _extract_instruction(optimized_program.doctor_inner)
        if best_instr:
            print(f"Optimized Instruction: {best_instr}")
            try:
                write_prompt("simulated_doctor", "system_prompt", best_instr)
            except Exception:
                pass
                
    except Exception as e:
        print(f"Optimization failed, falling back to standard execution: {e}")
        # Fallback: Just run once to generate traces
        pass


# --- Simplified Trainer (Non-DSPy Optimizer for now, just Runner) ---

def run_training_rollouts(
    cases: List[DoctorTrainingCase],
    model_name: str,
    num_rollouts: int = 1
):
    """
    Runs simulations to generate 'Golden Traces'.
    Does not actively optimize the prompt in real-time (too slow for 1-day agent loop),
    but collects the data needed for offline optimization/training.
    """
    configure_dspy(model_name) # If we use dspy module
    
    # If num_rollouts > 1, we trigger the optimization loop
    if num_rollouts > 1:
        optimize_doctor_agent(cases, model_name)
        return [] # Traces are logged internally
    
    # Otherwise just run once
    doc_agent = SimulatedDoctorAgent(model=model_name)
    controller = SimulationController()
    judge = DoctorCritiqueAgent(model=model_name)
    
    results = []
    
    for i in range(num_rollouts):
        for case in cases:
            print(f"Running Case {case.case_id} (Rollout {i+1}/{num_rollouts})...")
            
            pat_agent = SimulatedPatientAgent(
                blp=case.blp, 
                patient_profile=case.patient_profile,
                model=model_name
            )
            
            trace = controller.run_simulation(
                doctor_agent=doc_agent,
                patient_agent=pat_agent,
                case_id=case.case_id,
                ground_truth_diagnosis=case.patient_profile.primary_diagnoses,
                max_turns=10
            )
            
            reward = judge.critique(trace, case.patient_profile)
            
            log_trace_for_grpo(trace, reward, doc_agent.system_prompt)
            results.append((trace, reward))
            
    return results
