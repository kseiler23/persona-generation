from __future__ import annotations

from typing import Dict, List, Any, Optional
from uuid import uuid4
import io
from contextlib import redirect_stdout, redirect_stderr
import os

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import threading
from dotenv import load_dotenv

load_dotenv()

from .blp_extractor import BLPExtractor
from .critique_agent import PersonaCritiqueAgent
from .prompts import read_prompt
from .models import (
    BehavioralLinguisticProfile,
    ConversationConstraints,
    ConversationTurn,
    CritiqueResult,
    PatientProfile,
    SimulationTrace,
    DoctorCritiqueResult,
)
from .patient_profile_builder import PatientProfileBuilder
from .simulated_patient_agent import SimulatedPatientAgent
from .simulated_doctor import SimulatedDoctorAgent
from .simulation_controller import SimulationController
from .doctor_critique import DoctorCritiqueAgent
from .transcript_anonymizer import TranscriptAnonymizer
from .blp_prompt_optimization import (
    BLPLearningExample,
    optimize_blp_prompt,
    OptimizationCancelled,
    configure_dspy,
)
from .config import get_value
from .train_doctor import DoctorTrainingCase, log_trace_for_grpo, run_training_rollouts
from .data_loader import load_training_cases


app = FastAPI(
    title="Personas API",
    version="0.1.0",
    description=(
        "HTTP API for the persona generation pipeline: transcript → BLP, "
        "case → patient profile, and a simulated-patient agent for training."
    ),
)

# Allow browser clients (e.g., Vite dev server) to call this API.
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # includes OPTIONS for preflight
    allow_headers=["*"],
)


# --- Request / response models -------------------------------------------------


class BLPRequest(BaseModel):
    transcript: str
    model: str | None = None
    anonymizer_model: str | None = None
    max_tokens: int | None = None
    api_key: str | None = None


class BLPResponse(BaseModel):
    anonymized_transcript: str
    blp: BehavioralLinguisticProfile


class PatientProfileRequest(BaseModel):
    raw_case: str
    model: str | None = None
    max_tokens: int | None = None
    review_passes: int | None = None
    api_key: str | None = None


class PatientProfileResponse(BaseModel):
    patient_profile: PatientProfile


class SessionInitRequest(BaseModel):
    blp: BehavioralLinguisticProfile
    patient_profile: PatientProfile
    constraints: ConversationConstraints | None = None
    model: str | None = None
    api_key: str | None = None


class SessionInitResponse(BaseModel):
    session_id: str


class TurnRequest(BaseModel):
    doctor_utterance: str


class TurnResponse(BaseModel):
    patient_reply: str


class ResetSessionResponse(BaseModel):
    session_id: str
    status: str


class CritiqueRequest(BaseModel):
    """
    Payload for the critique agent.
    """

    blp: BehavioralLinguisticProfile
    patient_profile: PatientProfile
    raw_transcript: str
    raw_case: str
    conversation: List[ConversationTurn]
    model: str | None = None
    max_tokens: int | None = None
    api_key: str | None = None


class CritiqueResponse(BaseModel):
    critique: CritiqueResult


class DoctorSessionRequest(BaseModel):
    """
    Request to run an automated Doctor vs Patient session.
    """
    blp: BehavioralLinguisticProfile
    patient_profile: PatientProfile
    doctor_model: str | None = None
    patient_model: str | None = None
    max_turns: int = 10
    api_key: str | None = None


class DoctorSessionResponse(BaseModel):
    trace: SimulationTrace
    critique: DoctorCritiqueResult


class TrainDoctorRequest(BaseModel):
    """
    Request to run a batch of simulations to generate training traces.
    """
    cases: List[Dict[str, Any]] | None = None  # List of objects with {raw_case, raw_transcript} or pre-built profiles
    # For simplicity, we'll accept pre-built profiles + blps in a simplified format or just assume 1 case for now
    # To keep payload small, let's accept just one case for the "run" button in UI for now, 
    # or a list of structured objects.
    
    # Let's reuse the single case structure for the "Training" tab UI 
    # which is often "Optimize this current case".
    blp: BehavioralLinguisticProfile | None = None
    patient_profile: PatientProfile | None = None
    case_id: str = "manual_case"

    # New: load from data files
    use_data_files: bool = False
    num_cases: int = 2

    iterations: int = 1
    model: str | None = None
    api_key: str | None = None


class TrainDoctorResponse(BaseModel):
    job_id: str
    status: str


# --- In-memory session store (for now) ----------------------------------------


_SESSIONS: Dict[str, SimulatedPatientAgent] = {}
_JOBS: Dict[str, Dict[str, Any]] = {}


def prepare_cases_from_payload(
    cases_payload: List[Dict[str, Any]],
    fallback_blp: BehavioralLinguisticProfile,
    fallback_profile: PatientProfile,
) -> List[DoctorTrainingCase]:
    """
    Build DoctorTrainingCase objects from a payload list.
    Falls back to the provided BLP/patient profile when items are partial.
    """
    cases: List[DoctorTrainingCase] = []
    for idx, case in enumerate(cases_payload):
        if not isinstance(case, dict):
            continue
        blp = case.get("blp") or fallback_blp
        patient_profile = case.get("patient_profile") or fallback_profile
        case_id = str(
            case.get("case_id")
            or getattr(patient_profile, "id", None)
            or f"case_{idx}"
        )
        cases.append(
            DoctorTrainingCase(
                case_id=case_id,
                patient_profile=patient_profile,
                blp=blp,
            )
        )
    return cases


# --- Endpoints ----------------------------------------------------------------


@app.post("/api/blp", response_model=BLPResponse)
def create_blp(payload: BLPRequest) -> BLPResponse:
    """
    Top branch of the architecture:
    Raw transcript → anonymized transcript → BLP.
    """

    if not payload.transcript.strip():
        raise HTTPException(status_code=400, detail="Transcript is empty.")

    if payload.api_key:
        os.environ["OPENAI_API_KEY"] = payload.api_key
        os.environ["GEMINI_API_KEY"] = payload.api_key
        os.environ["GOOGLE_API_KEY"] = payload.api_key

    anonymizer_model = payload.anonymizer_model or payload.model or TranscriptAnonymizer.model
    extractor_model = payload.model or BLPExtractor.model
    extractor_max_tokens = payload.max_tokens if payload.max_tokens is not None else BLPExtractor.max_tokens

    try:
        anonymizer = TranscriptAnonymizer(model=anonymizer_model)
        extractor = BLPExtractor(model=extractor_model, max_tokens=extractor_max_tokens)

        anonymized = anonymizer.anonymize(payload.transcript)
        blp = extractor.extract(anonymized)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"BLP generation failed: {e}")
    return BLPResponse(anonymized_transcript=anonymized, blp=blp)


@app.post("/api/patient-profile", response_model=PatientProfileResponse)
def create_patient_profile(payload: PatientProfileRequest) -> PatientProfileResponse:
    """
    Bottom branch:
    Raw case (structured + unstructured) → Patient Profile.
    """

    if not payload.raw_case.strip():
        raise HTTPException(status_code=400, detail="Raw case is empty.")

    if payload.api_key:
        os.environ["OPENAI_API_KEY"] = payload.api_key
        os.environ["GEMINI_API_KEY"] = payload.api_key
        os.environ["GOOGLE_API_KEY"] = payload.api_key

    
    review_passes = (
        int(payload.review_passes)
        if payload.review_passes is not None
        else PatientProfileBuilder.review_passes
    )

    try:
        builder = PatientProfileBuilder(
            model=payload.model or PatientProfileBuilder.model,
            max_tokens=payload.max_tokens if payload.max_tokens is not None else PatientProfileBuilder.max_tokens,
            review_passes=review_passes,
        )
        profile = builder.build_from_case(payload.raw_case)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Patient profile generation failed: {e}")
    return PatientProfileResponse(patient_profile=profile)


@app.post("/api/simulated-patient/session", response_model=SessionInitResponse)
def init_session(payload: SessionInitRequest) -> SessionInitResponse:
    """
    Initialize a simulated patient session from BLP + Patient Profile (+ optional constraints).
    Returns a session_id used to drive a multi-turn interaction.
    """

    if payload.api_key:
        os.environ["OPENAI_API_KEY"] = payload.api_key
        os.environ["GEMINI_API_KEY"] = payload.api_key
        os.environ["GOOGLE_API_KEY"] = payload.api_key

    session_id = str(uuid4())
    
    agent = SimulatedPatientAgent(
        blp=payload.blp,
        patient_profile=payload.patient_profile,
        constraints=payload.constraints or ConversationConstraints.default_for_training(),
        model=payload.model or SimulatedPatientAgent.model,
    )
    _SESSIONS[session_id] = agent
    return SessionInitResponse(session_id=session_id)


@app.post("/api/simulated-patient/{session_id}/turn", response_model=TurnResponse)
def send_turn(session_id: str, payload: TurnRequest) -> TurnResponse:
    """
    Send a single clinician utterance and get the simulated patient's reply.
    """

    agent = _SESSIONS.get(session_id)
    if agent is None:
        raise HTTPException(status_code=404, detail="Session not found.")

    
    try:
        reply = agent.reply(payload.doctor_utterance)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Simulated patient reply failed: {e}")
    return TurnResponse(patient_reply=reply)


@app.post("/api/simulated-patient/{session_id}/reset", response_model=ResetSessionResponse)
def reset_simulated_patient_session(session_id: str) -> ResetSessionResponse:
    """
    Clear the multi-turn history for a simulated patient session (reset chat).
    """
    agent = _SESSIONS.get(session_id)
    if agent is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    agent.reset()
    return ResetSessionResponse(session_id=session_id, status="reset")


import sys
import traceback

@app.post("/api/critique", response_model=CritiqueResponse)
def critique_simulation(payload: CritiqueRequest) -> CritiqueResponse:
    """
    Side-channel evaluation:
    Raw transcript + BLP + raw case + Patient Profile + conversation → critique.
    """

    if not payload.conversation:
        raise HTTPException(status_code=400, detail="Conversation is empty.")

    if payload.api_key:
        os.environ["OPENAI_API_KEY"] = payload.api_key
        os.environ["GEMINI_API_KEY"] = payload.api_key
        os.environ["GOOGLE_API_KEY"] = payload.api_key
        # litellm.api_key = payload.api_key # removed this line to fix network error caused by potential race conditions or library behavior

    agent = PersonaCritiqueAgent(
        blp=payload.blp,
        patient_profile=payload.patient_profile,
        raw_transcript=payload.raw_transcript,
        raw_case=payload.raw_case,
        model=payload.model or PersonaCritiqueAgent.model,
        max_tokens=payload.max_tokens if payload.max_tokens is not None else PersonaCritiqueAgent.max_tokens,
        api_key=payload.api_key,
    )

    try:
        critique = agent.critique(payload.conversation)
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        raise HTTPException(status_code=502, detail=f"Critique generation failed: {e}")
    return CritiqueResponse(critique=critique)


@app.post("/api/simulate/doctor-session", response_model=DoctorSessionResponse)
def simulate_doctor_session(payload: DoctorSessionRequest) -> DoctorSessionResponse:
    """
    Run a fully automated session between the Simulated Doctor and Simulated Patient.
    Returns the simulation trace and the Doctor's critique score.
    """
    if payload.api_key:
        os.environ["OPENAI_API_KEY"] = payload.api_key
        os.environ["GEMINI_API_KEY"] = payload.api_key
        os.environ["GOOGLE_API_KEY"] = payload.api_key

    try:
        # 1. Init Agents
        doc_agent = SimulatedDoctorAgent(
            model=payload.doctor_model or SimulatedDoctorAgent.model
        )
        pat_agent = SimulatedPatientAgent(
            blp=payload.blp,
            patient_profile=payload.patient_profile,
            model=payload.patient_model or SimulatedPatientAgent.model
        )
        
        # 2. Run Controller
        controller = SimulationController()
        trace = controller.run_simulation(
            doctor_agent=doc_agent,
            patient_agent=pat_agent,
            case_id=payload.patient_profile.id,
            ground_truth_diagnosis=payload.patient_profile.primary_diagnoses,
            max_turns=payload.max_turns
        )
        
        # 3. Run Judge
        judge = DoctorCritiqueAgent(
            model=payload.doctor_model or "gemini/gemini-3-pro-preview" 
        )
        critique = judge.critique(trace, payload.patient_profile)
        
        return DoctorSessionResponse(trace=trace, critique=critique)
        
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Doctor simulation failed: {e}")


@app.post("/api/train/doctor", response_model=TrainDoctorResponse)
def train_doctor_job(payload: TrainDoctorRequest, background_tasks: BackgroundTasks) -> TrainDoctorResponse:
    """
    Start a background job to run simulations (rollouts) and log traces for training.
    """
    job_id = str(uuid4())

    # Store job status
    _JOBS[job_id] = {
        "status": "running",
        "progress": {"percent": 0, "rollouts": 0, "total": payload.iterations},
        "result": None
    }

    def _run_task():
        try:
            if payload.api_key:
                os.environ["OPENAI_API_KEY"] = payload.api_key
                os.environ["GEMINI_API_KEY"] = payload.api_key
                os.environ["GOOGLE_API_KEY"] = payload.api_key

            model_name = payload.model or "gemini/gemini-3-pro-preview"

            # Load cases from data files or payload
            if payload.use_data_files:
                cases = load_training_cases(
                    num_cases=payload.num_cases,
                    model=model_name
                )
            else:
                cases = prepare_cases_from_payload(payload.cases or [], payload.blp, payload.patient_profile)
                if not cases and payload.blp and payload.patient_profile:
                    cases = [
                        DoctorTrainingCase(
                            case_id=payload.case_id,
                            patient_profile=payload.patient_profile,
                            blp=payload.blp,
                        )
                    ]

            run_training_rollouts(
                cases=cases,
                model_name=model_name,
                num_rollouts=payload.iterations,
            )

            _JOBS[job_id]["status"] = "complete"
            _JOBS[job_id]["progress"]["percent"] = 100

        except Exception as e:
            _JOBS[job_id]["status"] = "error"
            _JOBS[job_id]["error"] = str(e)

    background_tasks.add_task(_run_task)

    return TrainDoctorResponse(job_id=job_id, status="started")


class TrainFromDataRequest(BaseModel):
    """
    Request to train from data files (parquet + transcripts).
    """
    num_cases: int = 10
    iterations: int = 2
    model: str | None = None
    api_key: str | None = None


@app.post("/api/train/doctor/from-data", response_model=TrainDoctorResponse)
def train_doctor_from_data(payload: TrainFromDataRequest, background_tasks: BackgroundTasks) -> TrainDoctorResponse:
    """
    Start a background training job using cases loaded from data files.
    This is a simplified endpoint that automatically loads clinical cases from
    data/clinical_cases/cases.parquet and transcripts from data/transcripts/.
    """
    job_id = str(uuid4())

    _JOBS[job_id] = {
        "status": "running",
        "progress": {"percent": 0, "rollouts": 0, "total": payload.iterations},
        "result": None
    }

    def _run_task():
        try:
            if payload.api_key:
                os.environ["OPENAI_API_KEY"] = payload.api_key
                os.environ["GEMINI_API_KEY"] = payload.api_key
                os.environ["GOOGLE_API_KEY"] = payload.api_key

            model_name = payload.model or "gemini/gemini-3-pro-preview"

            print(f"[DEBUG] Loading {payload.num_cases} cases from data files...")
            cases = load_training_cases(
                num_cases=payload.num_cases,
                model=model_name
            )
            print(f"[DEBUG] Loaded {len(cases)} cases")

            print(f"[DEBUG] Starting training rollouts with {payload.iterations} iterations...")
            run_training_rollouts(
                cases=cases,
                model_name=model_name,
                num_rollouts=payload.iterations,
            )

            print(f"[DEBUG] Training complete!")
            _JOBS[job_id]["status"] = "complete"
            _JOBS[job_id]["progress"]["percent"] = 100

        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            print(f"[ERROR] Training job failed: {error_msg}")
            _JOBS[job_id]["status"] = "error"
            _JOBS[job_id]["error"] = error_msg

    background_tasks.add_task(_run_task)

    return TrainDoctorResponse(job_id=job_id, status="started")


# --- GEPA optimization endpoint ------------------------------------------------


class OptimizeBLPRequest(BaseModel):
    """
    Trigger a GEPA optimization run from the current conversation context.
    Uses the conversation's doctor turns as the probe script.
    """

    raw_transcript: str
    raw_case: str
    conversation: List[ConversationTurn]
    # Optional GEPA budget knobs (sane defaults if not provided)
    reflection_minibatch_size: int | None = None
    candidate_selection_strategy: str | None = None
    max_metric_calls: int | None = None
    use_merge: bool | None = None
    track_stats: bool | None = None
    model: str | None = None
    api_key: str | None = None


class OptimizeBLPResponse(BaseModel):
    original_prompt: str
    optimized_prompt: str


class OptimizeStartResponse(BaseModel):
    job_id: str


class OptimizeProgressResponse(BaseModel):
    status: str
    percent: int
    metric_calls: Optional[int] = None
    max_metric_calls: Optional[int] = None
    latest_score: Optional[float] = None
    best_score: Optional[float] = None
    message: Optional[str] = None


class OptimizeCancelResponse(BaseModel):
    job_id: str
    status: str
    message: Optional[str] = None


@app.post("/api/optimize/blp-prompt", response_model=OptimizeBLPResponse)
def optimize_blp_prompt_endpoint(payload: OptimizeBLPRequest) -> OptimizeBLPResponse:
    """
    Run GEPA to optimize the BLP extraction instruction using the supplied
    raw transcript, raw case, and the doctor-side of the provided conversation.
    Returns the original vs optimized instruction text. Also persists the
    optimized prompt back into prompts.yaml for runtime use.
    """

    if not payload.raw_transcript.strip():
        raise HTTPException(status_code=400, detail="raw_transcript is required.")
    if not payload.raw_case.strip():
        raise HTTPException(status_code=400, detail="raw_case is required.")
    if not payload.conversation:
        raise HTTPException(status_code=400, detail="conversation is required.")

    # Build a probe script using doctor utterances from the conversation
    doctor_script: List[str] = [
        turn.content for turn in payload.conversation if turn.role == "doctor"
    ]
    if not doctor_script:
        raise HTTPException(
            status_code=400,
            detail="Conversation must include at least one doctor utterance.",
        )

    
    original_prompt = read_prompt("blp_extraction", "system_prompt", "")

    example = BLPLearningExample(
        raw_transcript=payload.raw_transcript,
        raw_case=payload.raw_case,
        doctor_script=doctor_script,
    )

    # Prepare GEPA budget knobs (use defaults if None)
    kwargs = dict(
        reflection_minibatch_size=payload.reflection_minibatch_size
        if payload.reflection_minibatch_size is not None
        else 3,
        candidate_selection_strategy=payload.candidate_selection_strategy
        if payload.candidate_selection_strategy is not None
        else "pareto",
        max_metric_calls=payload.max_metric_calls
        if payload.max_metric_calls is not None
        else 200,
        use_merge=True if payload.use_merge is None else payload.use_merge,
        track_stats=True if payload.track_stats is None else payload.track_stats,
    )

    if payload.api_key:
        os.environ["OPENAI_API_KEY"] = payload.api_key
        os.environ["GEMINI_API_KEY"] = payload.api_key
        os.environ["GOOGLE_API_KEY"] = payload.api_key

    # Run optimization (persona-only aggregation by default)
    default_opt_model = get_value("gepa", "optimization_model", "gemini/gemini-3-pro-preview")
    model_for_opt = payload.model or default_opt_model
    # Preconfigure DSPy in this thread; ignore if already configured elsewhere
    try:
        configure_dspy(model_for_opt)
    except Exception:
        pass
    try:
        # Suppress library prints during GEPA runs
        _sink_out, _sink_err = io.StringIO(), io.StringIO()
        with redirect_stdout(_sink_out), redirect_stderr(_sink_err):
            optimize_blp_prompt(
                examples=[example],
                model_name=model_for_opt,
                w_clinical=0.0,
                w_persona=1.0,
                skip_configure=True,
                **kwargs,
            )
    except OptimizationCancelled as e:
        raise HTTPException(status_code=400, detail=str(e) or "Optimization cancelled.")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"BLP prompt optimization failed: {e}")

    optimized_prompt = read_prompt("blp_extraction", "system_prompt", "")
    
    return OptimizeBLPResponse(
        original_prompt=original_prompt,
        optimized_prompt=optimized_prompt,
    )


@app.post("/api/optimize/blp-prompt/start", response_model=OptimizeStartResponse)
def optimize_blp_prompt_start(payload: OptimizeBLPRequest) -> OptimizeStartResponse:
    """
    Start a background GEPA optimization job and return a job_id.
    Progress can be polled via /progress/{job_id}, and the final result via /result/{job_id}.
    """
    if not payload.raw_transcript.strip():
        raise HTTPException(status_code=400, detail="raw_transcript is required.")
    if not payload.raw_case.strip():
        raise HTTPException(status_code=400, detail="raw_case is required.")
    if not payload.conversation:
        raise HTTPException(status_code=400, detail="conversation is required.")

    doctor_script: List[str] = [turn.content for turn in payload.conversation if turn.role == "doctor"]
    if not doctor_script:
        raise HTTPException(status_code=400, detail="Conversation must include at least one doctor utterance.")

    if payload.api_key:
        os.environ["OPENAI_API_KEY"] = payload.api_key
        os.environ["GEMINI_API_KEY"] = payload.api_key
        os.environ["GOOGLE_API_KEY"] = payload.api_key

    job_id = str(uuid4())
    original_prompt = read_prompt("blp_extraction", "system_prompt", "")
    # Prepare GEPA budget knobs (use defaults if None)
    kwargs = dict(
        reflection_minibatch_size=payload.reflection_minibatch_size if payload.reflection_minibatch_size is not None else 3,
        candidate_selection_strategy=payload.candidate_selection_strategy if payload.candidate_selection_strategy is not None else "pareto",
        max_metric_calls=payload.max_metric_calls if payload.max_metric_calls is not None else 200,
        use_merge=True if payload.use_merge is None else payload.use_merge,
        track_stats=True if payload.track_stats is None else payload.track_stats,
    )
    _JOBS[job_id] = {
        "status": "queued",
        "progress": {"percent": 0, "metric_calls": 0, "max_metric_calls": kwargs["max_metric_calls"], "latest_score": None, "best_score": None},
        "original_prompt": original_prompt,
        "result": None,
        "error": None,
        "cancel": False,
    }

    example = BLPLearningExample(
        raw_transcript=payload.raw_transcript,
        raw_case=payload.raw_case,
        doctor_script=doctor_script,
    )

    def _progress_cb(update: Dict[str, Any]) -> None:
        job = _JOBS.get(job_id)
        if job is None:
            return
        prog = job.get("progress", {})
        prog.update(
            {
                "percent": int(update.get("percent", prog.get("percent", 0)) or 0),
                "metric_calls": update.get("metric_calls", prog.get("metric_calls")),
                "max_metric_calls": update.get("max_metric_calls", prog.get("max_metric_calls")),
                "latest_score": update.get("latest_score", prog.get("latest_score")),
                "best_score": update.get("best_score", prog.get("best_score")),
            }
        )
        job["progress"] = prog
        job["status"] = update.get("status", job.get("status", "running"))

    def _run():
        try:
            _JOBS[job_id]["status"] = "running"
            default_opt_model = get_value("gepa", "optimization_model", "gemini/gemini-3-pro-preview")
            model_for_opt = payload.model or default_opt_model
            def _should_cancel() -> bool:
                job = _JOBS.get(job_id)
                return bool(job and job.get("cancel"))
            # Suppress library prints during GEPA runs
            _sink_out, _sink_err = io.StringIO(), io.StringIO()
            with redirect_stdout(_sink_out), redirect_stderr(_sink_err):
                optimize_blp_prompt(
                    examples=[example],
                    model_name=model_for_opt,
                    w_clinical=0.0,
                    w_persona=1.0,
                    progress_callback=_progress_cb,
                    should_cancel=_should_cancel,
                    skip_configure=True,
                    **kwargs,
                )
            optimized_prompt = read_prompt("blp_extraction", "system_prompt", "")
            _JOBS[job_id]["result"] = {
                "original_prompt": original_prompt,
                "optimized_prompt": optimized_prompt,
            }
            _JOBS[job_id]["status"] = "complete"
        except OptimizationCancelled:
            _JOBS[job_id]["status"] = "cancelled"
            _JOBS[job_id]["error"] = None
        except Exception as e:
            _JOBS[job_id]["status"] = "error"
            _JOBS[job_id]["error"] = str(e)

    threading.Thread(target=_run, daemon=True).start()
    return OptimizeStartResponse(job_id=job_id)


@app.get("/api/optimize/blp-prompt/progress/{job_id}", response_model=OptimizeProgressResponse)
def optimize_blp_prompt_progress(job_id: str) -> OptimizeProgressResponse:
    job = _JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    prog = job.get("progress", {})
    return OptimizeProgressResponse(
        status=str(job.get("status", "unknown")),
        percent=int(prog.get("percent", 0) or 0),
        metric_calls=prog.get("metric_calls"),
        max_metric_calls=prog.get("max_metric_calls"),
        latest_score=prog.get("latest_score"),
        best_score=prog.get("best_score"),
        message=job.get("error"),
    )


@app.get("/api/optimize/blp-prompt/result/{job_id}", response_model=OptimizeBLPResponse)
def optimize_blp_prompt_result(job_id: str) -> OptimizeBLPResponse:
    job = _JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    if job.get("status") != "complete" or not job.get("result"):
        raise HTTPException(status_code=202, detail="job not complete")
    result = job["result"]
    return OptimizeBLPResponse(
        original_prompt=result["original_prompt"],
        optimized_prompt=result["optimized_prompt"],
    )


@app.post("/api/optimize/blp-prompt/cancel/{job_id}", response_model=OptimizeCancelResponse)
def optimize_blp_prompt_cancel(job_id: str) -> OptimizeCancelResponse:
    """
    Request cancellation of a running GEPA optimization job.
    """
    job = _JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    job["cancel"] = True
    # Surface an intermediate status so the UI can show "cancelling"
    if job.get("status") == "running":
        job["status"] = "cancelling"
    return OptimizeCancelResponse(job_id=job_id, status=str(job.get("status", "unknown")))
