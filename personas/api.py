from __future__ import annotations

from typing import Dict, List, Any, Optional
from uuid import uuid4
import io
from contextlib import redirect_stdout, redirect_stderr

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import threading

from .blp_extractor import BLPExtractor
from .critique_agent import PersonaCritiqueAgent
from .prompts import read_prompt
from .models import (
    BehavioralLinguisticProfile,
    ConversationConstraints,
    ConversationTurn,
    CritiqueResult,
    PatientProfile,
)
from .patient_profile_builder import PatientProfileBuilder
from .simulated_patient_agent import SimulatedPatientAgent
from .transcript_anonymizer import TranscriptAnonymizer
from .blp_prompt_optimization import (
    BLPLearningExample,
    optimize_blp_prompt,
    OptimizationCancelled,
    configure_dspy,
)
from .config import get_value


app = FastAPI(
    title="Personas API",
    version="0.1.0",
    description=(
        "HTTP API for the persona generation pipeline: transcript → BLP, "
        "case → patient profile, and a simulated-patient agent for training."
    ),
)

# Allow browser clients (e.g., Vite dev server) to call this API.
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

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


class BLPResponse(BaseModel):
    anonymized_transcript: str
    blp: BehavioralLinguisticProfile


class PatientProfileRequest(BaseModel):
    raw_case: str
    model: str | None = None
    max_tokens: int | None = None


class PatientProfileResponse(BaseModel):
    patient_profile: PatientProfile


class SessionInitRequest(BaseModel):
    blp: BehavioralLinguisticProfile
    patient_profile: PatientProfile
    constraints: ConversationConstraints | None = None
    model: str | None = None


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


class CritiqueResponse(BaseModel):
    critique: CritiqueResult


# --- In-memory session store (for now) ----------------------------------------


_SESSIONS: Dict[str, SimulatedPatientAgent] = {}
_JOBS: Dict[str, Dict[str, Any]] = {}


# --- Endpoints ----------------------------------------------------------------


@app.post("/api/blp", response_model=BLPResponse)
def create_blp(payload: BLPRequest) -> BLPResponse:
    """
    Top branch of the architecture:
    Raw transcript → anonymized transcript → BLP.
    """

    if not payload.transcript.strip():
        raise HTTPException(status_code=400, detail="Transcript is empty.")

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

    
    try:
        builder = PatientProfileBuilder(
            model=payload.model or PatientProfileBuilder.model,
            max_tokens=payload.max_tokens if payload.max_tokens is not None else PatientProfileBuilder.max_tokens,
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


@app.post("/api/critique", response_model=CritiqueResponse)
def critique_simulation(payload: CritiqueRequest) -> CritiqueResponse:
    """
    Side-channel evaluation:
    Raw transcript + BLP + raw case + Patient Profile + conversation → critique.
    """

    if not payload.conversation:
        raise HTTPException(status_code=400, detail="Conversation is empty.")

    
    agent = PersonaCritiqueAgent(
        blp=payload.blp,
        patient_profile=payload.patient_profile,
        raw_transcript=payload.raw_transcript,
        raw_case=payload.raw_case,
        model=payload.model or PersonaCritiqueAgent.model,
        max_tokens=payload.max_tokens if payload.max_tokens is not None else PersonaCritiqueAgent.max_tokens,
    )

    try:
        critique = agent.critique(payload.conversation)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Critique generation failed: {e}")
    return CritiqueResponse(critique=critique)


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

    # Run optimization (persona-only aggregation by default)
    
    default_opt_model = get_value("gepa", "optimization_model", "gpt-4.1-mini")
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
            default_opt_model = get_value("gepa", "optimization_model", "gpt-4.1-mini")
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
