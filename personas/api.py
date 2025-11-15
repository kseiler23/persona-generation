from __future__ import annotations

from typing import Dict, List
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .blp_extractor import BLPExtractor
from .critique_agent import PersonaCritiqueAgent
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


class BLPResponse(BaseModel):
    anonymized_transcript: str
    blp: BehavioralLinguisticProfile


class PatientProfileRequest(BaseModel):
    raw_case: str
    model: str | None = None


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


class CritiqueRequest(BaseModel):
    """
    Payload for the critique agent.
    """

    blp: BehavioralLinguisticProfile
    patient_profile: PatientProfile
    raw_transcript: str
    raw_case: str
    conversation: List[ConversationTurn]


class CritiqueResponse(BaseModel):
    critique: CritiqueResult


# --- In-memory session store (for now) ----------------------------------------


_SESSIONS: Dict[str, SimulatedPatientAgent] = {}


# --- Endpoints ----------------------------------------------------------------


@app.post("/api/blp", response_model=BLPResponse)
def create_blp(payload: BLPRequest) -> BLPResponse:
    """
    Top branch of the architecture:
    Raw transcript → anonymized transcript → BLP.
    """

    if not payload.transcript.strip():
        raise HTTPException(status_code=400, detail="Transcript is empty.")

    anonymizer = TranscriptAnonymizer(model=payload.model or TranscriptAnonymizer.model)
    extractor = BLPExtractor(model=payload.model or BLPExtractor.model)

    anonymized = anonymizer.anonymize(payload.transcript)
    blp = extractor.extract(anonymized)
    return BLPResponse(anonymized_transcript=anonymized, blp=blp)


@app.post("/api/patient-profile", response_model=PatientProfileResponse)
def create_patient_profile(payload: PatientProfileRequest) -> PatientProfileResponse:
    """
    Bottom branch:
    Raw case (structured + unstructured) → Patient Profile.
    """

    if not payload.raw_case.strip():
        raise HTTPException(status_code=400, detail="Raw case is empty.")

    builder = PatientProfileBuilder(model=payload.model or PatientProfileBuilder.model)
    profile = builder.build_from_case(payload.raw_case)
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

    reply = agent.reply(payload.doctor_utterance)
    return TurnResponse(patient_reply=reply)


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
    )

    critique = agent.critique(payload.conversation)
    return CritiqueResponse(critique=critique)



