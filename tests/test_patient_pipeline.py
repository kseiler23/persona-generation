import json
import os
from pathlib import Path

import pytest

from personas.blp_extractor import BLPExtractor
from personas.patient_profile_builder import PatientProfileBuilder
from personas.simulated_patient_agent import SimulatedPatientAgent
from personas.transcript_anonymizer import TranscriptAnonymizer
from personas.judge import ConversationJudge
from personas.models import ConversationTurn


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "patient_case.json"
_HAS_KEY = bool(os.environ.get("OPENAI_API_KEY") or os.environ.get("GEMINI_API_KEY"))


@pytest.mark.skipif(
    not _HAS_KEY, reason="Requires LLM API key (OPENAI_API_KEY or GEMINI_API_KEY)"
)
def test_pipeline_end_to_end_smoke() -> None:
    fixture = json.loads(FIXTURE_PATH.read_text())
    raw_transcript = fixture["raw_transcript"]
    raw_case = fixture["raw_case"]
    doctor_script = fixture["doctor_script"]

    # 1) Anonymize transcript -> BLP
    anonymized = TranscriptAnonymizer().anonymize(raw_transcript)
    blp = BLPExtractor().extract(anonymized)

    # 2) Build patient profile with one review pass for coverage
    profile_builder = PatientProfileBuilder(review_passes=1)
    patient_profile = profile_builder.build_from_case(raw_case)

    # 3) Simulate conversation
    agent = SimulatedPatientAgent(blp=blp, patient_profile=patient_profile)
    conversation: list[ConversationTurn] = []
    for q in doctor_script:
        reply = agent.reply(q)
        conversation.append(ConversationTurn(role="doctor", content=q))
        conversation.append(ConversationTurn(role="patient", content=reply))

    # 4) Judge the interaction for rule violations
    judge = ConversationJudge()
    report = judge.evaluate(
        conversation=conversation,
        raw_case=raw_case,
        blp=blp,
        patient_profile=patient_profile,
    )

    assert "total_errors" in report
    assert isinstance(report["total_errors"], int)
    assert report["total_errors"] >= 0
    assert "per_category" in report
