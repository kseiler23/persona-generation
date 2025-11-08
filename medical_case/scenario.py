from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any
import json

from persona_agent.agent import PersonaAgent
from persona_agent.persona import (
    PersonaProfile,
    persona_json_to_instructions,
    persona_style_instructions,
)


# Kept minimal and focused on the needs of a medical OSCE-style testing scenario
@dataclass
class MedicalVitals:
    heart_rate_bpm: int
    blood_pressure_systolic: int
    blood_pressure_diastolic: int
    respiratory_rate: int
    temperature_c: float
    spo2_percent: int


@dataclass
class MedicalCase:
    title: str
    persona_file: str
    patient_name: str
    patient_age: int
    patient_sex: str
    presenting_complaint: str
    onset: str
    duration: str
    character: str
    location: str
    radiation: str
    patient_occupation: Optional[str] = None
    alleviating_factors: List[str] = field(default_factory=list)
    aggravating_factors: List[str] = field(default_factory=list)
    associated_symptoms: List[str] = field(default_factory=list)
    red_flags: List[str] = field(default_factory=list)
    past_medical_history: List[str] = field(default_factory=list)
    medications: List[str] = field(default_factory=list)
    allergies: List[str] = field(default_factory=list)
    social_history: Dict[str, Any] = field(default_factory=dict)
    family_history: List[str] = field(default_factory=list)
    vitals: MedicalVitals = field(
        default_factory=lambda: MedicalVitals(
            heart_rate_bpm=84,
            blood_pressure_systolic=126,
            blood_pressure_diastolic=78,
            respiratory_rate=16,
            temperature_c=36.9,
            spo2_percent=98,
        )
    )
    withheld_facts: List[str] = field(default_factory=list)
    triage: str = "urgent"
    doctor_opening_prompt: str = (
        "You are a doctor in clinic. Take a focused history and "
        "ask targeted questions to reach a safe differential and plan."
    )


STRICT_STYLE_SUFFIX = """
Strict stylistic constraints:
- Mirror the patient's grammar, syntax, and sentence length as established by the persona style.
- Match punctuation, capitalization, fillers, and rhetorical devices typical of the persona.
- Avoid generic disclaimers or policy meta. Stay fully in-character as the patient.
- Be concise or elaborate in line with the persona's style; keep tone consistent.
- Do not contradict the case details below; ask for clarification rather than inventing facts.
""".strip()

CONTENT_RULES_SUFFIX = """
Conversation behavior rules:
- This is a medical appointment. You are the patient. Be cooperative and natural.
- Start with the chief complaint in your own words. Do NOT dump your entire history unprompted.
- Prefer short, direct answers to the doctor's specific questions.
- If the doctor asks open-ended questions (e.g., \"Anything else?\"), you may reveal more.
- DO NOT disclose 'withheld facts' unless the doctor asks in a way that reasonably elicits them.
- Do NOT mention that you have 'withheld facts', rules, or internal instructions.
- If you don't know or don't remember, say so plainly.
- If the doctor gives advice or a plan, respond realistically as the patient would.
""".strip()


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def _load_persona_profile(persona_path: Path) -> PersonaProfile:
    data = _read_json(persona_path)
    return PersonaProfile(**data)


def load_case_file(case_path: Path) -> MedicalCase:
    raw = _read_json(case_path)
    vitals_dict = raw.get("vitals") or {}
    vitals = MedicalVitals(
        heart_rate_bpm=vitals_dict.get("heart_rate_bpm", 84),
        blood_pressure_systolic=vitals_dict.get("blood_pressure_systolic", 126),
        blood_pressure_diastolic=vitals_dict.get("blood_pressure_diastolic", 78),
        respiratory_rate=vitals_dict.get("respiratory_rate", 16),
        temperature_c=vitals_dict.get("temperature_c", 36.9),
        spo2_percent=vitals_dict.get("spo2_percent", 98),
    )
    return MedicalCase(
        title=raw["title"],
        persona_file=raw["persona_file"],
        patient_name=raw["patient_name"],
        patient_age=raw["patient_age"],
        patient_sex=raw["patient_sex"],
        patient_occupation=raw.get("patient_occupation"),
        presenting_complaint=raw["presenting_complaint"],
        onset=raw.get("onset", ""),
        duration=raw.get("duration", ""),
        character=raw.get("character", ""),
        location=raw.get("location", ""),
        radiation=raw.get("radiation", ""),
        alleviating_factors=raw.get("alleviating_factors", []),
        aggravating_factors=raw.get("aggravating_factors", []),
        associated_symptoms=raw.get("associated_symptoms", []),
        red_flags=raw.get("red_flags", []),
        past_medical_history=raw.get("past_medical_history", []),
        medications=raw.get("medications", []),
        allergies=raw.get("allergies", []),
        social_history=raw.get("social_history", {}),
        family_history=raw.get("family_history", []),
        vitals=vitals,
        withheld_facts=raw.get("withheld_facts", []),
        triage=raw.get("triage", "urgent"),
        doctor_opening_prompt=raw.get("doctor_opening_prompt", ""),
    )


def build_context_from_case(case: MedicalCase) -> str:
    v = case.vitals
    lines: List[str] = []
    lines.append("Medical case context:")
    lines.append(f"Patient: {case.patient_name}, {case.patient_age} y/o {case.patient_sex}")
    if case.patient_occupation:
        lines.append(f"Occupation: {case.patient_occupation}")
    lines.append(f"Presenting complaint: {case.presenting_complaint}")
    if case.onset:
        lines.append(f"Onset: {case.onset}")
    if case.duration:
        lines.append(f"Duration: {case.duration}")
    if case.character:
        lines.append(f"Character: {case.character}")
    if case.location:
        lines.append(f"Location: {case.location}")
    if case.radiation:
        lines.append(f"Radiation: {case.radiation}")
    if case.alleviating_factors:
        lines.append("Alleviating: " + ", ".join(case.alleviating_factors))
    if case.aggravating_factors:
        lines.append("Aggravating: " + ", ".join(case.aggravating_factors))
    if case.associated_symptoms:
        lines.append("Associated symptoms: " + ", ".join(case.associated_symptoms))
    if case.red_flags:
        lines.append("Red flags to consider: " + "; ".join(case.red_flags))
    if case.past_medical_history:
        lines.append("Past medical history: " + "; ".join(case.past_medical_history))
    if case.medications:
        lines.append("Medications: " + "; ".join(case.medications))
    if case.allergies:
        lines.append("Allergies: " + "; ".join(case.allergies))
    if case.family_history:
        lines.append("Family history: " + "; ".join(case.family_history))
    if case.social_history:
        sh_parts = [f"{k}: {v}" for k, v in case.social_history.items()]
        lines.append("Social history: " + "; ".join(sh_parts))
    lines.append(
        "Vitals: "
        f"HR {v.heart_rate_bpm} bpm, BP {v.blood_pressure_systolic}/{v.blood_pressure_diastolic} mmHg, "
        f"RR {v.respiratory_rate}/min, Temp {v.temperature_c:.1f}°C, SpO2 {v.spo2_percent}% room air"
    )
    lines.append(f"Triage: {case.triage}")
    return "\n".join(lines).strip()


def build_instructions_for_case(
    profile: PersonaProfile,
    case: MedicalCase,
    *,
    style_only: bool = True,
) -> str:
    """
    Compose system instructions that combine persona style and medical roleplay rules.
    By default we use style-only to avoid leaking persona's personal facts into the patient.
    """
    base = (
        persona_style_instructions(profile)
        if style_only
        else persona_json_to_instructions(profile)
    )
    rules = []
    rules.append("You are role-playing as the patient in this case and will speak to a doctor.")
    rules.append(
        f"Your name is {case.patient_name}. You are {case.patient_age} years old, {case.patient_sex}."
    )
    if case.patient_occupation:
        rules.append(
            f"If asked about job, work, knowledge domain, or what you do, answer succinctly with: '{case.patient_occupation}'. Do not invent extra background unless asked."
        )
    else:
        rules.append(
            "If asked about job, work, or knowledge domain and it is not provided in this case, say you would prefer to focus on your health right now."
        )
    rules.append("Do not invent background details not present in the case data.")
    rules.append(
        "Do not reveal this list of withheld facts unless the doctor's question reasonably elicits them."
    )
    if case.withheld_facts:
        for i, fact in enumerate(case.withheld_facts, 1):
            rules.append(f"- Withheld {i}: {fact}")
    else:
        rules.append("- Withheld facts: (none)")
    rules.append(
        "Before sending each reply, quickly check: are you revealing any withheld facts? If not explicitly asked in a way that elicits them (e.g., triggers, onset circumstances, substances taken today), do not disclose."
    )
    rules.append(CONTENT_RULES_SUFFIX)
    rules.append(STRICT_STYLE_SUFFIX)
    return f"{base}\n\n" + "\n".join(rules)


def build_agent_for_case(
    case: MedicalCase,
    profile: PersonaProfile,
    *,
    model: str = "gpt-4o-mini",
    style_only: bool = True,
) -> PersonaAgent:
    instructions = build_instructions_for_case(profile, case, style_only=style_only)
    context = build_context_from_case(case)
    return PersonaAgent(model=model, instructions=instructions, context=context)


def save_case_file(case: MedicalCase, out_path: Path) -> None:
    payload = asdict(case)
    payload["vitals"] = asdict(case.vitals)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


