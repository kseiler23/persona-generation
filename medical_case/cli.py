import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel

from .scenario import (
    MedicalCase,
    MedicalVitals,
    build_agent_for_case,
    load_case_file,
    save_case_file,
)
from persona_agent.persona import PersonaProfile


app = typer.Typer(help="Medical Case scenario CLI")
console = Console()


DEFAULT_PERSONA_PATH = Path("data/personas/Speaker_1.json")
DEFAULT_CASE_OUT = Path("medical_case/cases/speaker1_chest_pain.json")


def _load_persona(persona_path: Path) -> PersonaProfile:
    data = json.loads(persona_path.read_text())
    return PersonaProfile(**data)


@app.command("create")
def create(
    persona_file: str = typer.Option(
        str(DEFAULT_PERSONA_PATH), help="Path to a persona JSON file."
    ),
    out_path: str = typer.Option(
        str(DEFAULT_CASE_OUT), help="Where to write the generated case JSON."
    ),
):
    """
    Create a ready-to-use test medical case for the given persona.
    """
    p = Path(persona_file).expanduser().resolve()
    if not p.exists():
        raise typer.BadParameter(f"persona_file not found: {p}")

    # Sample case: Chest pain with realistic withheld facts
    case = MedicalCase(
        title="Chest pain in clinic (Speaker_1 style)",
        persona_file=str(p),
        patient_name="Kirk",
        patient_age=45,
        patient_sex="male",
        patient_occupation="Professional negotiator and trainer",
        presenting_complaint="Chest pain",
        onset="Started 2 hours ago at rest",
        duration="Intermittent episodes, each ~10 minutes",
        character="Pressure-like tightness, heavy",
        location="Central chest, retrosternal",
        radiation="Sometimes to left arm and jaw",
        alleviating_factors=["Sitting still", "Resting"],
        aggravating_factors=["Walking upstairs", "Stress"],
        associated_symptoms=["Mild shortness of breath", "Nausea", "Lightheadedness"],
        red_flags=["Exertional chest pain", "Radiation to arm/jaw", "Nausea/diaphoresis"],
        past_medical_history=["Borderline high cholesterol"],
        medications=["Multivitamin daily"],
        allergies=["No known drug allergies"],
        social_history={"smoking": "Occasional cigar (1-2/month)", "alcohol": "Social", "exercise": "Irregular"},
        family_history=["Father had MI at 54"],
        vitals=MedicalVitals(
            heart_rate_bpm=96,
            blood_pressure_systolic=148,
            blood_pressure_diastolic=92,
            respiratory_rate=18,
            temperature_c=37.1,
            spo2_percent=97,
        ),
        withheld_facts=[
            "Took sildenafil earlier today for erectile dysfunction.",
            "Chest pain began after a heated argument at work.",
            "Occasional episodes over the last 2 weeks but shorter.",
            "Stopped a prescribed statin 6 months ago due to muscle aches.",
        ],
        triage="urgent",
        doctor_opening_prompt="You are a doctor. Take a focused chest pain history including risk factors and red flags.",
    )
    out = Path(out_path).expanduser().resolve()
    save_case_file(case, out)
    console.print(Panel.fit(f"Wrote case JSON to: {out}", title="Medical Case"))


@app.command("print")
def print_case(
    case_file: str = typer.Option(
        str(DEFAULT_CASE_OUT), help="Path to a medical case JSON file."
    )
):
    """
    Pretty-print a saved case for review.
    """
    cpath = Path(case_file).expanduser().resolve()
    if not cpath.exists():
        raise typer.BadParameter(f"case_file not found: {cpath}")
    case = load_case_file(cpath)
    console.print(Panel.fit(case.title, title="Title"))
    console.print(Panel.fit(json.dumps(json.loads(cpath.read_text()), indent=2), title="Case JSON"))


@app.command("chat")
def chat(
    case_file: str = typer.Option(
        str(DEFAULT_CASE_OUT), help="Path to a medical case JSON file."
    ),
    model: str = typer.Option("gpt-4o-mini", help="LLM model for PersonaAgent."),
    style_only: bool = typer.Option(
        True, help="Use persona style only; avoid leaking persona content into patient."
    ),
):
    """
    Start an interactive chat with the patient (PersonaAgent) for this case.
    Note: Requires litellm to be configured with a valid API provider/key.
    """
    cpath = Path(case_file).expanduser().resolve()
    if not cpath.exists():
        raise typer.BadParameter(f"case_file not found: {cpath}")
    case = load_case_file(cpath)

    ppath = Path(case.persona_file).expanduser().resolve()
    if not ppath.exists():
        raise typer.BadParameter(f"persona_file referenced by case not found: {ppath}")
    profile = _load_persona(ppath)

    agent = build_agent_for_case(case, profile, model=model, style_only=style_only)

    console.print(Panel.fit("Medical case chat started. Type /exit to quit.", title="Chat"))
    console.print(Panel.fit(case.doctor_opening_prompt, title="Doctor Prompt"))
    while True:
        user_input = typer.prompt("Doctor")
        if user_input.strip().lower() in {"/exit", "exit", "quit"}:
            break
        reply = agent.ask(user_input)
        console.print(Panel(reply, title="Patient"))


if __name__ == "__main__":
    app()


