# Personas Project

This project implements a multi-step **persona generation pipeline** for training clinicians to interact with specific patient archetypes.

Core components:

- **Behavioral & Linguistic Profile (BLP) Extractor**: Takes anonymized interview transcripts and produces a structured profile of language patterns, coping strategies, emotional markers, and other behavioral signals.
- **Patient Profile Builder**: Consumes structured and semi-structured case data (history, diagnoses, risk factors, context) and turns it into a richly-typed patient profile.
- **Simulated Patient Agent**: Combines the BLP and patient profile, plus strict conversational constraints, to act as a high-fidelity simulated patient in clinician–patient interactions.

The environment is managed with `uv` and a local `.venv`. To get started:

```bash
cd /Users/apple/Desktop/Personas
uv venv .venv
source .venv/bin/activate
uv sync
```

## Quickstart: Backend API

Prereqs: Python 3.11+, git, OpenAI-compatible key (`OPENAI_API_KEY`).

```bash
cd /Users/tonyohalloran/Desktop/persona-generation
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .                    # installs deps from pyproject

export OPENAI_API_KEY="sk-..."       # required for LiteLLM calls
uvicorn personas.api:app --reload --port 8000
```

- Defaults target `gemini-3-pro-preview` (see `personas/config.yaml`). Override via `PERSONA_CONFIG_PATH` or per-request model parameters.
- Live LLM smoke test (incurs token usage):

```bash
OPENAI_API_KEY="sk-..." pytest tests/test_patient_pipeline.py -q
```

### Environment variables (`.env`)

Backend LLM access is configured via a `.env` file loaded with `python-dotenv`. Create a `.env`
file in the project root with at least:

```bash
# Gemini (default) via LiteLLM
GEMINI_API_KEY="your-gemini-key"     # or GOOGLE_API_KEY if preferred

# Optional OpenAI-compatible key (if you override models)
OPENAI_API_KEY="sk-..."
```

Do **not** commit your real key to version control.

# Personas Frontend

Node.js + React (Vite) frontend that mirrors the persona-generation architecture:

- **Top left**: Raw interview transcript → anonymized transcript → Behavioral & Linguistic Profile (BLP).
- **Bottom left**: Raw case (structured + unstructured) → structured Patient Profile.
- **Right**: Simulated patient conversation that will be driven by the BLP + Patient Profile + constraints.

The frontend talks to the FastAPI backend via HTTP endpoints under `/api`.

## Getting started

From the project root:

```bash
cd /Users/apple/Desktop/Personas/frontend
npm install
VITE_API_BASE_URL="http://localhost:8000" npm run dev
```

Then open the printed localhost URL (by default `http://localhost:5173`) in your browser.

- Frontend model defaults are `gemini-3-pro-preview` (editable in the UI).
- If pointing to a remote API, set `VITE_API_BASE_URL` before `npm run dev`.

With both servers running, the UI will call the FastAPI backend at `/api/*` for BLP extraction, patient profile building, simulated patient turns, critiques, and prompt optimization.***
