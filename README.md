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

### Environment variables (`.env`)

Backend LLM access is configured via a `.env` file loaded with `python-dotenv`. Create a `.env`
file in the project root with at least:

```bash
OPENAI_API_KEY="sk-..."  # used by LiteLLM for OpenAI-compatible models
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
npm run dev
```

Then open the printed localhost URL (by default `http://localhost:5173`) in your browser.

## Backend integration sketch

When you are ready to connect this UI to your Python pipeline, you can:

- Expose HTTP endpoints (e.g., `/api/blp`, `/api/patient-profile`, `/api/simulated-patient`) from the Python side.
- Replace the placeholder logic in `src/App.tsx` (`handleExtractBLP`, `handleBuildPatientProfile`, and `handleSendTurn`) with `fetch` calls to those endpoints.




