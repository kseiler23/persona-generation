## Persona Generation

This project provides a minimal, self-contained pipeline to:
- Ingest transcripts (text-only)
- Enrich them (chunking, sentiment, concise summaries via an LLM)
- Interact with a Litellm-powered agent that takes instructions and a transcript context

No audio/video generation is included.

### Requirements
- Python 3.10+
- `uv` package manager installed (`pipx install uv` or see uv docs)
- An LLM API key (e.g., `OPENAI_API_KEY`) supported by Litellm

### Quickstart (using uv)
```bash
# If you're already in this project directory, you can skip the cd
cd "Persona Generation"
uv venv
source .venv/bin/activate  # or use `uv run` for each command without activating
uv sync

# TextBlob corpora (once)
python -m textblob.download_corpora
```

### Provide transcripts
- Put raw transcripts as JSON files under `data/transcripts/`.
- Each file should be a JSON list of objects with keys at least: `{"speaker": "...", "text": "..."}`.
- Example minimal structure:
```json
[
  {"speaker": "Speaker 1", "text": "Hello, everyone."},
  {"speaker": "Speaker 1", "text": "Today we will discuss..."}
]
```

Optional: copy existing transcripts from another folder (use your absolute path):
```bash
persona-gen sync-transcripts --from "/absolute/path/to/your/transcripts"
```

### No existing data? Create a sample transcript now
Paste the commands below to create a small example file locally, then run the pipeline end‑to‑end.
```bash
mkdir -p "/Users/apple/Desktop/Persona Generation/data/transcripts"
cat > "/Users/apple/Desktop/Persona Generation/data/transcripts/sample.json" << 'EOF'
[
  {"speaker": "Speaker 1", "text": "Hello, everyone."},
  {"speaker": "Speaker 1", "text": "Today we will discuss product updates and priorities."},
  {"speaker": "Speaker 2", "text": "Can you share the KPIs and timelines?"}
]
EOF
```

### Enrich transcripts
This will produce per-chunk enriched JSON files under `data/enriched/`.
```bash
persona-gen enrich --model gpt-4o-mini
```
Environment: ensure your provider key is set (e.g., `export OPENAI_API_KEY=...`).

### Generate personas (from your data)
Create structured persona JSONs under `data/personas/` for each speaker (or a specific one).
```bash
# From enriched data (recommended)
persona-gen persona --context-source enriched --model gpt-4o-mini

# Only for a specific speaker label (exact match)
persona-gen persona --context-source enriched --speaker "Speaker 1"
```
This produces files like `data/personas/Speaker_1.json`.

### Chat with the agent
By default, chat runs in style-only mode with transcripts context included.
```bash
persona-gen chat \
  --persona-file data/personas/Speaker_1.json \
  --model gpt-4o-mini
```
You’ll enter an interactive prompt. Type your message and press Enter. Type `/exit` to quit.

Optionally limit context to one speaker:
```bash
persona-gen chat \
  --context-source enriched \
  --speaker-filter "Speaker 1" \
  --model gpt-4o-mini
```

### Default behavior: style-only, with context
By default, the assistant mimics grammar/syntax/formatting without transferring transcript facts (names, dates, quotes).
```bash
persona-gen chat \
  --persona-file data/personas/Speaker_1.json \
  --model gpt-4o-mini
```
Override examples:
- Exclude transcripts: add `--no-context True`
- Disable style-only (allow persona content transfer): add `--style-only False`

### Commands
```bash
# Copy transcripts into this project
persona-gen sync-transcripts --from /path/to/transcripts

# Run enrichment over JSON transcripts in data/transcripts
persona-gen enrich --model gpt-4o-mini --token-limit 350

# Start interactive chat (default: style-only, with context)
persona-gen chat --persona-file data/personas/Speaker_1.json --model gpt-4o-mini

# Disable transcripts context (opt-out)
persona-gen chat --persona-file data/personas/Speaker_1.json --no-context True --model gpt-4o-mini

# Generate persona JSON(s)
persona-gen persona --context-source enriched --model gpt-4o-mini
```

### Configuration notes
- Litellm selects the provider by the model name. For OpenAI, set `OPENAI_API_KEY`.
- Adjust the model via `--model` to target another provider supported by Litellm.
- Enrichment uses simple sentence-based chunking (by token budget) and TextBlob for sentiment.

### Best quality persona tips
- **Use enriched data**: Run `persona-gen enrich` first and generate personas from `--context-source enriched` for better signal (summaries + sentiment).
- **Pick your best model**: Higher-quality models materially improve persona depth and consistency. Example: `--model gpt-4o-mini` or better.
- **Normalize speaker labels**: Ensure the same person has an identical `speaker` string across files (e.g., always `Speaker 1`).
- **Provide ample coverage**: Include transcripts that span typical interactions, domains, and tones for that speaker.
- **Low temperature for stability**: The pipeline already uses a low temperature for concise, consistent output.
- **Focus context during chat**: Use `--speaker-filter` when chatting to keep responses aligned with a specific persona’s content.
- **Optionally curate personas**: Open the generated file under `data/personas/` and refine fields (e.g., dos/don’ts, catchphrases) to taste.

### Transcript schema
The pipeline only requires `speaker` and `text`. Extra fields are ignored.
```json
[
  {"speaker": "Speaker 1", "text": "Hello, team. Let's start."},
  {"speaker": "Speaker 2", "text": "Agenda item one is metrics."},
  {"speaker": "Speaker 1", "text": "Q3 saw strong growth in PLG."}
]
```
Optional fields (ignored by the core pipeline but safe to include): `timestamp`, `duration`, `turn_id`, etc.

### Advanced options
- **Enrich**
  - `--token-limit`: Adjust chunk size (default 350). Larger = fewer, longer chunks.
  - `--max-files`: Limit how many transcript files to process.
- **Persona**
  - `--context-source`: `enriched` (recommended) or `transcripts`.
  - `--speaker`: Only synthesize for a single exact-matching speaker.
- **Chat**
  - `--persona-file`: Use a generated persona JSON to auto-build instructions.
  - `--style-only` (default True): Mimic style only (grammar/syntax/formatting), ban content transfer.
  - `--no-context` (default False): Do not include transcripts as context (prevents leak).
  - `--speaker-filter`: Restrict context to a single exact-matching speaker.
  - `--context-source`: `enriched` or `transcripts`.
  - `--instructions`: If not using `--persona-file`, provide your own instructions.

### Multiple datasets via group subfolders
You can keep multiple datasets isolated using a `--group` name. Files are organized under:
- `data/transcripts/<group>/`
- `data/enriched/<group>/`
- `data/personas/<group>/`

Example workflow for a dataset named `team_a`:
```bash
# 1) Copy transcripts into data/transcripts/team_a
persona-gen sync-transcripts --from "/abs/path/to/team_a_json" --group team_a

# 2) Enrich only that group's transcripts → data/enriched/team_a
persona-gen enrich --group team_a --model gpt-4o-mini

# 3) Generate personas into data/personas/team_a
persona-gen persona --group team_a --context-source enriched --model gpt-4o-mini

# 4) Chat using that group's context (default)
persona-gen chat --group team_a --context-source enriched \
  --speaker-filter "Speaker 1" --model gpt-4o-mini
```

You can repeat the same flow with a different `--group` (e.g., `team_b`) to maintain separate datasets in parallel.

### Project layout
```
Persona Generation/
  data/
    transcripts/        # input transcripts (JSON)
      team_a/           # group A transcripts
      team_b/           # group B transcripts
    enriched/           # output enriched JSON
      team_a/           # group A enriched chunks
      team_b/           # group B enriched chunks
    personas/           # generated persona JSON
      team_a/           # group A personas
      team_b/           # group B personas
  persona_agent/
    agent.py            # litellm-based agent
    data_pipeline.py    # enrichment + context building
    persona.py          # persona synthesis utilities
    cli.py              # Typer CLI
    __init__.py
  pyproject.toml
  README.md
  .gitignore
```


