import json
import re
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from litellm import completion
from pydantic import BaseModel, Field
from textblob import TextBlob

from .data_pipeline import (
    ensure_directories,
    enriched_dir,
    transcripts_dir,
    data_dir,
)


class PersonaProfile(BaseModel):
    speaker: str = Field(..., description="Name/label of the speaker this persona represents.")
    summary: str = Field(..., description="Concise 3-5 sentence overview of the persona.")
    roles: List[str] = Field(default_factory=list, description="Common roles and responsibilities.")
    goals: List[str] = Field(default_factory=list, description="Primary goals or objectives.")
    values: List[str] = Field(default_factory=list, description="Core values and principles.")
    expertise: List[str] = Field(default_factory=list, description="Domains of expertise or skills.")
    knowledge_domains: List[str] = Field(default_factory=list, description="Areas of consistent knowledge.")
    communication_style: Dict[str, str] = Field(
        default_factory=dict,
        description="How they speak: tone, pacing, vocabulary, rhetorical patterns.",
    )
    preferences: List[str] = Field(default_factory=list, description="Clear likes, defaults, and preferences.")
    quirks: List[str] = Field(default_factory=list, description="Recurring habits, phrases, or idiosyncrasies.")
    dos: List[str] = Field(default_factory=list, description="Tactics that work well with this persona.")
    donts: List[str] = Field(default_factory=list, description="Tactics that do not work well.")
    blindspots: List[str] = Field(default_factory=list, description="Common gaps or weaknesses.")
    top_topics: List[str] = Field(default_factory=list, description="Frequently discussed topics.")
    catchphrases: List[str] = Field(default_factory=list, description="Representative quotes or phrases.")
    emotional_profile: Dict[str, float] = Field(
        default_factory=dict,
        description="Aggregate emotions/sentiment (-1..1 for sentiment; other emotions optional).",
    )


STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "if",
    "then",
    "else",
    "for",
    "to",
    "from",
    "of",
    "in",
    "on",
    "with",
    "by",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "am",
    "it",
    "this",
    "that",
    "these",
    "those",
    "as",
    "at",
    "we",
    "you",
    "they",
    "he",
    "she",
    "i",
    "me",
    "my",
    "our",
    "your",
    "their",
}


def personas_dir(group: Optional[str] = None) -> Path:
    """
    Returns the personas directory. If group is provided, returns data/personas/<group>.
    """
    base = data_dir() / "personas"
    return base / group if group else base


def ensure_personas_dir(group: Optional[str] = None) -> None:
    ensure_directories(group=group)
    personas_dir(group).mkdir(parents=True, exist_ok=True)


def _tokenize(text: str) -> List[str]:
    words = re.findall(r"\b[a-zA-Z][a-zA-Z0-9'-]*\b", text.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 2]


def _top_keywords(chunks: List[str], k: int = 20) -> List[Tuple[str, int]]:
    counter: Counter = Counter()
    for c in chunks:
        counter.update(_tokenize(c))
    return counter.most_common(k)


def collect_corpus_per_speaker(
    context_source: str = "enriched",
    group: Optional[str] = None,
) -> Dict[str, Dict[str, List[str]]]:
    """
    Returns: dict[speaker] -> {"summaries": [...], "quotes": [...], "all_text": [...]}
    """
    ensure_directories(group=group)
    buckets: Dict[str, Dict[str, List[str]]] = {}
    if context_source == "enriched":
        for jf in sorted(enriched_dir(group).glob("*.json")):
            try:
                data = json.loads(Path(jf).read_text())
            except Exception:
                continue
            for item in data:
                speaker = item.get("speaker", "Unknown")
                quotes = item.get("text", "")
                summ = item.get("summary", "") or quotes
                b = buckets.setdefault(speaker, {"summaries": [], "quotes": [], "all_text": []})
                if summ:
                    b["summaries"].append(summ)
                if quotes:
                    b["quotes"].append(quotes)
                if quotes:
                    b["all_text"].append(quotes)
    elif context_source == "transcripts":
        for jf in sorted(transcripts_dir(group).glob("*.json")):
            try:
                data = json.loads(Path(jf).read_text())
            except Exception:
                continue
            for seg in data:
                speaker = seg.get("speaker", "Unknown")
                text = seg.get("text", "")
                b = buckets.setdefault(speaker, {"summaries": [], "quotes": [], "all_text": []})
                if text:
                    b["summaries"].append(text)
                    b["quotes"].append(text)
                    b["all_text"].append(text)
    else:
        raise ValueError("context_source must be 'enriched' or 'transcripts'")
    return buckets


def _persona_prompt(speaker: str, corpus: Dict[str, List[str]]) -> str:
    summaries_text = "\n".join(corpus.get("summaries", [])[:2000])  # keep it bounded but generous
    quotes = corpus.get("quotes", [])[:50]
    quotes_text = "\n".join(f"- {q}" for q in quotes)

    sentiments = [TextBlob(t).sentiment.polarity for t in corpus.get("all_text", [])[:1000]]
    avg_sent = sum(sentiments) / len(sentiments) if sentiments else 0.0

    keywords = [kw for kw, _ in _top_keywords(corpus.get("all_text", []) + corpus.get("summaries", []), 25)]
    keywords_text = ", ".join(keywords)

    schema_hint = json.dumps(
        {
            "speaker": speaker,
            "summary": "3-5 sentences...",
            "roles": ["..."],
            "goals": ["..."],
            "values": ["..."],
            "expertise": ["..."],
            "knowledge_domains": ["..."],
            "communication_style": {
                "tone": "...",
                "pacing": "...",
                "vocabulary": "...",
                "structure": "...",
            },
            "preferences": ["..."],
            "quirks": ["..."],
            "dos": ["..."],
            "donts": ["..."],
            "blindspots": ["..."],
            "top_topics": ["..."],
            "catchphrases": ["..."],
            "emotional_profile": {"sentiment": avg_sent},
        },
        indent=2,
    )

    prompt = f"""
You are an expert analyst that builds detailed communication personas from conversation corpora.
Extract the persona for the speaker named "{speaker}" using the corpus below.

Output strictly valid JSON matching the schema example (types and keys) and do not include any prose outside JSON.
When unsure, infer cautiously from patterns and representative quotes. Use neutral, descriptive language.

Context signals:
- Average sentiment (approx): {avg_sent:.3f}
- Top keywords: {keywords_text}
- Representative quotes (truncated):
{quotes_text}

Corpus summaries:
{summaries_text}

Schema example (use same keys; fill with content from this corpus):
{schema_hint}
"""
    return prompt.strip()


def synthesize_persona_for_speaker(
    speaker: str,
    corpus: Dict[str, List[str]],
    model: str,
) -> PersonaProfile:
    messages = [
        {
            "role": "system",
            "content": "You convert corpora into structured JSON personas. Respond with JSON only.",
        },
        {"role": "user", "content": _persona_prompt(speaker, corpus)},
    ]
    resp = completion(model=model, messages=messages, temperature=0.2)
    try:
        content = resp.choices[0].message["content"]
    except Exception:
        content = ""
    try:
        data = json.loads(content)
        return PersonaProfile(**data)
    except Exception:
        # Minimal fallback
        return PersonaProfile(
            speaker=speaker,
            summary=content.strip()[:2000] or f"Persona for {speaker} based on provided corpus.",
        )


def persona_json_to_instructions(profile: PersonaProfile) -> str:
    """
    Convert a PersonaProfile into a crisp system instruction string for chat.
    """
    style = profile.communication_style or {}
    lines: List[str] = []
    lines.append(f"You are role-playing as '{profile.speaker}'.")
    if profile.summary:
        lines.append(profile.summary.strip())
    if profile.roles:
        lines.append("Roles: " + ", ".join(profile.roles))
    if profile.goals:
        lines.append("Goals: " + "; ".join(profile.goals))
    if profile.values:
        lines.append("Values: " + ", ".join(profile.values))
    if profile.expertise:
        lines.append("Expertise: " + ", ".join(profile.expertise))
    if profile.knowledge_domains:
        lines.append("Knowledge: " + ", ".join(profile.knowledge_domains))
    if style:
        parts = []
        for k in ["tone", "pacing", "vocabulary", "structure"]:
            v = style.get(k)
            if v:
                parts.append(f"{k}: {v}")
        if parts:
            lines.append("Communication style: " + "; ".join(parts))
    if profile.preferences:
        lines.append("Preferences: " + "; ".join(profile.preferences))
    if profile.quirks:
        lines.append("Quirks: " + "; ".join(profile.quirks))
    if profile.dos:
        lines.append("Do: " + "; ".join(profile.dos))
    if profile.donts:
        lines.append("Don't: " + "; ".join(profile.donts))
    if profile.catchphrases:
        lines.append("Representative phrases: " + "; ".join(profile.catchphrases))

    lines.append("Stay consistent with these constraints in all responses.")
    return "\n".join(lines).strip()


def persona_style_instructions(profile: PersonaProfile) -> str:
    """
    Build a style-only instruction string: mimic grammar/syntax/register/formatting,
    but do not transfer personal facts, preferences, goals, or knowledge.
    """
    style = profile.communication_style or {}
    lines: List[str] = []
    lines.append(f"You are role-playing as '{profile.speaker}'.")
    parts = []
    for k in ["tone", "pacing", "vocabulary", "structure"]:
        v = style.get(k)
        if v:
            parts.append(f"{k}: {v}")
    if parts:
        lines.append("Communication style: " + "; ".join(parts))
    # Style cues beyond core style: quirks and characteristic phrasing (no personal facts)
    if profile.quirks:
        lines.append("Stylistic quirks to emulate (use naturally and sparingly): " + "; ".join(profile.quirks[:5]))
    if profile.catchphrases:
        lines.append(
            "If it fits, lightly echo characteristic phrasing (paraphrased, not verbatim; do not reveal identity)."
        )
    lines.append(
        "Style-only mode: Do not transfer any personal facts, preferences, or private details from the persona."
    )
    lines.append(
        "Rely only on this chat's inputs and general domain knowledge; ask clarifying questions if needed."
    )
    lines.append("Stay consistent with these constraints in all responses.")
    return "\n".join(lines).strip()


def generate_personas(
    model: str = "gpt-4o-mini",
    context_source: str = "enriched",
    speaker: Optional[str] = None,
    group: Optional[str] = None,
) -> List[PersonaProfile]:
    """
    Generate persona JSON files under data/personas for each speaker (or a single target).
    """
    ensure_personas_dir(group=group)
    corpora = collect_corpus_per_speaker(context_source=context_source, group=group)
    results: List[PersonaProfile] = []
    targets = [speaker] if speaker else list(corpora.keys())
    for sp in targets:
        corpus = corpora.get(sp)
        if not corpus:
            continue
        profile = synthesize_persona_for_speaker(sp, corpus, model=model)
        out_path = personas_dir(group) / f"{sp.replace(' ', '_')}.json"
        out_path.write_text(json.dumps(profile.model_dump(), indent=2, ensure_ascii=False))
        results.append(profile)
    return results


