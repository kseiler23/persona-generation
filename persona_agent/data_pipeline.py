import os
import json
import uuid
import shutil
import datetime
from pathlib import Path
from typing import Dict, List, Any, Iterable, Optional

from textblob import TextBlob
import tiktoken
from tqdm import tqdm
from litellm import completion


TOKEN_LIMIT_DEFAULT = 350
ENCODER = tiktoken.get_encoding("cl100k_base")


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def data_dir() -> Path:
    return project_root() / "data"


def transcripts_dir(group: Optional[str] = None) -> Path:
    """
    Returns the transcripts directory. If a group is provided, returns data/transcripts/<group>.
    """
    base = data_dir() / "transcripts"
    return base / group if group else base


def enriched_dir(group: Optional[str] = None) -> Path:
    """
    Returns the enriched directory. If a group is provided, returns data/enriched/<group>.
    """
    base = data_dir() / "enriched"
    return base / group if group else base


def ensure_directories(group: Optional[str] = None) -> None:
    """
    Ensure base and (optionally) group-specific directories exist.
    """
    base_dirs = [data_dir(), transcripts_dir(), enriched_dir()]
    for d in base_dirs:
        d.mkdir(parents=True, exist_ok=True)
    if group:
        for d in [transcripts_dir(group), enriched_dir(group)]:
            d.mkdir(parents=True, exist_ok=True)


def copy_transcripts_from(source: Path, group: Optional[str] = None) -> None:
    """
    Copy *.json transcripts from a source directory into data/transcripts.
    """
    ensure_directories(group=group)
    if not source.exists() or not source.is_dir():
        raise FileNotFoundError(f"Transcript source not found: {source}")
    for src in source.glob("*.json"):
        dest = transcripts_dir(group) / src.name
        shutil.copy2(src, dest)


def chunk_text(text: str, limit: int = TOKEN_LIMIT_DEFAULT) -> Iterable[str]:
    sentences = TextBlob(text).sentences
    buffer: List[str] = []
    size = 0
    for sentence in sentences:
        token_count = len(ENCODER.encode(str(sentence)))
        if size + token_count > limit and buffer:
            yield " ".join(buffer)
            buffer, size = [], 0
        buffer.append(str(sentence))
        size += token_count
    if buffer:
        yield " ".join(buffer)


def _extract_message_content(litellm_response) -> str:
    # Litellm returns an OpenAI-style response object
    try:
        return litellm_response.choices[0].message["content"]
    except Exception:
        try:
            return litellm_response["choices"][0]["message"]["content"]
        except Exception:
            return ""


def enrich_segment(
    segment: Dict[str, Any],
    seg_idx: int,
    doc_id: str,
    model: str,
) -> List[Dict[str, Any]]:
    text = segment.get("text", "").strip()
    speaker = segment.get("speaker", "Unknown")
    enriched_chunks: List[Dict[str, Any]] = []

    for chunk_idx, chunk in enumerate(chunk_text(text)):
        sentiment = TextBlob(chunk).sentiment.polarity
        token_count = len(ENCODER.encode(chunk))

        response = completion(
            model=model,
            messages=[
                {"role": "user", "content": f"Summarise in <30 words: {chunk}"},
            ],
            temperature=0.2,
        )
        summary = _extract_message_content(response)

        enriched_chunks.append(
            {
                "chunk_id": str(uuid.uuid4()),
                "doc_id": doc_id,
                "speaker": speaker,
                "segment_idx": seg_idx,
                "chunk_idx": chunk_idx,
                "text": chunk,
                "tokens": token_count,
                "sentiment": sentiment,
                "summary": summary,
                "created_at": datetime.datetime.utcnow().isoformat() + "Z",
            }
        )
    return enriched_chunks


def enrich_transcript(
    transcript: List[Dict[str, Any]],
    doc_id: str,
    model: str,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for i, seg in enumerate(transcript):
        results.extend(enrich_segment(seg, i, doc_id, model))
    return results


def run_enrichment(
    model: str,
    token_limit: int = TOKEN_LIMIT_DEFAULT,
    max_files: Optional[int] = None,
    group: Optional[str] = None,
) -> None:
    """
    Iterate over transcripts in data/transcripts and write enriched JSON into data/enriched.
    """
    global TOKEN_LIMIT_DEFAULT
    TOKEN_LIMIT_DEFAULT = token_limit

    ensure_directories(group=group)
    files = list(transcripts_dir(group).glob("*.json"))
    if max_files is not None:
        files = files[:max_files]

    for transcript_file in tqdm(files, desc="Enriching transcripts"):
        doc_id = transcript_file.stem
        out_path = enriched_dir(group) / f"{doc_id}_enriched.json"
        if out_path.exists():
            continue
        with open(transcript_file, "r") as f:
            transcript = json.load(f)
        enriched = enrich_transcript(transcript, doc_id, model)
        with open(out_path, "w") as f:
            json.dump(enriched, f, indent=2, ensure_ascii=False)


def build_context_from_enriched(
    speaker_filter: Optional[str] = None,
    group: Optional[str] = None,
) -> str:
    """
    Build a single string context from all JSON in data/enriched.
    Optionally filter to a single speaker.
    """
    ensure_directories(group=group)
    pieces: List[str] = []
    for jf in sorted(enriched_dir(group).glob("*.json")):
        try:
            with open(jf, "r") as f:
                data = json.load(f)
            for item in data:
                speaker = item.get("speaker", "Unknown")
                if speaker_filter and speaker != speaker_filter:
                    continue
                text = item.get("text", "")
                sentiment = item.get("sentiment", 0)
                summary = item.get("summary", "")
                pieces.append(
                    f"Speaker: {speaker}\nText: {text}\nSentiment: {sentiment}\nSummary: {summary}\n---"
                )
        except Exception as e:
            pieces.append(f"[WARN] Failed reading {jf.name}: {e}")
    return "Transcript context:\n\n" + "\n".join(pieces)


def build_context_from_transcripts(
    speaker_filter: Optional[str] = None,
    group: Optional[str] = None,
) -> str:
    """
    Build a single string context by concatenating all raw transcripts.
    Optionally filter to a single speaker.
    """
    ensure_directories(group=group)
    pieces: List[str] = []
    for jf in sorted(transcripts_dir(group).glob("*.json")):
        try:
            with open(jf, "r") as f:
                data = json.load(f)
            for seg in data:
                speaker = seg.get("speaker", "Unknown")
                if speaker_filter and speaker != speaker_filter:
                    continue
                text = seg.get("text", "")
                pieces.append(f"{speaker}: {text}")
        except Exception as e:
            pieces.append(f"[WARN] Failed reading {jf.name}: {e}")
    return "Transcript context:\n\n" + "\n".join(pieces)


