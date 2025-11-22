"""
Tag clinical profile JSON files with a suitability flag for doctor-patient conversation diagnosis.

For each profile JSON in a target folder (e.g., data/preprocessed/profiles), the script:
1) Calls an LLM with a short prompt asking whether a doctor could likely reach a good diagnosis
   from a conversation with the patient, given the profile details.
2) Writes back to each JSON:
     "conversation_suitability": {
         "is_suitable": true/false,
         "confidence": "<low|medium|high>",
         "reason": "<model rationale>",
         "model": "<model name used>"
     }

Usage:
  python scripts/tag_conversation_suitability.py data/preprocessed/profiles
  python scripts/tag_conversation_suitability.py data/preprocessed/profiles/16.json --dry-run

Env:
  OPENAI_API_KEY must be set (uses the OpenAI API).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import List, Tuple
from dotenv import load_dotenv
from openai import AsyncOpenAI
load_dotenv()

DEFAULT_MODEL = "gpt-5-mini"


def build_prompt(profile: dict) -> str:
    """Create a concise prompt describing the assessment we want."""
    return (
        "You are assessing a simulated doctor-patient interview scenario.\n"
        "Given the clinical profile below, determine if a doctor could realistically reach a good diagnosis "
        "through conversation with the patient (no labs/imaging beyond what is described). "
        "Answer strictly with JSON using keys: is_suitable (true/false), confidence (low|medium|high), reason.\n"
        f"Clinical profile:\n{json.dumps(profile, ensure_ascii=False, indent=2)}\n"
    )


async def call_llm(client: AsyncOpenAI, model: str, prompt: str) -> Tuple[bool, str, str]:
    """Invoke the LLM and parse the JSON response."""
    resp = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=2000,
        temperature=1,
    )
    content = resp.choices[0].message.content or ""
    content = content.replace("```json", "").replace("```", "").strip()
    try:
        data = json.loads(content)
        is_suitable = bool(data.get("is_suitable"))
        confidence = str(data.get("confidence", "unknown"))
        reason = str(data.get("reason", "")).strip()
        return is_suitable, confidence, reason
    except Exception as exc:
        raise RuntimeError(f"Failed to parse model response as JSON: {content}") from exc


async def tag_profile(path: Path, client: AsyncOpenAI, model: str, dry_run: bool = False) -> None:
    """Load a profile JSON, call the LLM, and write back the suitability tag."""
    profile = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(profile, dict):
        raise ValueError(f"Profile at {path} is not a JSON object.")

    prompt = build_prompt(profile)
    suitable, confidence, reason = await call_llm(client, model, prompt)

    profile["conversation_suitability"] = {
        "is_suitable": suitable,
        "confidence": confidence,
        "reason": reason,
        "model": model,
    }

    if dry_run:
        print(f"[DRY-RUN] Would update {path} with: {profile['conversation_suitability']}")
        return

    path.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Updated {path} (is_suitable={suitable}, confidence={confidence})")


async def main_async() -> None:
    parser = argparse.ArgumentParser(description="Tag profiles with conversation suitability for diagnosis.")
    parser.add_argument(
        "profile_path",
        type=Path,
        help="Path to a profile JSON file or a folder containing profile JSONs (e.g., data/preprocessed/profiles)",
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="LLM model to use (default: gpt-4o-mini)")
    parser.add_argument("--dry-run", action="store_true", help="Do not write changes, just show what would happen.")
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=5,
        help="Maximum concurrent LLM calls.",
    )
    args = parser.parse_args()

    # if not os.environ.get("OPENAI_API_KEY"):
    #     raise EnvironmentError("OPENAI_API_KEY must be set.")

    target = args.profile_path
    if not target.exists():
        raise FileNotFoundError(f"Profile path not found: {target}")

    if target.is_dir():
        profile_files: List[Path] = sorted(target.glob("*.json"))
    else:
        profile_files = [target]

    if not profile_files:
        print("No JSON profile files found to process.")
        return

    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(args.max_concurrent)

    async def bound_tag(path: Path) -> None:
        async with semaphore:
            try:
                await tag_profile(path, client, args.model, dry_run=args.dry_run)
            except Exception as exc:
                print(f"[ERROR] {path}: {exc}")

    await asyncio.gather(*(bound_tag(p) for p in profile_files))


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
