"""
Identify telehealth-related cases from cases.parquet and mark their BLP JSON files.

Steps performed:
1) Load the cases parquet.
2) Flag rows whose text mentions telehealth/telemedicine/virtual care keywords.
3) Map case IDs to BLP IDs via data/preprocessed/case_persona_mapping.json.
4) For each BLP involved, add a boolean `telehealth_case: true` (idempotent).

Usage:
  python scripts/tag_telehealth_cases.py           # updates BLP JSONs
  python scripts/tag_telehealth_cases.py --dry-run # just prints what would change
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Set

import pandas as pd


CASES_PATH = Path("data/clinical_cases/cases.parquet")
MAPPING_PATH = Path("data/preprocessed/case_persona_mapping.json")
BLP_DIR = Path("data/preprocessed/blps")

# Keyword list kept short/on-purpose; tune if needed.
TELEHEALTH_KEYWORDS = [
    "telehealth",
    "tele-health",
    "telemedicine",
    "tele-medicine",
    "telemed",
    "virtual visit",
    "virtual care",
    "video visit",
    "video consultation",
    "video consult",
    "remote consultation",
    "remote visit",
    "phone visit",
    "telephone visit",
    "audio-only visit",
]

# Text columns we will scan if present in the parquet.
TEXT_COLUMNS = [
    "case_text",
    "case",
    "abstract",
    "summary",
    "title",
    "notes",
    "narrative",
    "diagnosis_text",
]


def get_case_id(row: pd.Series) -> str:
    """
    Try a few common column names to recover a string case identifier.
    Falls back to the dataframe index.
    """
    for key in ("case_id", "article_id", "id", "CaseID"):
        if key in row and pd.notna(row[key]):
            return str(row[key])
    return str(row.name)


def row_mentions_telehealth(row: pd.Series) -> bool:
    texts: List[str] = []
    for col in TEXT_COLUMNS:
        if col in row and pd.notna(row[col]):
            val = row[col]
            if isinstance(val, str):
                texts.append(val.lower())
    if not texts:
        return False
    blob = " ".join(texts)
    return any(term in blob for term in TELEHEALTH_KEYWORDS)


def load_mapping() -> dict:
    with MAPPING_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_telehealth_cases() -> List[str]:
    df = pd.read_parquet(CASES_PATH)
    telehealth_rows = df[df.apply(row_mentions_telehealth, axis=1)]
    return [get_case_id(row) for _, row in telehealth_rows.iterrows()]


def cases_to_blp_ids(case_ids: Iterable[str]) -> Set[str]:
    mapping = load_mapping().get("mapping", {})
    found: Set[str] = set()
    for cid in case_ids:
        blp_id = mapping.get(str(cid))
        if blp_id:
            found.add(blp_id)
    return found


def update_blp(blp_id: str, dry_run: bool = False) -> None:
    path = BLP_DIR / f"{blp_id}.json"
    if not path.exists():
        print(f"[WARN] BLP file not found for ID {blp_id}: {path}")
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        already = data.get("telehealth_case") is True
        if dry_run:
            if not already:
                print(f"[DRY-RUN] Would mark {path} as telehealth_case=True")
            return
        if not already:
            data["telehealth_case"] = True
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[OK] Marked telehealth_case=True in {path}")
    else:
        print(f"[WARN] Unexpected JSON structure in {path} (not a dict)")


def main(dry_run: bool = False) -> None:
    case_ids = find_telehealth_cases()
    if not case_ids:
        print("No telehealth-related cases detected.")
        return
    print(f"Detected {len(case_ids)} telehealth cases: {', '.join(case_ids)}")

    blp_ids = cases_to_blp_ids(case_ids)
    if not blp_ids:
        print("No BLP IDs matched these cases; nothing to update.")
        return

    print(f"Updating BLP files for IDs: {', '.join(sorted(blp_ids))}")
    for blp_id in sorted(blp_ids):
        update_blp(blp_id, dry_run=dry_run)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Mark telehealth cases in BLP JSON files.")
    parser.add_argument("--dry-run", action="store_true", help="Report changes without writing files.")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
