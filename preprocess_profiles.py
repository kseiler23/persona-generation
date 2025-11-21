"""
Preprocessing script to extract and store BLP and clinical profiles.

This script processes the data to create:
1. BLPs (Behavioral Linguistic Profiles) from transcripts
2. Clinical profiles (Patient Profiles) from case texts

The preprocessed data is saved to disk for faster loading during training.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional
import pandas as pd
from tqdm import tqdm

from personas.blp_extractor import BLPExtractor
from personas.patient_profile_builder import PatientProfileBuilder
from personas.models import BehavioralLinguisticProfile, PatientProfile


def preprocess_blps(
    transcripts_dir: str = "data/transcripts",
    output_dir: str = "data/preprocessed/blps",
    model: str = "gemini/gemini-3-pro-preview",
    limit: Optional[int] = None
) -> None:
    """
    Extract BLPs from all transcripts and save them as JSON files.

    Args:
        transcripts_dir: Directory containing transcript files
        output_dir: Directory to save preprocessed BLP JSON files
        model: Model to use for BLP extraction
        limit: Optional limit on number of transcripts to process
    """
    transcripts_path = Path(transcripts_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get all transcript files
    transcript_files = sorted(transcripts_path.glob("*.txt"))
    if limit:
        transcript_files = transcript_files[:limit]

    print(f"Processing {len(transcript_files)} transcripts...")

    # Initialize extractor
    blp_extractor = BLPExtractor(model=model)

    # Process each transcript
    for transcript_file in tqdm(transcript_files, desc="Extracting BLPs"):
        # Read transcript
        with open(transcript_file, "r", encoding="utf-8") as f:
            transcript = f.read()

        # Skip empty transcripts
        if not transcript.strip():
            print(f"Skipping empty transcript: {transcript_file.name}")
            continue

        # Extract BLP
        try:
            blp = blp_extractor.extract(transcript)

            # Save to JSON file
            output_file = output_path / f"{transcript_file.stem}.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(blp.model_dump(), f, indent=2)

        except Exception as e:
            print(f"Error processing {transcript_file.name}: {e}")
            continue

    print(f"BLP extraction complete. Saved to {output_path}")


def preprocess_clinical_profiles(
    cases_file: str = "data/clinical_cases/cases.parquet",
    output_dir: str = "data/preprocessed/profiles",
    model: str = "gemini/gemini-3-pro-preview",
    limit: Optional[int] = None
) -> None:
    """
    Extract clinical profiles from case texts and save them as JSON files.

    Args:
        cases_file: Path to parquet file with clinical cases
        output_dir: Directory to save preprocessed profile JSON files
        model: Model to use for profile building
        limit: Optional limit on number of cases to process
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load cases
    df = pd.read_parquet(cases_file)
    if limit:
        df = df.head(limit)

    print(f"Processing {len(df)} clinical cases...")

    # Initialize profile builder
    profile_builder = PatientProfileBuilder(model=model)

    # Process each case
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Building profiles"):
        case_id = row["case_id"]
        case_text = row["case_text"]

        # Fallback to abstract if case_text is empty
        if pd.isna(case_text) or not case_text.strip():
            case_text = row.get("abstract", "")

        # Skip if still empty
        if not case_text.strip():
            print(f"Skipping case {case_id}: no text available")
            continue

        # Build patient profile
        try:
            patient_profile = profile_builder.build_from_case(case_text)

            # Save to JSON file
            output_file = output_path / f"{case_id}.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(patient_profile.model_dump(), f, indent=2)

        except Exception as e:
            print(f"Error processing case {case_id}: {e}")
            continue

    print(f"Profile building complete. Saved to {output_path}")


def main():
    """Main preprocessing pipeline."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Preprocess BLPs and clinical profiles from source data"
    )
    parser.add_argument(
        "--task",
        choices=["blp", "profiles", "both"],
        default="both",
        help="Which preprocessing task to run"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of items to process (useful for testing)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini/gemini-3-pro-preview",
        help="Model to use for extraction"
    )
    parser.add_argument(
        "--transcripts-dir",
        type=str,
        default="data/transcripts",
        help="Directory containing transcript files"
    )
    parser.add_argument(
        "--cases-file",
        type=str,
        default="data/clinical_cases/cases.parquet",
        help="Path to parquet file with clinical cases"
    )
    parser.add_argument(
        "--blp-output",
        type=str,
        default="data/preprocessed/blps",
        help="Output directory for BLP JSON files"
    )
    parser.add_argument(
        "--profile-output",
        type=str,
        default="data/preprocessed/profiles",
        help="Output directory for profile JSON files"
    )

    args = parser.parse_args()

    if args.task in ["blp", "both"]:
        print("\n=== Preprocessing BLPs ===")
        preprocess_blps(
            transcripts_dir=args.transcripts_dir,
            output_dir=args.blp_output,
            model=args.model,
            limit=args.limit
        )

    if args.task in ["profiles", "both"]:
        print("\n=== Preprocessing Clinical Profiles ===")
        preprocess_clinical_profiles(
            cases_file=args.cases_file,
            output_dir=args.profile_output,
            model=args.model,
            limit=args.limit
        )

    print("\n=== Preprocessing Complete ===")


if __name__ == "__main__":
    main()
