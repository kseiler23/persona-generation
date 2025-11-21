from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional
import pandas as pd

from .blp_extractor import BLPExtractor
from .patient_profile_builder import PatientProfileBuilder
from .train_doctor import DoctorTrainingCase
from .models import BehavioralLinguisticProfile, PatientProfile


def load_training_cases(
    num_cases: int = 10,
    cases_file: str = "data/clinical_cases/cases.parquet",
    transcripts_dir: str = "data/transcripts",
    model: str = "gemini/gemini-3-pro-preview",
    use_preprocessed: bool = True,
    blp_dir: Optional[str] = "data/preprocessed/blps",
    profile_dir: Optional[str] = "data/preprocessed/profiles"
) -> List[DoctorTrainingCase]:
    """
    Load training cases from parquet file and transcripts directory.

    Args:
        num_cases: Number of cases to load (default: 10)
        cases_file: Path to parquet file with clinical cases
        transcripts_dir: Path to directory with transcript files
        model: Model to use for BLP extraction and patient profile building
        use_preprocessed: If True, load from preprocessed JSON files (default: True)
        blp_dir: Directory containing preprocessed BLP JSON files
        profile_dir: Directory containing preprocessed profile JSON files

    Returns:
        List of DoctorTrainingCase objects
    """
    # Load clinical cases
    print(f"[DEBUG] Reading from data files...")
    df = pd.read_parquet(cases_file)

    # Load transcripts
    transcripts_path = Path(transcripts_dir)
    transcript_files = sorted(transcripts_path.glob("*.txt"))

    # Initialize paths for preprocessed data
    blp_path = Path(blp_dir) if blp_dir else None
    profile_path = Path(profile_dir) if profile_dir else None

    # Check if we can use preprocessed data
    can_use_preprocessed = use_preprocessed and blp_path and profile_path
    if can_use_preprocessed:
        can_use_preprocessed = blp_path.exists() and profile_path.exists()
        if not can_use_preprocessed:
            print(f"[DEBUG] Preprocessed directories not found, will extract on-the-fly")

    # Initialize extractors only if needed
    blp_extractor = None
    profile_builder = None
    if not can_use_preprocessed:
        print(f"[DEBUG] Initializing extractors for on-the-fly processing...")
        blp_extractor = BLPExtractor(model=model)
        profile_builder = PatientProfileBuilder(model=model)

    cases = []

    for idx in range(min(num_cases, len(df), len(transcript_files))):
        case_row = df.iloc[idx]
        transcript_file = transcript_files[idx]
        case_id = case_row["case_id"]

        # Load or extract BLP
        if can_use_preprocessed:
            blp_file = blp_path / f"{transcript_file.stem}.json"
            if blp_file.exists():
                print(f"[DEBUG] Loading preprocessed BLP for {transcript_file.stem}...")
                with open(blp_file, "r", encoding="utf-8") as f:
                    blp_data = json.load(f)
                blp = BehavioralLinguisticProfile.model_validate(blp_data)
            else:
                print(f"[DEBUG] Preprocessed BLP not found for {transcript_file.stem}, extracting...")
                if blp_extractor is None:
                    blp_extractor = BLPExtractor(model=model)
                with open(transcript_file, "r", encoding="utf-8") as f:
                    transcript = f.read()
                blp = blp_extractor.extract(transcript)
        else:
            print(f"[DEBUG] Reading transcript and extracting BLP...")
            with open(transcript_file, "r", encoding="utf-8") as f:
                transcript = f.read()
            blp = blp_extractor.extract(transcript)

        # Load or build patient profile
        if can_use_preprocessed:
            profile_file = profile_path / f"{case_id}.json"
            if profile_file.exists():
                print(f"[DEBUG] Loading preprocessed profile for {case_id}...")
                with open(profile_file, "r", encoding="utf-8") as f:
                    profile_data = json.load(f)
                patient_profile = PatientProfile.model_validate(profile_data)
            else:
                print(f"[DEBUG] Preprocessed profile not found for {case_id}, building...")
                if profile_builder is None:
                    profile_builder = PatientProfileBuilder(model=model)
                case_text = case_row["case_text"]
                if pd.isna(case_text) or not case_text.strip():
                    case_text = case_row.get("abstract", "")
                patient_profile = profile_builder.build_from_case(case_text)
        else:
            print(f"[DEBUG] Building patient profile from case...")
            case_text = case_row["case_text"]
            if pd.isna(case_text) or not case_text.strip():
                case_text = case_row.get("abstract", "")
            patient_profile = profile_builder.build_from_case(case_text)

        # Create training case
        case = DoctorTrainingCase(
            case_id=case_id,
            patient_profile=patient_profile,
            blp=blp
        )
        cases.append(case)
        print(f"[DEBUG] Loaded case {idx + 1}/{num_cases}")

    return cases
