from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from .train_doctor import DoctorTrainingCase
from .models import BehavioralLinguisticProfile, PatientProfile


def load_training_cases(
    num_cases: int = 10,
    blp_dir: str = "data/preprocessed/blps",
    profile_dir: str = "data/preprocessed/profiles",
    case_persona_mapping: Optional[str] = None,
    verbose: bool = False
) -> List[DoctorTrainingCase]:
    """
    Load training cases from preprocessed JSON files.

    This function loads BLPs and clinical profiles that have been preprocessed
    and saved as JSON files. It pairs them based on matching filenames or an
    optional case-persona mapping file.

    Args:
        num_cases: Number of cases to load (default: 10)
        blp_dir: Directory containing preprocessed BLP JSON files
        profile_dir: Directory containing preprocessed profile JSON files
        case_persona_mapping: Optional path to JSON file mapping case IDs to BLP files
                             (for when you have more profiles than BLPs)
        verbose: If True, print debug messages

    Returns:
        List of DoctorTrainingCase objects

    Examples:
        # Load with matching filenames (01.json profile with 01.json BLP)
        cases = load_training_cases(num_cases=10)

        # Load with case-persona mapping (for round-robin or random matching)
        cases = load_training_cases(
            num_cases=100,
            case_persona_mapping="data/preprocessed/case_persona_mapping.json"
        )
    """
    blp_path = Path(blp_dir)
    profile_path = Path(profile_dir)

    # Check if directories exist
    if not blp_path.exists():
        raise FileNotFoundError(f"BLP directory not found: {blp_dir}")
    if not profile_path.exists():
        raise FileNotFoundError(f"Profile directory not found: {profile_dir}")

    # Get all BLP and profile files
    blp_files = {f.stem: f for f in sorted(blp_path.glob("*.json"))}
    profile_files = {f.stem: f for f in sorted(profile_path.glob("*.json"))}

    if not blp_files:
        raise ValueError(f"No BLP files found in {blp_dir}")
    if not profile_files:
        raise ValueError(f"No profile files found in {profile_dir}")

    if verbose:
        print(f"[DEBUG] Found {len(blp_files)} BLP files")
        print(f"[DEBUG] Found {len(profile_files)} profile files")

    # Load case-persona mapping if provided
    mapping = None
    if case_persona_mapping:
        mapping_path = Path(case_persona_mapping)
        if mapping_path.exists():
            with open(mapping_path, "r", encoding="utf-8") as f:
                mapping_data = json.load(f)
                mapping = mapping_data.get("mapping", {})
            if verbose:
                print(f"[DEBUG] Loaded case-persona mapping with {len(mapping)} entries")
        else:
            if verbose:
                print(f"[DEBUG] Case-persona mapping file not found: {case_persona_mapping}")

    cases = []
    processed = 0

    # Strategy 1: Use case-persona mapping if provided
    if mapping:
        if verbose:
            print(f"[DEBUG] Using case-persona mapping strategy")

        for profile_name, profile_file in profile_files.items():
            if processed >= num_cases:
                break

            # Load profile
            with open(profile_file, "r", encoding="utf-8") as f:
                profile_data = json.load(f)
            patient_profile = PatientProfile.model_validate(profile_data)

            # Use filename as the key for mapping lookup (more reliable than profile ID)
            # The mapping file uses case_ids from the original parquet, but we use filenames
            case_id = profile_name  # Use filename (e.g., "07", "162") as identifier

            # Find matching BLP using mapping (try both filename and profile ID)
            blp_name = mapping.get(case_id) or mapping.get(patient_profile.id if patient_profile.id else "")
            if not blp_name:
                if verbose:
                    print(f"[DEBUG] No BLP mapping found for {case_id}, skipping")
                continue

            blp_file = blp_files.get(blp_name)
            if not blp_file:
                if verbose:
                    print(f"[DEBUG] BLP file {blp_name} not found, skipping")
                continue

            # Load BLP
            with open(blp_file, "r", encoding="utf-8") as f:
                blp_data = json.load(f)
            blp = BehavioralLinguisticProfile.model_validate(blp_data)

            # Create training case
            case = DoctorTrainingCase(
                case_id=case_id,
                patient_profile=patient_profile,
                blp=blp
            )
            cases.append(case)
            processed += 1

            if verbose:
                print(f"[DEBUG] Loaded case {processed}/{num_cases}: {case_id} with BLP {blp_name}")

    # Strategy 2: Match by filename (01.json profile with 01.json BLP)
    else:
        if verbose:
            print(f"[DEBUG] Using filename matching strategy")

        # Find matching pairs (profiles and BLPs with same filename)
        matching_names = set(blp_files.keys()) & set(profile_files.keys())

        if not matching_names:
            raise ValueError(
                f"No matching BLP/profile pairs found. "
                f"Consider using case_persona_mapping parameter."
            )

        if verbose:
            print(f"[DEBUG] Found {len(matching_names)} matching pairs")

        for name in sorted(matching_names):
            if processed >= num_cases:
                break

            blp_file = blp_files[name]
            profile_file = profile_files[name]

            # Load BLP
            with open(blp_file, "r", encoding="utf-8") as f:
                blp_data = json.load(f)
            blp = BehavioralLinguisticProfile.model_validate(blp_data)

            # Load profile
            with open(profile_file, "r", encoding="utf-8") as f:
                profile_data = json.load(f)
            patient_profile = PatientProfile.model_validate(profile_data)

            # Get case_id from profile or use filename
            case_id = patient_profile.id or name

            # Create training case
            case = DoctorTrainingCase(
                case_id=case_id,
                patient_profile=patient_profile,
                blp=blp
            )
            cases.append(case)
            processed += 1

            if verbose:
                print(f"[DEBUG] Loaded case {processed}/{num_cases}: {name}")

    if not cases:
        raise ValueError(
            f"No training cases could be loaded. Check that BLP and profile files exist."
        )

    if verbose:
        print(f"[DEBUG] Successfully loaded {len(cases)} training cases")

    return cases
