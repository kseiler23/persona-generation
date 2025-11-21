"""
Async preprocessing script to extract and store BLP and clinical profiles.

This script processes the data concurrently to create:
1. BLPs (Behavioral Linguistic Profiles) from transcripts
2. Clinical profiles (Patient Profiles) from case texts

Includes optional filtering step to select high-quality cases.
The preprocessed data is saved to disk for faster loading during training.

Format consistency:
- Input: transcript files (01.txt, 02.txt, ...) and parquet with case_ids
- Output: JSON files with same naming (01.json matches 01.txt)
- Index mapping: Maps filtered case indices back to original indices
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Optional, List, Tuple, Dict
import pandas as pd
from tqdm.asyncio import tqdm as async_tqdm

from personas.blp_extractor_async import AsyncBLPExtractor
from personas.patient_profile_builder_async import AsyncPatientProfileBuilder
from personas.case_filter import CaseFilter
from personas.models import BehavioralLinguisticProfile, PatientProfile


async def filter_cases(
    df: pd.DataFrame,
    transcript_files: List[Path],
    filter_model: str = "gpt-4o-mini",
    max_concurrent: int = 10,
) -> Tuple[List[int], Dict[str, any]]:
    """
    Filter cases using async GPT calls while maintaining index consistency.

    Args:
        df: DataFrame with cases
        transcript_files: List of transcript files (for index mapping)
        filter_model: Model to use for filtering
        max_concurrent: Maximum concurrent filter requests

    Returns:
        Tuple of (included_indices, filter_report)
    """
    case_filter = CaseFilter(model=filter_model)
    semaphore = asyncio.Semaphore(max_concurrent)

    async def filter_one(idx: int, row: pd.Series, transcript_file: Path) -> dict:
        async with semaphore:
            case_text = row.get("case_text", "")
            abstract = row.get("abstract", "")
            if pd.isna(case_text):
                case_text = ""
            if pd.isna(abstract):
                abstract = ""

            result = await case_filter.should_include(case_text, abstract)
            result["index"] = idx
            result["case_id"] = row["case_id"]
            result["transcript_file"] = transcript_file.name
            return result

    print("Filtering cases...")
    tasks = [
        filter_one(idx, row, transcript_files[idx])
        for idx, (_, row) in enumerate(df.iterrows())  # Fixed: unpack (index, row) tuple
        if idx < len(transcript_files)
    ][:len(transcript_files)]  # Ensure we don't exceed transcript count

    filter_results = []
    for coro in async_tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Filtering"):
        result = await coro
        filter_results.append(result)

    # Sort by original index to maintain order
    filter_results.sort(key=lambda x: x["index"])

    # Get indices of included cases
    included_indices = [r["index"] for r in filter_results if r.get("include", False)]

    print(f"\nFiltering complete: {len(included_indices)}/{len(filter_results)} cases included")

    # Print filtering statistics
    confidence_counts = {}
    for r in filter_results:
        if r.get("include", False):
            conf = r.get("confidence", "unknown")
            confidence_counts[conf] = confidence_counts.get(conf, 0) + 1

    print(f"Confidence breakdown: {confidence_counts}")

    # Create filter report
    filter_report = {
        "total_cases": len(filter_results),
        "included_cases": len(included_indices),
        "excluded_cases": len(filter_results) - len(included_indices),
        "inclusion_rate": len(included_indices) / len(filter_results) if filter_results else 0,
        "confidence_breakdown": confidence_counts,
        "results": filter_results
    }

    return included_indices, filter_report


async def preprocess_blps_async(
    transcripts_dir: str = "data/transcripts",
    output_dir: str = "data/preprocessed/blps",
    model: str = "gemini/gemini-3-pro-preview",
    limit: Optional[int] = None,
    max_concurrent: int = 5,
    included_indices: Optional[List[int]] = None,
) -> None:
    """
    Extract BLPs from all transcripts and save them as JSON files (async).

    Args:
        transcripts_dir: Directory containing transcript files
        output_dir: Directory to save preprocessed BLP JSON files
        model: Model to use for BLP extraction
        limit: Optional limit on number of transcripts to process
        max_concurrent: Maximum concurrent extraction requests
        included_indices: Optional list of indices to process (from filtering)
    """
    transcripts_path = Path(transcripts_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get all transcript files
    transcript_files = sorted(transcripts_path.glob("*.txt"))
    if limit:
        transcript_files = transcript_files[:limit]

    # Filter by included indices if provided
    if included_indices is not None:
        transcript_files = [transcript_files[i] for i in included_indices if i < len(transcript_files)]

    print(f"\nProcessing {len(transcript_files)} transcripts...")

    # Initialize extractor
    blp_extractor = AsyncBLPExtractor(model=model)

    # Create semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_one(transcript_file: Path) -> Tuple[str, Optional[str]]:
        """Process a single transcript and return (filename, error_msg)."""
        async with semaphore:
            # Read transcript
            try:
                with open(transcript_file, "r", encoding="utf-8") as f:
                    transcript = f.read()

                # Skip empty transcripts
                if not transcript.strip():
                    return transcript_file.name, "Empty transcript"

                # Extract BLP
                blp = await blp_extractor.extract(transcript)

                # Save to JSON file with same basename
                output_file = output_path / f"{transcript_file.stem}.json"
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(blp.model_dump(), f, indent=2)

                return transcript_file.name, None

            except Exception as e:
                return transcript_file.name, str(e)

    # Process all transcripts concurrently
    tasks = [process_one(tf) for tf in transcript_files]
    results = []

    for coro in async_tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Extracting BLPs"):
        filename, error = await coro
        if error:
            results.append(f"Error processing {filename}: {error}")

    if results:
        print(f"\nEncountered {len(results)} errors:")
        for msg in results[:10]:  # Show first 10 errors
            print(f"  - {msg}")

    print(f"BLP extraction complete. Saved to {output_path}")


async def preprocess_clinical_profiles_async(
    cases_file: str = "data/clinical_cases/cases.parquet",
    transcripts_dir: str = "data/transcripts",
    output_dir: str = "data/preprocessed/profiles",
    model: str = "gemini/gemini-3-pro-preview",
    limit: Optional[int] = None,
    max_concurrent: int = 5,
    enable_filtering: bool = True,
    filter_model: str = "gpt-4o-mini",
    filter_output: Optional[str] = None,
    index_mapping_output: Optional[str] = None,
) -> None:
    """
    Extract clinical profiles from case texts and save them as JSON files (async).

    Args:
        cases_file: Path to parquet file with clinical cases
        transcripts_dir: Directory containing transcript files
        output_dir: Directory to save preprocessed profile JSON files
        model: Model to use for profile building
        limit: Optional limit on number of cases to process
        max_concurrent: Maximum concurrent extraction requests
        enable_filtering: Whether to filter cases before processing
        filter_model: Model to use for case filtering
        filter_output: Optional path to save filter results JSON
        index_mapping_output: Optional path to save index mapping JSON
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load cases
    df = pd.read_parquet(cases_file)

    # Get transcript files for index consistency
    transcripts_path = Path(transcripts_dir)
    transcript_files = sorted(transcripts_path.glob("*.txt"))

    # Determine number of cases to process
    num_items = len(df)
    if limit:
        num_items = min(num_items, limit)

    df = df.head(num_items)

    # For filtering and profiling, we don't need transcripts
    # But we'll use transcript filenames for consistent output naming
    # Generate transcript-style names for all cases if we have more cases than transcripts
    if len(df) > len(transcript_files):
        print(f"\n[INFO] More cases ({len(df)}) than transcripts ({len(transcript_files)})")
        print(f"[INFO] Generating transcript-style indices for all cases")
        # Create dummy transcript paths for consistent naming (Path already imported at top)
        transcript_files = [
            Path(f"{i+1:02d}.txt") for i in range(len(df))
        ]
    else:
        transcript_files = transcript_files[:num_items]

    print(f"\nProcessing {len(df)} clinical cases")

    # Filter cases if enabled
    included_indices = list(range(len(df)))  # Default: include all
    filter_report = None

    if enable_filtering:
        included_indices, filter_report = await filter_cases(
            df, transcript_files, filter_model, max_concurrent
        )

        # Save filter report if requested
        if filter_output and filter_report:
            with open(filter_output, "w", encoding="utf-8") as f:
                json.dump(filter_report, f, indent=2)
            print(f"Filter results saved to {filter_output}")

    # Create index mapping: output_index -> original_index
    index_mapping = {
        "mapping": {i: included_indices[i] for i in range(len(included_indices))},
        "total_original": len(df),
        "total_filtered": len(included_indices),
        "transcript_mapping": {
            transcript_files[included_indices[i]].stem: included_indices[i]
            for i in range(len(included_indices))
            if included_indices[i] < len(transcript_files)
        }
    }

    if index_mapping_output:
        with open(index_mapping_output, "w", encoding="utf-8") as f:
            json.dump(index_mapping, f, indent=2)
        print(f"Index mapping saved to {index_mapping_output}")

    # Process only included cases
    df_filtered = df.iloc[included_indices].reset_index(drop=True)
    transcript_files_filtered = [transcript_files[i] for i in included_indices]

    print(f"\nProcessing {len(df_filtered)} clinical cases...")

    # Initialize profile builder
    profile_builder = AsyncPatientProfileBuilder(model=model)

    # Create semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_one(row: pd.Series, transcript_file: Path) -> Tuple[str, Optional[str]]:
        """Process a single case and return (filename, error_msg)."""
        async with semaphore:
            case_id = row["case_id"]
            case_text = row["case_text"]

            # Fallback to abstract if case_text is empty
            if pd.isna(case_text) or not case_text.strip():
                case_text = row.get("abstract", "")

            # Skip if still empty
            if not case_text.strip():
                return transcript_file.stem, "No text available"

            try:
                # Build patient profile
                patient_profile = await profile_builder.build_from_case(case_text)

                # Save to JSON file using transcript basename (for consistency)
                output_file = output_path / f"{transcript_file.stem}.json"
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(patient_profile.model_dump(), f, indent=2)

                return transcript_file.stem, None

            except Exception as e:
                return transcript_file.stem, str(e)

    # Process all cases concurrently
    tasks = [
        process_one(row, transcript_files_filtered[idx])
        for idx, (_, row) in enumerate(df_filtered.iterrows())  # Fixed: unpack (index, row) tuple
    ]
    results = []

    for coro in async_tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Building profiles"):
        filename, error = await coro
        if error:
            results.append(f"Error processing {filename}: {error}")

    if results:
        print(f"\nEncountered {len(results)} errors:")
        for msg in results[:10]:  # Show first 10 errors
            print(f"  - {msg}")

    print(f"Profile building complete. Saved to {output_path}")


async def main_async():
    """Main async preprocessing pipeline."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Preprocess BLPs and clinical profiles from source data (async)"
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
        "--filter-model",
        type=str,
        default="gpt-4o-mini",
        help="Model to use for case filtering"
    )
    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="Disable case filtering"
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=5,
        help="Maximum concurrent API requests"
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
    parser.add_argument(
        "--filter-output",
        type=str,
        default="data/preprocessed/filter_results.json",
        help="Output file for filter results"
    )
    parser.add_argument(
        "--index-mapping-output",
        type=str,
        default="data/preprocessed/index_mapping.json",
        help="Output file for index mapping"
    )

    args = parser.parse_args()

    # Store included indices for consistency across BLP and profiles
    included_indices = None

    if args.task in ["profiles", "both"]:
        print("\n=== Preprocessing Clinical Profiles (Async) ===")
        await preprocess_clinical_profiles_async(
            cases_file=args.cases_file,
            transcripts_dir=args.transcripts_dir,
            output_dir=args.profile_output,
            model=args.model,
            limit=args.limit,
            max_concurrent=args.max_concurrent,
            enable_filtering=not args.no_filter,
            filter_model=args.filter_model,
            filter_output=args.filter_output if not args.no_filter else None,
            index_mapping_output=args.index_mapping_output if not args.no_filter else None,
        )

        # Load index mapping for BLP processing
        if not args.no_filter and Path(args.index_mapping_output).exists():
            with open(args.index_mapping_output) as f:
                index_mapping = json.load(f)
                included_indices = list(index_mapping["mapping"].values())
                print(f"\nUsing {len(included_indices)} filtered indices for BLP extraction")

    if args.task in ["blp", "both"]:
        print("\n=== Preprocessing BLPs (Async) ===")
        await preprocess_blps_async(
            transcripts_dir=args.transcripts_dir,
            output_dir=args.blp_output,
            model=args.model,
            limit=args.limit,
            max_concurrent=args.max_concurrent,
            included_indices=included_indices,
        )

    print("\n=== Preprocessing Complete ===")
    print(f"\nOutput structure:")
    print(f"  - BLPs: {args.blp_output}/")
    print(f"  - Profiles: {args.profile_output}/")
    if not args.no_filter:
        print(f"  - Filter results: {args.filter_output}")
        print(f"  - Index mapping: {args.index_mapping_output}")
    print(f"\nFormat: Each transcript file (e.g., 01.txt) has matching BLP (01.json) and profile (01.json)")


def main():
    """Entry point that runs the async main."""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
