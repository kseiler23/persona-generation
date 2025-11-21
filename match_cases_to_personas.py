"""
Script to match clinical cases with personas (BLPs from transcripts).

This creates a mapping where each clinical case is paired with a BLP,
allowing you to create training cases even when you have more clinical
cases than transcripts.

Strategy options:
1. Round-robin: Cycle through available BLPs
2. Random: Randomly assign BLPs to cases
3. One-to-one: Match first N cases with first N BLPs (default)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
import random


def match_cases_to_personas(
    cases_file: str = "data/clinical_cases/cases.parquet",
    blp_dir: str = "data/preprocessed/blps",
    output_file: str = "data/preprocessed/case_persona_mapping.json",
    strategy: str = "round-robin",
    limit: Optional[int] = None,
    seed: int = 42,
) -> Dict:
    """
    Match clinical cases with available BLPs.

    Args:
        cases_file: Path to cases parquet
        blp_dir: Directory with BLP JSON files
        output_file: Where to save the mapping
        strategy: Matching strategy ('round-robin', 'random', 'one-to-one')
        limit: Optional limit on number of cases
        seed: Random seed for reproducibility

    Returns:
        Dictionary with mapping information
    """
    # Load cases
    df = pd.read_parquet(cases_file)
    if limit:
        df = df.head(limit)

    # Get available BLPs
    blp_path = Path(blp_dir)
    blp_files = sorted(blp_path.glob("*.json"))

    if not blp_files:
        raise ValueError(f"No BLP files found in {blp_dir}")

    print(f"Clinical cases: {len(df)}")
    print(f"Available BLPs: {len(blp_files)}")
    print(f"Matching strategy: {strategy}")

    # Create mapping based on strategy
    mapping = {}

    if strategy == "one-to-one":
        # Match first N cases with first N BLPs
        for i in range(min(len(df), len(blp_files))):
            case_id = df.iloc[i]["case_id"]
            blp_file = blp_files[i].stem  # e.g., "01" from "01.json"
            mapping[case_id] = blp_file

    elif strategy == "round-robin":
        # Cycle through available BLPs
        for i, (_, row) in enumerate(df.iterrows()):
            case_id = row["case_id"]
            blp_idx = i % len(blp_files)
            blp_file = blp_files[blp_idx].stem
            mapping[case_id] = blp_file

    elif strategy == "random":
        # Randomly assign BLPs
        random.seed(seed)
        blp_stems = [f.stem for f in blp_files]
        for _, row in df.iterrows():
            case_id = row["case_id"]
            blp_file = random.choice(blp_stems)
            mapping[case_id] = blp_file

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Create full mapping info
    mapping_info = {
        "strategy": strategy,
        "total_cases": len(df),
        "total_blps": len(blp_files),
        "total_matched": len(mapping),
        "available_blps": [f.stem for f in blp_files],
        "mapping": mapping,
        "usage_stats": {}
    }

    # Calculate usage statistics
    usage_counts = {}
    for blp_file in mapping.values():
        usage_counts[blp_file] = usage_counts.get(blp_file, 0) + 1

    mapping_info["usage_stats"] = usage_counts

    # Save mapping
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(mapping_info, f, indent=2)

    print(f"\nMapping saved to {output_path}")
    print(f"Matched {len(mapping)} cases")
    print(f"\nUsage statistics:")
    for blp_file, count in sorted(usage_counts.items()):
        print(f"  {blp_file}: used {count} times")

    return mapping_info


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Match clinical cases with personas (BLPs)"
    )
    parser.add_argument(
        "--cases-file",
        type=str,
        default="data/clinical_cases/cases.parquet",
        help="Path to cases parquet"
    )
    parser.add_argument(
        "--blp-dir",
        type=str,
        default="data/preprocessed/blps",
        help="Directory with BLP JSON files"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/preprocessed/case_persona_mapping.json",
        help="Output mapping file"
    )
    parser.add_argument(
        "--strategy",
        choices=["one-to-one", "round-robin", "random"],
        default="round-robin",
        help="Matching strategy"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of cases to process"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    match_cases_to_personas(
        cases_file=args.cases_file,
        blp_dir=args.blp_dir,
        output_file=args.output,
        strategy=args.strategy,
        limit=args.limit,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
