"""
Script to match preprocessed profiles with BLPs.

This creates a mapping where each preprocessed profile filename is paired with a BLP,
using only the files that actually exist in the preprocessed directories.

Strategy options:
1. Round-robin: Cycle through available BLPs evenly
2. Random: Randomly assign BLPs to profiles
3. One-to-one: Match filenames directly (profile 01.json -> BLP 01.json)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional
import random


def match_profiles_to_blps(
    profile_dir: str = "data/preprocessed/profiles",
    blp_dir: str = "data/preprocessed/blps",
    output_file: str = "data/preprocessed/case_persona_mapping.json",
    strategy: str = "round-robin",
    seed: int = 42,
) -> Dict:
    """
    Match preprocessed profile files with available BLPs.

    Args:
        profile_dir: Directory with preprocessed profile JSON files
        blp_dir: Directory with BLP JSON files
        output_file: Where to save the mapping
        strategy: Matching strategy ('round-robin', 'random', 'one-to-one')
        seed: Random seed for reproducibility

    Returns:
        Dictionary with mapping information
    """
    # Get available profiles and BLPs
    profile_path = Path(profile_dir)
    blp_path = Path(blp_dir)

    profile_files = sorted(profile_path.glob("*.json"))
    blp_files = sorted(blp_path.glob("*.json"))

    if not profile_files:
        raise ValueError(f"No profile files found in {profile_dir}")
    if not blp_files:
        raise ValueError(f"No BLP files found in {blp_dir}")

    print(f"Preprocessed profiles: {len(profile_files)}")
    print(f"Available BLPs: {len(blp_files)}")
    print(f"Matching strategy: {strategy}")

    # Extract stems (filenames without .json)
    profile_stems = [f.stem for f in profile_files]
    blp_stems = [f.stem for f in blp_files]

    print(f"\nProfile filenames (first 10): {profile_stems[:10]}")
    print(f"BLP filenames: {blp_stems}")

    # Create mapping based on strategy
    mapping = {}

    if strategy == "one-to-one":
        # Match profiles with BLPs that have the same filename
        # Only creates mappings for profiles that have a matching BLP filename
        for profile_stem in profile_stems:
            if profile_stem in blp_stems:
                mapping[profile_stem] = profile_stem

        if not mapping:
            print("\n[WARNING] No matching filenames found!")
            print("Falling back to round-robin strategy...")
            strategy = "round-robin"  # Fall back to round-robin

    if strategy == "round-robin":
        # Cycle through available BLPs evenly
        for i, profile_stem in enumerate(profile_stems):
            blp_idx = i % len(blp_stems)
            blp_stem = blp_stems[blp_idx]
            mapping[profile_stem] = blp_stem

    elif strategy == "random":
        # Randomly assign BLPs
        random.seed(seed)
        for profile_stem in profile_stems:
            blp_stem = random.choice(blp_stems)
            mapping[profile_stem] = blp_stem

    # Create full mapping info
    mapping_info = {
        "strategy": strategy,
        "total_cases": len(profile_files),
        "total_blps": len(blp_files),
        "total_matched": len(mapping),
        "available_blps": blp_stems,
        "mapping": mapping,
        "usage_stats": {}
    }

    # Calculate usage statistics
    usage_counts = {}
    for blp_stem in mapping.values():
        usage_counts[blp_stem] = usage_counts.get(blp_stem, 0) + 1

    mapping_info["usage_stats"] = usage_counts

    # Save mapping
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(mapping_info, f, indent=2)

    print(f"\nMapping saved to {output_path}")
    print(f"Matched {len(mapping)} profiles to BLPs")
    print(f"\nUsage statistics:")
    for blp_stem, count in sorted(usage_counts.items()):
        print(f"  {blp_stem}: used {count} times")

    print(f"\nSample mappings (profile -> BLP):")
    for profile_stem, blp_stem in list(mapping.items())[:10]:
        print(f"  {profile_stem}.json -> {blp_stem}.json")

    return mapping_info


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Match preprocessed profiles with BLPs"
    )
    parser.add_argument(
        "--profile-dir",
        type=str,
        default="data/preprocessed/profiles",
        help="Directory with preprocessed profile JSON files"
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
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    match_profiles_to_blps(
        profile_dir=args.profile_dir,
        blp_dir=args.blp_dir,
        output_file=args.output,
        strategy=args.strategy,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
