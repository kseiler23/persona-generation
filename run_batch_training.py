"""
Batch training script using preprocessed data.

This script loads preprocessed BLPs and clinical profiles, then runs
the doctor agent training pipeline with them.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from personas.data_loader import load_training_cases
from personas.train_doctor import run_training_rollouts, optimize_doctor_agent


def run_batch_training(
    num_cases: int = 10,
    num_rollouts: int = 1,
    model_name: str = "gemini/gemini-3-pro-preview",
    blp_dir: str = "data/preprocessed/blps",
    profile_dir: str = "data/preprocessed/profiles",
    case_persona_mapping: Optional[str] = "data/preprocessed/case_persona_mapping.json",
    use_optimization: bool = False,
    only_suitable_profiles: bool = False,
    verbose: bool = True
):
    """
    Run batch training using preprocessed data.

    Args:
        num_cases: Number of training cases to load
        num_rollouts: Number of rollouts per case (if > 1, triggers optimization)
        model_name: Model to use for training
        blp_dir: Directory with preprocessed BLP JSON files
        profile_dir: Directory with preprocessed profile JSON files
        case_persona_mapping: Path to case-persona mapping file (for BLP reuse)
        use_optimization: If True, use DSPy optimization (MIPROv2)
        only_suitable_profiles: If True, only use profiles flagged with conversation_suitability.is_suitable == True
        verbose: Print progress messages
    """

    if verbose:
        print("=" * 60)
        print("BATCH TRAINING WITH PREPROCESSED DATA")
        print("=" * 60)
        print(f"\nConfiguration:")
        print(f"  Cases to load: {num_cases}")
        print(f"  Rollouts per case: {num_rollouts}")
        print(f"  Model: {model_name}")
        print(f"  BLP directory: {blp_dir}")
        print(f"  Profile directory: {profile_dir}")
        print(f"  Case-persona mapping: {case_persona_mapping}")
        print(f"  Use optimization: {use_optimization}")
        print(f"  Only suitable profiles: {only_suitable_profiles}")
        print()

    # Check if preprocessed directories exist
    blp_path = Path(blp_dir)
    profile_path = Path(profile_dir)

    if not blp_path.exists():
        print(f"[ERROR] BLP directory not found: {blp_dir}")
        print("Run preprocessing first:")
        print("  python preprocess_profiles_async.py --task blp")
        return

    if not profile_path.exists():
        print(f"[ERROR] Profile directory not found: {profile_dir}")
        print("Run preprocessing first:")
        print("  python preprocess_profiles_async.py --task profiles")
        return

    # Check for case-persona mapping if specified
    if case_persona_mapping:
        mapping_path = Path(case_persona_mapping)
        if not mapping_path.exists():
            print(f"[WARNING] Case-persona mapping not found: {case_persona_mapping}")
            print("Creating mapping now...")

            # Auto-generate mapping
            import json
            profile_files = sorted(profile_path.glob("*.json"))
            blp_files = sorted(blp_path.glob("*.json"))
            blps = [f.stem for f in blp_files]

            if not blps:
                print(f"[ERROR] No BLP files found in {blp_dir}")
                return

            mapping = {f.stem: blps[i % len(blps)] for i, f in enumerate(profile_files)}
            result = {
                'strategy': 'round-robin',
                'total_cases': len(mapping),
                'total_blps': len(blps),
                'available_blps': blps,
                'mapping': mapping,
                'usage_stats': {b: sum(1 for v in mapping.values() if v == b) for b in blps}
            }

            mapping_path.parent.mkdir(parents=True, exist_ok=True)
            with open(mapping_path, 'w') as f:
                json.dump(result, f, indent=2)

            print(f"[OK] Created mapping: {len(mapping)} profiles → {len(blps)} BLPs")

    # Load training cases from preprocessed data
    if verbose:
        print("\n" + "=" * 60)
        print("LOADING TRAINING CASES")
        print("=" * 60)

    try:
        cases = load_training_cases(
            num_cases=num_cases,
            blp_dir=blp_dir,
            profile_dir=profile_dir,
            case_persona_mapping=case_persona_mapping,
            only_suitable_profiles=only_suitable_profiles,
            verbose=verbose
        )

        if verbose:
            print(f"\n[OK] Loaded {len(cases)} training cases")

            # Show BLP reuse statistics
            blp_ids = {}
            for case in cases:
                blp_id = case.blp.id if case.blp.id else "unknown"
                blp_ids[blp_id] = blp_ids.get(blp_id, 0) + 1

            print(f"\nBLP Reuse Statistics:")
            print(f"  Total cases: {len(cases)}")
            print(f"  Unique BLPs: {len(blp_ids)}")
            print(f"  Reuse ratio: {len(cases) / len(blp_ids) if blp_ids else 0:.1f}x")

            for blp_id, count in sorted(blp_ids.items(), key=lambda x: -x[1])[:5]:
                blp_id_short = blp_id[:20] + "..." if len(blp_id) > 20 else blp_id
                print(f"    {blp_id_short}: used {count} times")

    except Exception as e:
        print(f"\n[ERROR] Failed to load training cases: {e}")
        import traceback
        traceback.print_exc()
        return

    # Run training
    if verbose:
        print("\n" + "=" * 60)
        print("RUNNING TRAINING")
        print("=" * 60)

    try:
        if use_optimization:
            print("\n[INFO] Using DSPy optimization (MIPROv2)")
            print(f"[INFO] This will optimize the doctor agent prompt over {num_cases} cases")
            print("[INFO] May take a while depending on model and API rate limits...\n")

            optimize_doctor_agent(
                cases=cases,
                model_name=model_name,
                max_metric_calls=min(20, num_cases * 2)
            )

            print("\n[OK] Optimization complete!")
            print("[INFO] Optimized prompt saved to prompts/simulated_doctor/system_prompt.txt")
            print("[INFO] Training traces saved to data/traces/")

        else:
            print(f"\n[INFO] Running {num_rollouts} rollout(s) per case")
            print(f"[INFO] Total simulations: {len(cases) * num_rollouts}")
            print("[INFO] Generating training data (no prompt optimization)...\n")

            results = run_training_rollouts(
                cases=cases,
                model_name=model_name,
                num_rollouts=num_rollouts
            )

            print(f"\n[OK] Training complete!")
            print(f"[INFO] Generated {len(results)} simulation traces")
            print(f"[INFO] Traces saved to data/traces/")

            # Show summary statistics
            if results:
                scores = [r[1].overall_score for r in results]
                avg_score = sum(scores) / len(scores)
                print(f"\nPerformance Summary:")
                print(f"  Average score: {avg_score:.2f}")
                print(f"  Best score: {max(scores):.2f}")
                print(f"  Worst score: {min(scores):.2f}")

    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        return

    if verbose:
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Review training traces in data/traces/")
        print("2. Analyze performance metrics")
        print("3. Run more training rounds if needed")
        if not use_optimization:
            print("4. Consider using --optimize flag for prompt optimization")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run batch training with preprocessed data"
    )
    parser.add_argument(
        "--num-cases",
        type=int,
        default=10,
        help="Number of training cases to load (default: 10)"
    )
    parser.add_argument(
        "--num-rollouts",
        type=int,
        default=1,
        help="Number of rollouts per case (default: 1)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini/gemini-3-pro-preview",
        help="Model to use for training (default: gemini/gemini-3-pro-preview)"
    )
    parser.add_argument(
        "--blp-dir",
        type=str,
        default="data/preprocessed/blps",
        help="Directory with BLP JSON files"
    )
    parser.add_argument(
        "--profile-dir",
        type=str,
        default="data/preprocessed/profiles",
        help="Directory with profile JSON files"
    )
    parser.add_argument(
        "--case-persona-mapping",
        type=str,
        default="data/preprocessed/case_persona_mapping.json",
        help="Path to case-persona mapping file"
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Use DSPy optimization (MIPROv2) instead of simple rollouts"
    )
    parser.add_argument(
        "--only-suitable-profiles",
        action="store_true",
        help="Use only profiles where conversation_suitability.is_suitable == True"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output"
    )

    args = parser.parse_args()

    run_batch_training(
        num_cases=args.num_cases,
        num_rollouts=args.num_rollouts,
        model_name=args.model,
        blp_dir=args.blp_dir,
        profile_dir=args.profile_dir,
        case_persona_mapping=args.case_persona_mapping,
        use_optimization=args.optimize,
        only_suitable_profiles=args.only_suitable_profiles,
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main()
