#!/usr/bin/env python3
"""
Data Validation Usage Examples

This script demonstrates how to use data validation utilities in:
- train.py: Validate data before training
- eval.py: Ensure evaluation uses correct preprocessed data
- analyze_dataset.py: Check dataset preprocessing status

Author: AquaSOT-3D Team
Date: 2025-12-13
"""

import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from aquasot3d.utils.data_validation import (
    rect_data_check,
    validate_data_root,
    check_rect_readiness,
    block_polar_data,
    generate_metadata_dict,
    validate_sequences_or_raise,
    get_validation_summary,
    EXPECTED_SONAR_RECT_RESOLUTION,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def example_train_py_validation():
    """
    Example: How to use data validation in train.py

    This shows validation before starting training to ensure all
    sequences have proper rect preprocessing.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 1: train.py validation")
    print("=" * 60)

    # Simulated training configuration
    data_root = "data/"
    train_sequences = ["uuv1", "uuv2", "uuv4"]
    config_path = "configs/train.yaml"

    logger.info("Starting training data validation...")

    try:
        # Step 1: Validate all training sequences have rect data
        # This will raise RuntimeError if any sequence is not ready
        validate_sequences_or_raise(
            data_root=data_root,
            sequence_names=train_sequences,
            strict=True  # Fail fast if data is not ready
        )
        logger.info("All training sequences validated successfully")

        # Step 2: Generate metadata for this training run
        metadata = generate_metadata_dict(
            config_path=config_path,
            extra_info={
                "data_root": data_root,
                "train_sequences": train_sequences,
                "rect_check": True,  # Enable rect validation in metadata
            }
        )

        logger.info(f"Training run metadata:")
        logger.info(f"  Run ID: {metadata['run_id']}")
        logger.info(f"  Git hash: {metadata['git_hash']}")
        logger.info(f"  Config MD5: {metadata['config_md5']}")
        logger.info(f"  Rect check passed: {metadata.get('rect_check_passed', 'N/A')}")

        # Metadata can be saved to training output directory
        # save_json(metadata, f"outputs/{metadata['run_id']}/metadata.json")

        logger.info("Training validation complete. Ready to train!")

    except RuntimeError as e:
        logger.error(f"Training validation failed: {e}")
        logger.error("Please run preprocessing before training:")
        logger.error("  python scripts/preprocess_sonar.py --all")
        return False

    return True


def example_eval_py_validation():
    """
    Example: How to use data validation in eval.py

    This shows validation for evaluation sequences and blocking
    polar data usage.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 2: eval.py validation")
    print("=" * 60)

    # Simulated evaluation configuration
    data_root = "data/"
    eval_sequences = ["uuv3", "uuv5"]

    logger.info("Starting evaluation data validation...")

    # Step 1: Check if sequences are ready
    ready, not_ready, issues = check_rect_readiness(
        data_root=data_root,
        sequence_names=eval_sequences
    )

    if not_ready:
        logger.error(f"Evaluation sequences not ready: {not_ready}")
        for seq_name in not_ready:
            logger.error(f"  {seq_name}: {issues[seq_name]}")
        logger.error("Run preprocessing before evaluation")
        return False

    logger.info(f"All evaluation sequences ready: {ready}")

    # Step 2: Block polar data for each sequence
    for seq_name in eval_sequences:
        seq_path = Path(data_root) / seq_name
        try:
            block_polar_data(seq_path)
            logger.info(f"  {seq_name}: Using rect data (validated)")
        except RuntimeError as e:
            logger.error(f"  {seq_name}: {e}")
            return False

    # Step 3: Generate evaluation metadata
    metadata = generate_metadata_dict(
        config_path="configs/eval.yaml",
        extra_info={
            "eval_sequences": eval_sequences,
            "data_root": data_root,
        }
    )

    logger.info(f"Evaluation run ID: {metadata['run_id']}")
    logger.info("Evaluation validation complete. Ready to evaluate!")

    return True


def example_analyze_dataset_py():
    """
    Example: How to use data validation in analyze_dataset.py

    This shows comprehensive dataset analysis and preprocessing
    status reporting.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 3: analyze_dataset.py validation")
    print("=" * 60)

    data_root = "data/"

    logger.info("Analyzing dataset preprocessing status...")

    # Step 1: Get comprehensive validation summary
    summary = get_validation_summary(data_root)
    print("\n" + summary)

    # Step 2: Get detailed results for each sequence
    ready, not_ready, issues = check_rect_readiness(data_root)

    print("\n" + "-" * 60)
    print("DETAILED ANALYSIS")
    print("-" * 60)

    # Analyze ready sequences
    if ready:
        print(f"\nReady for training/evaluation ({len(ready)} sequences):")
        for seq_name in ready:
            seq_path = Path(data_root) / seq_name
            result = rect_data_check(seq_path, modality="sonar", check_resolution=True)
            print(f"  ✓ {seq_name}:")
            print(f"      Images: {result['num_images']}")
            print(f"      Annotations: {result['num_annotations']}")
            print(f"      Resolution: {result['actual_resolution']}")

    # Analyze problematic sequences
    if not_ready:
        print(f"\nNeeds preprocessing ({len(not_ready)} sequences):")
        for seq_name in not_ready:
            print(f"  ✗ {seq_name}:")
            for issue in issues[seq_name]:
                print(f"      - {issue}")

        print("\nPreprocessing commands:")
        for seq_name in not_ready:
            print(f"  python scripts/preprocess_sonar.py --sequence {seq_name}")

    # Step 3: Verify expected resolution constant
    print(f"\nExpected sonar rect resolution: {EXPECTED_SONAR_RECT_RESOLUTION[0]}x{EXPECTED_SONAR_RECT_RESOLUTION[1]}")

    return len(not_ready) == 0


def example_detailed_sequence_check():
    """
    Example: Detailed single sequence validation

    This shows how to perform detailed validation on a single sequence
    for debugging purposes.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Detailed single sequence check")
    print("=" * 60)

    sequence_name = "uuv1"
    data_root = "data/"
    seq_path = Path(data_root) / sequence_name

    if not seq_path.exists():
        logger.warning(f"Sequence not found: {seq_path}")
        logger.info("Using example path for demonstration...")
        return

    logger.info(f"Checking sequence: {sequence_name}")

    # Check sonar rect data
    logger.info("\nSonar modality validation:")
    sonar_result = rect_data_check(seq_path, modality="sonar", check_resolution=True)

    print(f"  Has rect data: {sonar_result['has_rect_data']}")
    print(f"  Image directory exists: {sonar_result['img_dir_exists']}")
    print(f"  Groundtruth file exists: {sonar_result['gt_file_exists']}")
    print(f"  Resolution correct: {sonar_result['resolution_ok']}")

    if sonar_result['actual_resolution']:
        print(f"  Actual resolution: {sonar_result['actual_resolution'][0]}x{sonar_result['actual_resolution'][1]}")

    print(f"  Number of images: {sonar_result['num_images']}")
    print(f"  Number of annotations: {sonar_result['num_annotations']}")

    if sonar_result['errors']:
        print(f"  Errors:")
        for error in sonar_result['errors']:
            print(f"    - {error}")

    # Check RGB data
    logger.info("\nRGB modality validation:")
    rgb_result = rect_data_check(seq_path, modality="rgb", check_resolution=False)

    print(f"  Has data: {rgb_result['has_rect_data']}")
    print(f"  Number of images: {rgb_result['num_images']}")
    print(f"  Number of annotations: {rgb_result['num_annotations']}")

    if rgb_result['actual_resolution']:
        print(f"  Resolution: {rgb_result['actual_resolution'][0]}x{rgb_result['actual_resolution'][1]}")


def example_integration_with_config():
    """
    Example: Integration with configuration management

    Shows how to combine data validation with config loading
    for complete reproducibility tracking.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Integration with config management")
    print("=" * 60)

    config_path = "configs/train.yaml"
    data_root = "data/"

    logger.info("Loading configuration and validating data...")

    # Step 1: Load config (pseudo-code)
    # from aquasot3d.utils import load_config
    # cfg = load_config(config_path)
    # data_root = cfg.data.root
    # sequences = cfg.data.train_sequences

    # For this example, use dummy values
    sequences = ["uuv1", "uuv2"]

    # Step 2: Generate comprehensive metadata
    metadata = generate_metadata_dict(
        config_path=config_path,
        extra_info={
            "data_root": data_root,
            "sequences": sequences,
            "rect_check": True,
            "expected_sonar_resolution": EXPECTED_SONAR_RECT_RESOLUTION,
            # Add any custom fields
            "experiment_name": "baseline_training",
            "notes": "First training run with validated data",
        }
    )

    logger.info("Generated metadata:")
    for key, value in metadata.items():
        if key not in ["rect_issues", "rect_ready_sequences", "rect_not_ready_sequences"]:
            logger.info(f"  {key}: {value}")

    # This metadata should be saved with training outputs
    logger.info("\nMetadata should be saved to:")
    logger.info(f"  outputs/train/{metadata['run_id']}/metadata.json")

    return metadata


def main():
    """Run all examples"""
    print("\n" + "=" * 70)
    print("DATA VALIDATION UTILITIES - USAGE EXAMPLES")
    print("=" * 70)

    # Run examples
    examples = [
        ("train.py validation", example_train_py_validation),
        ("eval.py validation", example_eval_py_validation),
        ("analyze_dataset.py", example_analyze_dataset_py),
        ("Detailed sequence check", example_detailed_sequence_check),
        ("Config integration", example_integration_with_config),
    ]

    for name, func in examples:
        try:
            func()
        except Exception as e:
            logger.error(f"Example '{name}' failed: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("EXAMPLES COMPLETE")
    print("=" * 70)
    print("\nKey functions:")
    print("  - validate_sequences_or_raise(): Fast fail validation for train/eval")
    print("  - check_rect_readiness(): Scan dataset preprocessing status")
    print("  - generate_metadata_dict(): Create reproducibility metadata")
    print("  - block_polar_data(): Enforce rect data usage")
    print("  - get_validation_summary(): Human-readable status report")
    print("\nFor more details, see:")
    print("  - src/aquasot3d/utils/data_validation.py (implementation)")
    print("  - tests/test_data_validation.py (comprehensive tests)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
