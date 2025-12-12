"""
Example Usage of Run Manager Module

Demonstrates how to use the run_manager utilities for output versioning,
metadata tracking, and legacy output detection.

Author: AquaSOT-3D Team
Date: 2025-12-13
"""

import sys
from pathlib import Path

# Add src to path for direct execution
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.aquasot3d.utils import (
    generate_run_id,
    get_git_hash,
    get_git_status,
    create_run_directory,
    save_run_metadata,
    load_run_metadata,
    check_output_directory,
    detect_legacy_outputs,
    find_latest_run,
    setup_train_run,
    setup_eval_run,
)


def example_basic_usage():
    """Example: Basic run ID generation and directory creation."""
    print("=" * 60)
    print("Example 1: Basic Usage")
    print("=" * 60)

    # Generate unique run ID
    run_id = generate_run_id()
    print(f"Generated run ID: {run_id}")

    # Get git information
    git_hash = get_git_hash()
    print(f"Current git hash: {git_hash}")

    # Get detailed git status
    git_status = get_git_status()
    print(f"Git branch: {git_status['branch']}")
    print(f"Working directory clean: {not git_status['is_dirty']}")
    print()


def example_training_setup():
    """Example: Setup training run with versioned outputs."""
    print("=" * 60)
    print("Example 2: Training Run Setup")
    print("=" * 60)

    # Define configuration
    config = {
        'paths': {
            'work_dir': 'outputs/train'
        },
        'model': {
            'type': 'AquaSOT3DTracker',
            'backbone': {'type': 'vit_base'}
        },
        'training': {
            'lr': 0.0001,
            'num_epochs': 100
        }
    }

    # Setup training run (creates versioned directory structure)
    print("Setting up training run...")
    print(f"Base output: {config['paths']['work_dir']}")

    # In practice, you would use:
    # run_dir, checkpoint_dir, log_dir = setup_train_run(config, force=False)
    # print(f"Run directory: {run_dir}")
    # print(f"Checkpoints: {checkpoint_dir}")
    # print(f"Logs: {log_dir}")

    print("(Skipped actual directory creation in this demo)")
    print()


def example_evaluation_setup():
    """Example: Setup evaluation run with checkpoint tracking."""
    print("=" * 60)
    print("Example 3: Evaluation Run Setup")
    print("=" * 60)

    # Define configuration
    config = {
        'evaluation': {
            'output_dir': 'outputs/eval'
        },
        'model': {
            'type': 'AquaSOT3DTracker'
        }
    }

    checkpoint_path = 'outputs/train/20251213_120000_abc123/checkpoints/best_model.pth'

    print("Setting up evaluation run...")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output: {config['evaluation']['output_dir']}")

    # In practice, you would use:
    # run_dir, vis_dir = setup_eval_run(config, checkpoint_path, force=False)
    # print(f"Run directory: {run_dir}")
    # print(f"Visualizations: {vis_dir}")

    print("(Skipped actual directory creation in this demo)")
    print()


def example_metadata_tracking():
    """Example: Metadata tracking and loading."""
    print("=" * 60)
    print("Example 4: Metadata Tracking")
    print("=" * 60)

    import tempfile

    # Create temporary run directory for demo
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir) / '20251213_120000_abc123'
        run_dir.mkdir()

        # Save metadata
        config = {
            'model': {'type': 'AquaSOT3DTracker'},
            'training': {'lr': 0.0001}
        }

        rect_check_result = {
            'valid': True,
            'sequences_checked': ['uuv1', 'uuv2', 'uuv3'],
            'errors': []
        }

        print("Saving metadata...")
        metadata_path = save_run_metadata(
            str(run_dir),
            config=config,
            rect_check_result=rect_check_result,
            run_id='20251213_120000_abc123'
        )

        print(f"Metadata saved to: {metadata_path}")

        # Load metadata
        print("\nLoading metadata...")
        metadata = load_run_metadata(str(run_dir))

        print(f"Run ID: {metadata['run_id']}")
        print(f"Timestamp: {metadata['timestamp']}")
        print(f"Git hash: {metadata['git']['hash']}")
        print(f"Config model type: {metadata['config']['model']['type']}")
        print(f"Rect check valid: {metadata['rect_check']['valid']}")

    print()


def example_legacy_detection():
    """Example: Detect and handle legacy outputs."""
    print("=" * 60)
    print("Example 5: Legacy Output Detection")
    print("=" * 60)

    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create some legacy outputs
        (Path(tmpdir) / 'old_model.pth').touch()
        (Path(tmpdir) / 'results.json').touch()

        legacy_checkpoints = Path(tmpdir) / 'checkpoints'
        legacy_checkpoints.mkdir()
        (legacy_checkpoints / 'model_epoch_10.pth').touch()

        print(f"Created legacy outputs in: {tmpdir}")

        # Detect legacy outputs
        has_legacy, legacy_items = detect_legacy_outputs(tmpdir)

        if has_legacy:
            print(f"\nFound {len(legacy_items)} legacy output(s):")
            for item in legacy_items:
                print(f"  - {Path(item).name}")

            print("\nWarning: These outputs do not follow the run_id versioning scheme.")
            print("Consider migrating them using migrate_legacy_outputs().")

    print()


def example_finding_runs():
    """Example: Find latest run."""
    print("=" * 60)
    print("Example 6: Finding Latest Run")
    print("=" * 60)

    import tempfile
    import time

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create multiple runs
        run_ids = [
            '20251213_100000_aaa111',
            '20251213_120000_bbb222',
            '20251213_140000_ccc333'
        ]

        for run_id in run_ids:
            run_dir, _ = create_run_directory(tmpdir, run_id=run_id)
            config = {'model': {'type': 'test'}}
            save_run_metadata(run_dir, config=config, run_id=run_id)
            time.sleep(0.01)  # Small delay for timestamp ordering

        print(f"Created {len(run_ids)} runs")

        # Find latest run
        result = find_latest_run(tmpdir)

        if result:
            latest_run_dir, metadata = result
            print(f"\nLatest run: {metadata['run_id']}")
            print(f"Timestamp: {metadata['timestamp']}")

    print()


def main():
    """Run all examples."""
    print("\n")
    print("#" * 60)
    print("# Run Manager Module - Usage Examples")
    print("#" * 60)
    print()

    example_basic_usage()
    example_training_setup()
    example_evaluation_setup()
    example_metadata_tracking()
    example_legacy_detection()
    example_finding_runs()

    print("=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
    print()


if __name__ == '__main__':
    main()
