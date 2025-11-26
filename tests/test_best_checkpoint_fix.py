#!/usr/bin/env python3
"""
Test script to verify best checkpoint saving fix
Tests that checkpoint_best.pth is created after first epoch

Note: This is a manual verification script. Run with:
    python tests/test_best_checkpoint_fix.py
"""
import pytest
from pathlib import Path


@pytest.mark.skip(reason="Manual verification script - run directly with python, not pytest")
def test_best_checkpoint_exists():
    """Test that checkpoint_best.pth is created after training."""
    checkpoint_dir = Path('outputs/train/checkpoints')
    checkpoint_best = checkpoint_dir / 'checkpoint_best.pth'

    if not checkpoint_dir.exists():
        pytest.skip("Checkpoint directory does not exist - run training first")

    assert checkpoint_best.exists(), "checkpoint_best.pth was NOT created"


def main():
    """Manual verification entry point."""
    import sys

    checkpoint_dir = Path('outputs/train/checkpoints')
    checkpoint_last = checkpoint_dir / 'checkpoint_last.pth'
    checkpoint_best = checkpoint_dir / 'checkpoint_best.pth'

    print("=" * 80)
    print("Best Checkpoint Fix Verification")
    print("=" * 80)

    print(f"\nCheckpoint directory: {checkpoint_dir}")
    print(f"Checkpoint directory exists: {checkpoint_dir.exists()}")

    if checkpoint_dir.exists():
        checkpoints = list(checkpoint_dir.glob('*.pth'))
        print(f"\nFound checkpoints ({len(checkpoints)}):")
        for cp in sorted(checkpoints):
            size_mb = cp.stat().st_size / (1024 * 1024)
            print(f"  - {cp.name} ({size_mb:.1f} MB)")

        print(f"\ncheckpoint_last.pth exists: {checkpoint_last.exists()}")
        print(f"checkpoint_best.pth exists: {checkpoint_best.exists()}")

        if checkpoint_best.exists():
            print("\nSUCCESS: checkpoint_best.pth was created!")
            print("\nThe race condition fix is working correctly.")
            sys.exit(0)
        else:
            print("\nFAILURE: checkpoint_best.pth was NOT created!")
            print("\nThe fix may not be working. Please check logs for errors.")
            sys.exit(1)
    else:
        print(f"\nERROR: Checkpoint directory does not exist: {checkpoint_dir}")
        print("Please run training first.")
        sys.exit(1)


if __name__ == '__main__':
    main()
