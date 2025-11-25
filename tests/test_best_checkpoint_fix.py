#!/usr/bin/env python3
"""
Test script to verify best checkpoint saving fix
Tests that checkpoint_best.pth is created after first epoch
"""
import sys
import os
from pathlib import Path

# Test 1: Verify checkpoint files
checkpoint_dir = Path('../outputs/train_multiseq/checkpoints')
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
        print("\n✓ SUCCESS: checkpoint_best.pth was created!")
        print("\nThe race condition fix is working correctly.")
        sys.exit(0)
    else:
        print("\n✗ FAILURE: checkpoint_best.pth was NOT created!")
        print("\nThe fix may not be working. Please check logs for errors.")
        sys.exit(1)
else:
    print(f"\n✗ ERROR: Checkpoint directory does not exist: {checkpoint_dir}")
    print("Please run training first.")
    sys.exit(1)
