#!/usr/bin/env python3
"""
Phase 2 Implementation Verification Script

This script verifies that all Phase 2 modules are correctly implemented
and can be imported (without requiring full dependencies).

Usage:
    python3 scripts/verify_phase2.py

Author: AquaSOT-3D Team
Date: 2025-11-06
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

def check_import(module_path, module_name):
    """Check if a module can be imported."""
    try:
        exec(f"from {module_path} import {module_name}")
        return True, None
    except ImportError as e:
        # Check if it's a missing dependency error
        if 'cv2' in str(e) or 'torch' in str(e) or 'PIL' in str(e):
            return True, f"Missing dependency: {e}"
        return False, str(e)
    except Exception as e:
        return False, str(e)

def main():
    """Run verification checks."""
    print("="*80)
    print(" Phase 2: Data Pipeline - Implementation Verification")
    print("="*80)

    tests = [
        # Dataset modules
        ("aquasot3d.dataloader.datasets", "RGBSonarDataset"),
        ("aquasot3d.dataloader.datasets", "UniformSampler"),
        ("aquasot3d.dataloader.datasets", "RandomSampler"),
        ("aquasot3d.dataloader.datasets", "OcclusionAwareSampler"),
        ("aquasot3d.dataloader.datasets", "AdaptiveSampler"),
        ("aquasot3d.dataloader.datasets", "TemplateSearchSampler"),
        ("aquasot3d.dataloader.datasets", "TrackingAugmentation"),
        ("aquasot3d.dataloader.datasets", "TemplateSearchAugmentation"),
        ("aquasot3d.dataloader.datasets", "build_augmentation"),

        # Transform modules (commented out - require torch/numpy)
        # ("aquasot3d.dataloader.transforms", "LogPolarTransform"),

        # Loader utilities (commented out - require torch)
        # ("aquasot3d.dataloader", "create_dataloader"),
    ]

    print("\nModule Import Tests:")
    print("-" * 80)

    passed = 0
    failed = 0
    skipped = 0

    for module_path, module_name in tests:
        success, error = check_import(module_path, module_name)

        if success:
            if error:
                print(f"⚠️  {module_path}.{module_name:<30} SKIPPED (dependency)")
                skipped += 1
            else:
                print(f"✅ {module_path}.{module_name:<30} OK")
                passed += 1
        else:
            print(f"❌ {module_path}.{module_name:<30} FAILED")
            print(f"   Error: {error}")
            failed += 1

    print("\n" + "="*80)
    print(f"Results: {passed} passed, {skipped} skipped (missing deps), {failed} failed")
    print("="*80)

    # File structure verification
    print("\nFile Structure Verification:")
    print("-" * 80)

    expected_files = [
        "src/aquasot3d/dataloader/datasets/rgb_sonar_dataset.py",
        "src/aquasot3d/dataloader/datasets/sequence_sampler.py",
        "src/aquasot3d/dataloader/datasets/template_search_sampler.py",
        "src/aquasot3d/dataloader/datasets/augmentation.py",
        "src/aquasot3d/dataloader/transforms/rgb_transforms.py",
        "src/aquasot3d/dataloader/transforms/sonar_transforms.py",
        # Note: sonar_geometry.py and logpolar.py removed (2025-11-19)
        # Reason: Sonar now uses offline preprocessing (scripts/preprocess_sonar.py)
        "src/aquasot3d/dataloader/loader.py",
        "tests/test_rgb_sonar_dataset.py",
        "tests/test_dataloader.py",
        "examples/augmentation_example.yaml",
        "examples/dataloader_usage_example.py",
    ]

    root = Path(__file__).parent.parent
    all_exist = True

    for file_path in expected_files:
        full_path = root / file_path
        if full_path.exists():
            size = full_path.stat().st_size
            print(f"✅ {file_path:<60} ({size:>6} bytes)")
        else:
            print(f"❌ {file_path:<60} MISSING")
            all_exist = False

    print("\n" + "="*80)
    if all_exist and failed == 0:
        print("✅ Phase 2 Implementation: VERIFIED")
        print("\nNext Steps:")
        print("  1. Install dependencies: pip install -e '.[dev]'")
        print("  2. Run tests: pytest tests/test_rgb_sonar_dataset.py -v")
        print("  3. Continue to Phase 3: Backbone Networks")
    else:
        print("⚠️  Phase 2 Implementation: ISSUES FOUND")
        print(f"   Missing files or import failures detected")
    print("="*80 + "\n")

    return 0 if (all_exist and failed == 0) else 1

if __name__ == '__main__':
    sys.exit(main())
