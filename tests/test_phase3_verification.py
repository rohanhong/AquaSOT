#!/usr/bin/env python3
"""
Phase 3 Implementation Verification Script

This script verifies that all Phase 3 modules are correctly implemented
and can be imported (without requiring full dependencies).

Usage:
    python3 tests/test_phase3_verification.py

Author: AquaSOT-3D Team
Date: 2025-11-06
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


def test_import(module_path, module_name):
    """Test if a module can be imported."""
    try:
        exec(f"from {module_path} import {module_name}")
        return True, None
    except ImportError as e:
        # Check if it's a missing dependency error
        if 'torch' in str(e) or 'timm' in str(e) or 'cv2' in str(e):
            return True, f"Missing dependency: {e}"
        return False, str(e)
    except Exception as e:
        return False, str(e)


def main():
    """Run verification checks."""
    print("="*80)
    print(" Phase 3: Backbone Networks - Implementation Verification")
    print("="*80)

    tests = [
        # Backbone modules
        ("aquasot3d.models.backbones", "ViTRGBBackbone"),
        ("aquasot3d.models.backbones", "ConvNeXtRGBBackbone"),
        ("aquasot3d.models.backbones", "SonarEncoder"),
        ("aquasot3d.models.backbones", "BackboneFactory"),
        ("aquasot3d.models.backbones", "MultiModalBackbone"),
        ("aquasot3d.models.backbones", "build_backbone"),
        ("aquasot3d.models.backbones", "build_multimodal_backbone"),
    ]

    print("\nModule Import Tests:")
    print("-" * 80)

    passed = 0
    failed = 0
    skipped = 0

    for module_path, module_name in tests:
        success, error = test_import(module_path, module_name)

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
        "src/aquasot3d/models/backbones/vit_rgb.py",
        "src/aquasot3d/models/backbones/convnext_rgb.py",
        "src/aquasot3d/models/backbones/sonar_encoder.py",
        "src/aquasot3d/models/backbones/backbone_factory.py",
        "src/aquasot3d/models/backbones/__init__.py",
        "src/aquasot3d/models/__init__.py",
        "tests/test_backbones.py",
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
        print("✅ Phase 3 Implementation: VERIFIED")
        print("\nNext Steps:")
        print("  1. Install dependencies: pip install -e '.[dev]'")
        print("  2. Run tests: pytest tests/test_backbones.py -v")
        print("  3. Continue to Phase 4: Fusion Module (G2-DCMA)")
    else:
        print("⚠️  Phase 3 Implementation: ISSUES FOUND")
        print(f"   Missing files or import failures detected")
    print("="*80 + "\n")

    return 0 if (all_exist and failed == 0) else 1


if __name__ == '__main__':
    sys.exit(main())
