#!/usr/bin/env python3
"""
Export Smoke Test for AquaSOT-3D

Verifies that the export script can successfully export models without errors.
This is a smoke test to catch basic export issues.

Author: AquaSOT-3D Team
Date: 2025-11-24
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

import torch
import tempfile
import pytest

from aquasot3d.models.tracker import AquaSOT3DTracker


class TestExportSmoke:
    """Smoke tests for model export functionality"""

    def test_onnx_wrapper_output_format(self):
        """Test that ONNXExportWrapper converts dict to tuple correctly"""
        from scripts.export import ONNXExportWrapper

        # Create minimal tracker
        tracker = AquaSOT3DTracker(
            rgb_backbone_config={'type': 'vit_rgb', 'pretrained': False},
            sonar_backbone_config={'type': 'vit_sonar', 'pretrained': False},
            fusion_config={'type': 'conditional_dcma'},
            box_head_config={'type': 'BoxHead', 'use_confidence_head': True}
        )
        tracker.eval()

        # Wrap for ONNX export
        wrapper = ONNXExportWrapper(tracker)

        # Verify wrapper is in eval mode
        assert not wrapper.training, "Wrapper should be in eval mode"

        # Test inputs
        B = 1
        rgb_template = torch.randn(B, 3, 256, 256)
        rgb_search = torch.randn(B, 3, 512, 512)
        sonar_template = torch.randn(B, 1, 256, 256)
        sonar_search = torch.randn(B, 1, 512, 512)

        # Forward pass
        with torch.no_grad():
            outputs = wrapper(rgb_template, rgb_search, sonar_template, sonar_search)

        # Verify output is tuple with 4 elements
        assert isinstance(outputs, tuple), "Output should be tuple"
        assert len(outputs) == 4, "Output should have 4 elements"

        rgb_boxes, sonar_boxes, rgb_confidence, sonar_confidence = outputs

        # Verify shapes
        assert rgb_boxes.shape == (B, 4), f"Expected rgb_boxes shape (1, 4), got {rgb_boxes.shape}"
        assert sonar_boxes.shape == (B, 4), f"Expected sonar_boxes shape (1, 4), got {sonar_boxes.shape}"
        assert rgb_confidence.shape == (B,), f"Expected rgb_confidence shape (1,), got {rgb_confidence.shape}"
        assert sonar_confidence.shape == (B,), f"Expected sonar_confidence shape (1,), got {sonar_confidence.shape}"

        # Verify device matching
        assert rgb_boxes.device == rgb_template.device, "Device should match input"
        assert rgb_boxes.dtype == rgb_template.dtype, "Dtype should match input"

    def test_onnx_export_basic(self):
        """Basic smoke test for ONNX export (no actual file write)"""
        from scripts.export import ONNXExportWrapper

        # Create minimal tracker
        tracker = AquaSOT3DTracker(
            rgb_backbone_config={'type': 'vit_rgb', 'pretrained': False},
            sonar_backbone_config={'type': 'vit_sonar', 'pretrained': False},
            fusion_config={'type': 'conditional_dcma'},
            box_head_config={'type': 'BoxHead', 'use_confidence_head': True}
        )
        tracker.eval()

        # Wrap model
        wrapper = ONNXExportWrapper(tracker)

        # Test with dummy inputs (use default sizes: template=256, search=512)
        B = 1
        rgb_template = torch.randn(B, 3, 256, 256)
        rgb_search = torch.randn(B, 3, 512, 512)
        sonar_template = torch.randn(B, 1, 256, 256)
        sonar_search = torch.randn(B, 1, 512, 512)

        # Just verify forward pass works (no actual ONNX export due to model size)
        with torch.no_grad():
            outputs = wrapper(rgb_template, rgb_search, sonar_template, sonar_search)

        assert isinstance(outputs, tuple) and len(outputs) == 4
        print("Export smoke test passed: ONNXExportWrapper produces correct output format")

    def test_load_model_config_extraction(self):
        """Test that load_model correctly extracts configs from checkpoint"""
        from scripts.export import load_model
        import tempfile

        # Create minimal checkpoint
        tracker = AquaSOT3DTracker(
            rgb_backbone_config={'type': 'vit_rgb', 'pretrained': False},
            sonar_backbone_config={'type': 'vit_sonar', 'pretrained': False},
            fusion_config={'type': 'conditional_dcma'},
            box_head_config={'type': 'BoxHead', 'use_confidence_head': True}
        )

        # Create checkpoint with config
        checkpoint = {
            'model_state_dict': tracker.state_dict(),
            'config': {
                'model': {
                    'backbone': {
                        'rgb': {'type': 'vit_rgb', 'pretrained': False},
                        'sonar': {'type': 'vit_sonar', 'pretrained': False}
                    },
                    'fusion': {'type': 'conditional_dcma'},
                    'box_head': {'type': 'BoxHead', 'use_confidence_head': True}
                }
            }
        }

        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp:
            torch.save(checkpoint, tmp.name)
            tmp_path = tmp.name

        try:
            # Load model from checkpoint
            loaded_model = load_model(tmp_path, device='cpu')

            # Verify model loads successfully
            assert loaded_model is not None
            assert isinstance(loaded_model, AquaSOT3DTracker)
            print("Export smoke test passed: load_model correctly extracts configs")
        finally:
            # Clean up
            Path(tmp_path).unlink()


def run_export_smoke_tests():
    """Run export smoke tests"""
    import subprocess

    result = subprocess.run(
        ['python3', '-m', 'pytest', __file__, '-v', '--tb=short'],
        capture_output=True,
        text=True,
        cwd=project_root
    )

    print(result.stdout)
    if result.stderr:
        print(result.stderr)

    return result.returncode == 0


if __name__ == '__main__':
    success = run_export_smoke_tests()
    sys.exit(0 if success else 1)
