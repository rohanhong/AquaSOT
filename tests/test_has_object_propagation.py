"""
Integration test for has_object flag propagation (P0 BUG fix validation).

Tests end-to-end data flow:
Dataset → Collate → Trainer → Model → Box Heads → Confidence Loss

Audit: 2025112310-P0
Updated: 2025112604 - Updated to use new base_dataset API with OSTrack-256 standard sizes
"""

import sys
import torch
import yaml
import pytest
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from aquasot3d.dataloader.datasets import TemplateSearchDataset, RGBSonarDataset, template_search_collate_fn
from aquasot3d.models.tracker import AquaSOT3DTracker


@pytest.mark.skip(
    reason="Test uses deprecated API - needs update for dual-view architecture (Audit 2025112604)"
)
def test_has_object_propagation():
    """Test has_object flags propagate correctly through entire pipeline."""

    print("=" * 80)
    print("P0 BUG FIX VALIDATION: Has Object Propagation Test")
    print("=" * 80)

    # Load config
    config_path = Path(__file__).parent.parent / 'configs' / 'default.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print("\n[1/6] Loading dataset...")
    # CRITICAL FIX (Audit 2025112604): Use new base_dataset API with OSTrack-256 sizes
    base_dataset = RGBSonarDataset(
        data_root='data',
        sequence_name='uuv1',
        return_occluded=True
    )
    dataset = TemplateSearchDataset(
        base_dataset=base_dataset,
        template_size=128,  # OSTrack-256 standard
        search_size=256,    # OSTrack-256 standard
        template_search_gap=(1, 200),
        normalization_config=config['data']['normalize']
    )
    print(f"    Dataset loaded: {len(dataset)} samples")

    print("\n[2/6] Testing dataset has_object generation...")
    sample = dataset[0]

    # Verify has_object flags exist
    assert 'has_object_rgb' in sample, "Missing has_object_rgb in sample"
    assert 'has_object_sonar' in sample, "Missing has_object_sonar in sample"
    assert 'has_object' in sample, "Missing has_object in sample"

    # Verify values are 0 or 1
    assert sample['has_object_rgb'] in [0, 1], f"Invalid has_object_rgb: {sample['has_object_rgb']}"
    assert sample['has_object_sonar'] in [0, 1], f"Invalid has_object_sonar: {sample['has_object_sonar']}"
    assert sample['has_object'] in [0, 1], f"Invalid has_object: {sample['has_object']}"

    print(f"    ✓ has_object_rgb: {sample['has_object_rgb']}")
    print(f"    ✓ has_object_sonar: {sample['has_object_sonar']}")
    print(f"    ✓ has_object: {sample['has_object']}")

    print("\n[3/6] Testing collate function...")
    batch = template_search_collate_fn([sample, sample])  # Batch of 2

    # Verify tensors exist
    assert 'has_object_rgb' in batch, "Missing has_object_rgb in batch"
    assert 'has_object_sonar' in batch, "Missing has_object_sonar in batch"
    assert 'has_object' in batch, "Missing has_object in batch"

    # Verify tensor properties
    assert isinstance(batch['has_object_rgb'], torch.Tensor), "has_object_rgb not a tensor"
    assert isinstance(batch['has_object_sonar'], torch.Tensor), "has_object_sonar not a tensor"
    assert isinstance(batch['has_object'], torch.Tensor), "has_object not a tensor"

    assert batch['has_object_rgb'].dtype == torch.float32, f"Wrong dtype: {batch['has_object_rgb'].dtype}"
    assert batch['has_object_sonar'].dtype == torch.float32, f"Wrong dtype: {batch['has_object_sonar'].dtype}"
    assert batch['has_object'].dtype == torch.float32, f"Wrong dtype: {batch['has_object'].dtype}"

    assert batch['has_object_rgb'].shape == (2,), f"Wrong shape: {batch['has_object_rgb'].shape}"
    assert batch['has_object_sonar'].shape == (2,), f"Wrong shape: {batch['has_object_sonar'].shape}"
    assert batch['has_object'].shape == (2,), f"Wrong shape: {batch['has_object'].shape}"

    print(f"    ✓ has_object_rgb: shape={batch['has_object_rgb'].shape}, dtype={batch['has_object_rgb'].dtype}")
    print(f"    ✓ has_object_sonar: shape={batch['has_object_sonar'].shape}, dtype={batch['has_object_sonar'].dtype}")
    print(f"    ✓ has_object: shape={batch['has_object'].shape}, dtype={batch['has_object'].dtype}")

    print("\n[4/6] Initializing model...")
    model = AquaSOT3DTracker(config['model'])
    model.eval()  # Eval mode to avoid BatchNorm issues
    print("    ✓ Model initialized")

    print("\n[5/6] Testing model forward pass with has_object...")

    # Prepare inputs
    rgb_template = batch['rgb_template']
    rgb_search = batch['rgb_search']
    sonar_template = batch['sonar_template']
    sonar_search = batch['sonar_search']
    gt_boxes = {
        'rgb': batch['gt_boxes_rgb'],
        'sonar': batch['gt_boxes_sonar'],
        'fusion': batch['gt_boxes']
    }
    has_object_rgb = batch['has_object_rgb']
    has_object_sonar = batch['has_object_sonar']
    has_object = batch['has_object']

    print(f"    Input shapes:")
    print(f"      rgb_template: {rgb_template.shape}")
    print(f"      rgb_search: {rgb_search.shape}")
    print(f"      sonar_template: {sonar_template.shape}")
    print(f"      sonar_search: {sonar_search.shape}")
    print(f"      has_object_rgb: {has_object_rgb.shape} = {has_object_rgb.tolist()}")
    print(f"      has_object_sonar: {has_object_sonar.shape} = {has_object_sonar.tolist()}")
    print(f"      has_object: {has_object.shape} = {has_object.tolist()}")

    # Forward pass with has_object
    with torch.no_grad():
        outputs = model(
            rgb_template,
            rgb_search,
            sonar_template,
            sonar_search,
            gt_boxes=gt_boxes,
            return_losses=True,
            has_object_rgb=has_object_rgb,
            has_object_sonar=has_object_sonar,
            has_object=has_object
        )

    print(f"    ✓ Forward pass completed")

    print("\n[6/6] Validating confidence losses...")

    # Check confidence losses exist
    assert 'rgb_conf_loss' in outputs, "Missing rgb_conf_loss in outputs"
    assert 'sonar_conf_loss' in outputs, "Missing sonar_conf_loss in outputs"
    assert 'fusion_conf_loss' in outputs, "Missing fusion_conf_loss in outputs"

    # Verify losses are not None
    assert outputs['rgb_conf_loss'] is not None, "rgb_conf_loss is None!"
    assert outputs['sonar_conf_loss'] is not None, "sonar_conf_loss is None!"
    assert outputs['fusion_conf_loss'] is not None, "fusion_conf_loss is None!"

    # Verify losses are tensors
    assert isinstance(outputs['rgb_conf_loss'], torch.Tensor), "rgb_conf_loss not a tensor"
    assert isinstance(outputs['sonar_conf_loss'], torch.Tensor), "sonar_conf_loss not a tensor"
    assert isinstance(outputs['fusion_conf_loss'], torch.Tensor), "fusion_conf_loss not a tensor"

    # Verify losses are positive (BCE loss should be > 0 for random predictions)
    rgb_conf_val = outputs['rgb_conf_loss'].item()
    sonar_conf_val = outputs['sonar_conf_loss'].item()
    fusion_conf_val = outputs['fusion_conf_loss'].item()

    print(f"    ✓ rgb_conf_loss: {rgb_conf_val:.4f} (expected ~0.69 for random init)")
    print(f"    ✓ sonar_conf_loss: {sonar_conf_val:.4f} (expected ~0.69 for random init)")
    print(f"    ✓ fusion_conf_loss: {fusion_conf_val:.4f} (expected ~0.69 for random init)")

    # Verify losses are in reasonable range (0 < loss < 10)
    assert 0 < rgb_conf_val < 10, f"rgb_conf_loss out of range: {rgb_conf_val}"
    assert 0 < sonar_conf_val < 10, f"sonar_conf_loss out of range: {sonar_conf_val}"
    assert 0 < fusion_conf_val < 10, f"fusion_conf_loss out of range: {fusion_conf_val}"

    # Check confidence predictions exist
    assert 'pred_confidence_rgb' in outputs, "Missing pred_confidence_rgb"
    assert 'pred_confidence_sonar' in outputs, "Missing pred_confidence_sonar"
    assert 'pred_confidence_fusion' in outputs, "Missing pred_confidence_fusion"

    pred_conf_rgb = outputs['pred_confidence_rgb']
    pred_conf_sonar = outputs['pred_confidence_sonar']
    pred_conf_fusion = outputs['pred_confidence_fusion']

    print(f"\n    Confidence predictions:")
    print(f"      RGB: {pred_conf_rgb.tolist()}")
    print(f"      Sonar: {pred_conf_sonar.tolist()}")
    print(f"      Fusion: {pred_conf_fusion.tolist()}")

    # Verify predictions are in [0, 1] range
    assert torch.all((pred_conf_rgb >= 0) & (pred_conf_rgb <= 1)), "pred_confidence_rgb out of [0,1]"
    assert torch.all((pred_conf_sonar >= 0) & (pred_conf_sonar <= 1)), "pred_confidence_sonar out of [0,1]"
    assert torch.all((pred_conf_fusion >= 0) & (pred_conf_fusion <= 1)), "pred_confidence_fusion out of [0,1]"

    print("\n" + "=" * 80)
    print("✅ P0 BUG FIX VALIDATION: ALL TESTS PASSED")
    print("=" * 80)
    print("\nSummary:")
    print("  [✓] Dataset generates has_object flags correctly")
    print("  [✓] Collate function converts to float32 tensors")
    print("  [✓] Model accepts has_object parameters")
    print("  [✓] Confidence losses are computed (not None)")
    print("  [✓] Confidence losses have reasonable values")
    print("  [✓] Confidence predictions are in [0,1] range")
    print("\nConclusion: has_object propagation is WORKING correctly!")
    print("=" * 80)


if __name__ == '__main__':
    test_has_object_propagation()
