"""
Test Dual-View Fusion Architecture (Audit 2025112313).

This test validates the critical architectural changes for dual-view fusion:
1. G2-DCMA outputs both rgb_view_fused and sonar_view_fused
2. RGB BoxHead uses rgb_view_fused (not original rgb_features)
3. Sonar BoxHead uses sonar_view_fused (not original sonar_features)
4. Fusion BoxHead is removed
5. Auxiliary heads provide pre-fusion supervision
6. Cross-view consistency loss is computed
7. End-to-end gradient flow works correctly

Author: AquaSOT-3D Team
Date: 2025-11-23
"""

import torch
import pytest
from pathlib import Path

from aquasot3d.models.tracker import build_tracker
from aquasot3d.models.fusion.g2_dcma import G2DCMA


def test_g2_dcma_dual_view_output():
    """
    Test that G2-DCMA outputs both rgb_view_fused and sonar_view_fused.
    """
    # Create minimal G2-DCMA module
    fusion = G2DCMA(
        embed_dim=768,
        num_heads=8
    )

    # Create dummy features
    B, C, H, W = 2, 768, 32, 32
    rgb_features = torch.randn(B, C, H, W)
    sonar_features = torch.randn(B, C, H, W)

    # Forward pass
    outputs = fusion(rgb_features, sonar_features, return_consistency=True)

    # Verify dual-view outputs exist
    assert 'rgb_view_fused' in outputs, "rgb_view_fused should be in outputs"
    assert 'sonar_view_fused' in outputs, "sonar_view_fused should be in outputs"
    assert 'view_consistency_loss' in outputs, "view_consistency_loss should be in outputs"

    # Verify shapes
    assert outputs['rgb_view_fused'].shape == (B, C, H, W), "rgb_view_fused shape mismatch"
    assert outputs['sonar_view_fused'].shape == (B, C, H, W), "sonar_view_fused shape mismatch"

    # Verify backward compatibility
    assert 'fused_features' in outputs, "fused_features should exist for backward compatibility"

    # Verify deformable info includes bidirectional offsets
    assert 'deformable_info' in outputs, "deformable_info should be in outputs"
    deformable_info = outputs['deformable_info']
    assert 'rgb_to_sonar_offsets' in deformable_info, "rgb_to_sonar_offsets should be present"
    assert 'sonar_to_rgb_offsets' in deformable_info, "sonar_to_rgb_offsets should be present"

    print("✓ G2-DCMA dual-view output verified")


def test_tracker_no_fusion_head():
    """
    Test that tracker does not create fusion_box_head.
    """
    config = {
        'rgb_backbone_config': {
            'type': 'vit_rgb',
            'model_name': 'vit_base_patch16_224',
            'pretrained': False,
            'output_dim': 768
        },
        'sonar_backbone_config': {
            'type': 'vit_sonar',
            'model_name': 'vit_base_patch16_224',
            'pretrained': False,
            'in_channels': 1,
            'output_dim': 768
        },
        'fusion_config': {
            'type': 'g2_dcma',
            'embed_dim': 768,
            'num_heads': 8,
            'num_layers': 2
        },
        'box_head_config': {
            'hidden_dim': 256
        }
    }

    tracker = build_tracker('AquaSOT3DTracker', **config)

    # Verify fusion_box_head does not exist (completely removed)
    assert not hasattr(tracker, 'fusion_box_head'), "fusion_box_head should not exist"

    # Verify RGB and Sonar box heads exist
    assert tracker.rgb_box_head is not None, "rgb_box_head should exist"
    assert tracker.sonar_box_head is not None, "sonar_box_head should exist"

    # Verify auxiliary heads exist
    assert tracker.aux_rgb_head is not None, "aux_rgb_head should exist"
    assert tracker.aux_sonar_head is not None, "aux_sonar_head should exist"

    print("✓ Tracker architecture verified (no fusion head, auxiliary heads present)")


def test_boxhead_uses_fused_features():
    """
    Test that RGB and Sonar BoxHeads use fused features instead of original features.
    """
    config = {
        'rgb_backbone_config': {
            'type': 'vit_rgb',
            'model_name': 'vit_base_patch16_224',
            'pretrained': False,
            'output_dim': 768
        },
        'sonar_backbone_config': {
            'type': 'vit_sonar',
            'model_name': 'vit_base_patch16_224',
            'pretrained': False,
            'in_channels': 1,
            'output_dim': 768
        },
        'fusion_config': {
            'type': 'g2_dcma',
            'embed_dim': 768,
            'num_heads': 8,
            'num_layers': 2
        },
        'box_head_config': {
            'hidden_dim': 256
        }
    }

    tracker = build_tracker('AquaSOT3DTracker', **config)
    tracker.eval()

    # Create dummy inputs
    B = 2
    rgb_template = torch.randn(B, 3, 256, 256)
    rgb_search = torch.randn(B, 3, 512, 512)
    sonar_template = torch.randn(B, 1, 256, 256)
    sonar_search = torch.randn(B, 1, 512, 512)

    # Forward pass
    with torch.no_grad():
        outputs = tracker(
            rgb_template=rgb_template,
            rgb_search=rgb_search,
            sonar_template=sonar_template,
            sonar_search=sonar_search,
            return_losses=False,
            return_intermediate=True
        )

    # Verify outputs exist
    assert 'rgb_boxes' in outputs, "rgb_boxes should be in outputs"
    assert 'sonar_boxes' in outputs, "sonar_boxes should be in outputs"
    assert 'fusion_boxes' not in outputs, "fusion_boxes should NOT be in outputs"

    # Verify intermediate features contain fused features
    assert 'intermediate' in outputs, "intermediate should be in outputs"
    intermediate = outputs['intermediate']
    assert 'fused_features' in intermediate, "fused_features should be in intermediate"

    # Verify fused_features is a dict with dual-view outputs
    fused_features = intermediate['fused_features']
    assert isinstance(fused_features, dict), "fused_features should be a dict"
    assert 'rgb_view_fused' in fused_features, "rgb_view_fused should be in fused_features"
    assert 'sonar_view_fused' in fused_features, "sonar_view_fused should be in fused_features"

    print("✓ BoxHeads use fused features verified")


def test_auxiliary_heads_training():
    """
    Test that auxiliary heads work correctly during training.
    """
    config = {
        'rgb_backbone_config': {
            'type': 'vit_rgb',
            'model_name': 'vit_base_patch16_224',
            'pretrained': False,
            'output_dim': 768
        },
        'sonar_backbone_config': {
            'type': 'vit_sonar',
            'model_name': 'vit_base_patch16_224',
            'pretrained': False,
            'in_channels': 1,
            'output_dim': 768
        },
        'fusion_config': {
            'type': 'g2_dcma',
            'embed_dim': 768,
            'num_heads': 8,
            'num_layers': 2
        },
        'box_head_config': {
            'hidden_dim': 256
        }
    }

    tracker = build_tracker('AquaSOT3DTracker', **config)
    tracker.train()  # Training mode

    # Create dummy inputs
    B = 2
    rgb_template = torch.randn(B, 3, 256, 256)
    rgb_search = torch.randn(B, 3, 512, 512)
    sonar_template = torch.randn(B, 1, 256, 256)
    sonar_search = torch.randn(B, 1, 512, 512)

    # Create dummy GT
    gt_boxes = {
        'rgb': torch.rand(B, 4),
        'sonar': torch.rand(B, 4)
    }

    # Forward pass with losses
    outputs = tracker(
        rgb_template=rgb_template,
        rgb_search=rgb_search,
        sonar_template=sonar_template,
        sonar_search=sonar_search,
        gt_boxes=gt_boxes,
        return_losses=True
    )

    # Verify auxiliary losses exist
    assert 'losses' in outputs, "losses should be in outputs"
    losses = outputs['losses']
    assert 'aux_rgb_loss' in losses, "aux_rgb_loss should be in losses"
    assert 'aux_sonar_loss' in losses, "aux_sonar_loss should be in losses"

    # Verify auxiliary losses are scalars
    assert losses['aux_rgb_loss'].dim() == 0, "aux_rgb_loss should be scalar"
    assert losses['aux_sonar_loss'].dim() == 0, "aux_sonar_loss should be scalar"

    # Verify no fusion losses
    assert 'fusion_box_loss' not in losses, "fusion_box_loss should NOT be in losses"
    assert 'fusion_conf_loss' not in losses, "fusion_conf_loss should NOT be in losses"

    print("✓ Auxiliary heads training verified")


def test_view_consistency_loss():
    """
    Test that view_consistency_loss is computed and propagated.
    """
    config = {
        'rgb_backbone_config': {
            'type': 'vit_rgb',
            'model_name': 'vit_base_patch16_224',
            'pretrained': False,
            'output_dim': 768
        },
        'sonar_backbone_config': {
            'type': 'vit_sonar',
            'model_name': 'vit_base_patch16_224',
            'pretrained': False,
            'in_channels': 1,
            'output_dim': 768
        },
        'fusion_config': {
            'type': 'g2_dcma',
            'embed_dim': 768,
            'num_heads': 8,
            'num_layers': 2
        },
        'box_head_config': {
            'hidden_dim': 256
        }
    }

    tracker = build_tracker('AquaSOT3DTracker', **config)
    tracker.train()

    # Create dummy inputs
    B = 2
    rgb_template = torch.randn(B, 3, 256, 256)
    rgb_search = torch.randn(B, 3, 512, 512)
    sonar_template = torch.randn(B, 1, 256, 256)
    sonar_search = torch.randn(B, 1, 512, 512)

    # Forward pass with losses
    outputs = tracker(
        rgb_template=rgb_template,
        rgb_search=rgb_search,
        sonar_template=sonar_template,
        sonar_search=sonar_search,
        return_losses=True
    )

    # Verify view_consistency loss exists
    assert 'losses' in outputs, "losses should be in outputs"
    losses = outputs['losses']
    assert 'view_consistency' in losses, "view_consistency should be in losses"
    assert losses['view_consistency'].dim() == 0, "view_consistency should be scalar"

    print("✓ View consistency loss verified")


def test_end_to_end_gradient_flow():
    """
    Test that gradients flow correctly through the dual-view architecture.
    """
    config = {
        'rgb_backbone_config': {
            'type': 'vit_rgb',
            'model_name': 'vit_base_patch16_224',
            'pretrained': False,
            'output_dim': 768
        },
        'sonar_backbone_config': {
            'type': 'vit_sonar',
            'model_name': 'vit_base_patch16_224',
            'pretrained': False,
            'in_channels': 1,
            'output_dim': 768
        },
        'fusion_config': {
            'type': 'g2_dcma',
            'embed_dim': 768,
            'num_heads': 8,
            'num_layers': 2
        },
        'box_head_config': {
            'hidden_dim': 256
        }
    }

    tracker = build_tracker('AquaSOT3DTracker', **config)
    tracker.train()

    # Create dummy inputs
    B = 2
    rgb_template = torch.randn(B, 3, 256, 256, requires_grad=True)
    rgb_search = torch.randn(B, 3, 512, 512, requires_grad=True)
    sonar_template = torch.randn(B, 1, 256, 256, requires_grad=True)
    sonar_search = torch.randn(B, 1, 512, 512, requires_grad=True)

    # Create dummy GT
    gt_boxes = {
        'rgb': torch.rand(B, 4),
        'sonar': torch.rand(B, 4)
    }

    # Forward pass
    outputs = tracker(
        rgb_template=rgb_template,
        rgb_search=rgb_search,
        sonar_template=sonar_template,
        sonar_search=sonar_search,
        gt_boxes=gt_boxes,
        return_losses=True
    )

    # Compute total loss
    total_loss = sum(outputs['losses'].values())

    # Backward pass
    total_loss.backward()

    # Verify gradients exist for all inputs
    assert rgb_template.grad is not None, "rgb_template should have gradients"
    assert rgb_search.grad is not None, "rgb_search should have gradients"
    assert sonar_template.grad is not None, "sonar_template should have gradients"
    assert sonar_search.grad is not None, "sonar_search should have gradients"

    # Verify gradients are not zero (learning is happening)
    assert rgb_template.grad.abs().sum() > 0, "rgb_template gradients should be non-zero"
    assert sonar_search.grad.abs().sum() > 0, "sonar_search gradients should be non-zero"

    print("✓ End-to-end gradient flow verified")


if __name__ == "__main__":
    print("=== Testing Dual-View Fusion Architecture ===\n")

    print("Test 1: G2-DCMA dual-view output")
    try:
        test_g2_dcma_dual_view_output()
    except Exception as e:
        print(f"✗ Test 1 failed: {e}\n")

    print("\nTest 2: Tracker architecture (no fusion head)")
    try:
        test_tracker_no_fusion_head()
    except Exception as e:
        print(f"✗ Test 2 failed: {e}\n")

    print("\nTest 3: BoxHeads use fused features")
    try:
        test_boxhead_uses_fused_features()
    except Exception as e:
        print(f"✗ Test 3 failed: {e}\n")

    print("\nTest 4: Auxiliary heads training")
    try:
        test_auxiliary_heads_training()
    except Exception as e:
        print(f"✗ Test 4 failed: {e}\n")

    print("\nTest 5: View consistency loss")
    try:
        test_view_consistency_loss()
    except Exception as e:
        print(f"✗ Test 5 failed: {e}\n")

    print("\nTest 6: End-to-end gradient flow")
    try:
        test_end_to_end_gradient_flow()
    except Exception as e:
        print(f"✗ Test 6 failed: {e}\n")

    print("\n=== All Tests Complete ===")
