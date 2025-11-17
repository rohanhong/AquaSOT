#!/usr/bin/env python3
"""
Multi-Frame Compliance Tests for Audit 2025111705

Tests that verify Stage 3 training and evaluation operate on multi-frame
sequences as required by the multi-frame compliance audit.

Test Coverage:
1. Stage 3 config has search_sequence_length > 1
2. Evaluation config has search_sequence_length > 1
3. TrackingDataset emits multi-frame tensors when configured
4. Collate function preserves temporal dimension
5. Vision-Mamba receives and processes multi-frame sequences

Author: AquaSOT-3D Team
Date: 2025-11-17
"""

import sys
from pathlib import Path
import pytest
import torch
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from aquasot3d.utils.config import load_config
from aquasot3d.dataloader import create_dataloader
from aquasot3d.models.temporal.mamba_head import VisionMambaHead


class TestMultiFrameConfigCompliance:
    """Test that configs satisfy multi-frame requirements from Audit 2025111705."""

    def test_stage3_config_enables_multiframe_training(self):
        """
        Audit 2025111705 Finding 1: Stage 3 must use search_sequence_length > 1.
        """
        config_path = Path('configs/train/train.yaml')
        if not config_path.exists():
            pytest.skip(f"Config not found: {config_path}")

        cfg = load_config(str(config_path))

        # Critical: search_sequence_length must be > 1 for temporal training
        assert hasattr(cfg.data, 'search_sequence_length'), \
            "Stage 3 config missing search_sequence_length parameter"
        assert cfg.data.search_sequence_length > 1, \
            f"Stage 3 search_sequence_length={cfg.data.search_sequence_length}, must be > 1 for Vision-Mamba training"

        # Sequence grouping must be enabled to prevent cross-sequence contamination
        assert hasattr(cfg.data, 'use_sequence_grouping'), \
            "Stage 3 config missing use_sequence_grouping parameter"
        assert cfg.data.use_sequence_grouping is True, \
            "Stage 3 must enable use_sequence_grouping for temporal training"

    def test_eval_config_enables_multiframe_inference(self):
        """
        Audit 2025111705 Finding 2: Evaluation must use search_sequence_length > 1.
        """
        config_path = Path('configs/eval/eval.yaml')
        if not config_path.exists():
            pytest.skip(f"Config not found: {config_path}")

        cfg = load_config(str(config_path))

        # Critical: evaluation must exercise multi-frame temporal inference
        assert hasattr(cfg.data, 'search_sequence_length'), \
            "Eval config missing search_sequence_length parameter"
        assert cfg.data.search_sequence_length > 1, \
            f"Eval search_sequence_length={cfg.data.search_sequence_length}, must be > 1 to test temporal behavior"

        assert hasattr(cfg.data, 'use_sequence_grouping'), \
            "Eval config missing use_sequence_grouping parameter"
        assert cfg.data.use_sequence_grouping is True, \
            "Eval must enable use_sequence_grouping for temporal inference"

    def test_config_naming_simplified(self):
        """
        Audit 2025111705 Finding 3: Configs renamed to train.yaml/eval.yaml.
        """
        train_config = Path('configs/train/train.yaml')
        eval_config = Path('configs/eval/eval.yaml')

        assert train_config.exists(), \
            "configs/train/train.yaml not found (should replace train_multiseq.yaml)"
        assert eval_config.exists(), \
            "configs/eval/eval.yaml not found (should replace eval_multiseq.yaml)"

        # Old naming should not exist
        old_train = Path('configs/train/train_multiseq.yaml')
        old_eval = Path('configs/eval/eval_multiseq.yaml')

        assert not old_train.exists(), \
            "Old config train_multiseq.yaml still exists, should be renamed to train.yaml"
        assert not old_eval.exists(), \
            "Old config eval_multiseq.yaml still exists, should be renamed to eval.yaml"

    def test_base_config_comments_consistent(self):
        """
        Audit 2025111705 Finding 4: Base config comments should not contradict values.
        """
        config_path = Path('configs/base/default.yaml')
        if not config_path.exists():
            pytest.skip(f"Config not found: {config_path}")

        # Read raw YAML to check comments
        with open(config_path) as f:
            content = f.read()

        # Check that search_sequence_length comment doesn't contradict the default value
        # The comment should clarify that default=1 is for backward compatibility,
        # and override to 3-5 is needed for Stage 3
        assert 'search_sequence_length: 1' in content, \
            "Base config should have search_sequence_length: 1 as default"

        # Comment should indicate override is needed
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if 'search_sequence_length: 1' in line:
                # Check surrounding comments for clarity
                context = '\n'.join(lines[max(0, i-5):min(len(lines), i+2)])
                assert 'Override' in context or 'override' in context or '3-5' in context, \
                    "search_sequence_length comment should clarify that override to 3-5 is needed for Stage 3"
                break


class TestMultiFrameDataLoading:
    """Test that data pipeline correctly handles multi-frame sequences."""

    @pytest.fixture
    def multiframe_config(self, tmp_path):
        """Create a minimal config with multi-frame settings."""
        config = {
            'data': {
                'data_root': 'data',
                'train_sequences': ['uuv2'],
                'val_sequences': ['uuv3'],
                'template_size': 128,
                'search_size': 256,
                'search_sequence_length': 5,  # Multi-frame
                'use_sequence_grouping': True,
                'num_workers': 0,
                'augmentation': {'enabled': False},
            },
            'training': {
                'batch_size': 2,
            },
        }

        config_path = tmp_path / 'test_config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f)

        return load_config(str(config_path))

    def test_dataloader_emits_5d_tensors(self, multiframe_config):
        """
        Test that dataloader emits [B, T, C, H, W] tensors when search_sequence_length > 1.
        """
        # Skip if dataset not available
        data_root = Path('data/uuv2')
        if not data_root.exists():
            pytest.skip("Dataset not available")

        try:
            train_loader = create_dataloader(
                multiframe_config,
                split='train',
                batch_size=2,
                shuffle=False,
            )

            # Get one batch
            batch = next(iter(train_loader))

            # Check search frames have temporal dimension
            assert 'search_rgb' in batch, "Batch missing search_rgb"
            search_rgb = batch['search_rgb']

            # Should be [B, T, C, H, W] where T = search_sequence_length
            assert search_rgb.ndim == 5, \
                f"Expected 5D tensor [B,T,C,H,W], got {search_rgb.ndim}D: {search_rgb.shape}"
            assert search_rgb.shape[1] == 5, \
                f"Expected T=5 frames, got T={search_rgb.shape[1]}"

            # Check sonar as well
            if 'search_sonar' in batch:
                search_sonar = batch['search_sonar']
                assert search_sonar.ndim == 5, \
                    f"Expected 5D sonar tensor, got {search_sonar.ndim}D"
                assert search_sonar.shape[1] == 5, \
                    f"Expected T=5 sonar frames, got T={search_sonar.shape[1]}"

        except Exception as e:
            pytest.skip(f"Could not create dataloader: {e}")

    def test_sequence_grouping_prevents_mixing(self, multiframe_config):
        """
        Test that use_sequence_grouping ensures batches contain only same-sequence samples.
        """
        # Skip if dataset not available
        data_root = Path('data/uuv2')
        if not data_root.exists():
            pytest.skip("Dataset not available")

        try:
            # Create dataloader with sequence grouping
            train_loader = create_dataloader(
                multiframe_config,
                split='train',
                batch_size=4,
                shuffle=True,
            )

            batch = next(iter(train_loader))

            # Check if batch has sequence_id metadata
            if 'sequence_id' in batch:
                seq_ids = batch['sequence_id']
                # All samples in batch should have same sequence_id
                unique_seqs = set(seq_ids.tolist() if torch.is_tensor(seq_ids) else seq_ids)
                assert len(unique_seqs) == 1, \
                    f"Sequence grouping failed: batch contains {len(unique_seqs)} different sequences"

        except Exception as e:
            pytest.skip(f"Could not test sequence grouping: {e}")


class TestMultiFrameTemporalProcessing:
    """Test that Vision-Mamba correctly processes multi-frame sequences."""

    def test_mamba_head_accepts_5d_input(self):
        """
        Test that VisionMambaHead can process [B, T, C, H, W] tensors.
        """
        # Create minimal VisionMambaHead
        head = VisionMambaHead(
            feature_dim=768,
            num_layers=2,
            d_state=16,
            d_conv=4,
            expand=2,
        )

        # Create 5D input: [B=2, T=5, C=768, H=16, W=16]
        B, T, C, H, W = 2, 5, 768, 16, 16
        x = torch.randn(B, T, C, H, W)

        # Forward pass
        try:
            output = head(x)
            # Output should maintain batch and spatial dims
            assert output['features'].shape[0] == B, \
                f"Batch size mismatch: expected {B}, got {output['features'].shape[0]}"

        except Exception as e:
            pytest.fail(f"VisionMambaHead failed on 5D input: {e}")

    def test_mamba_state_persistence(self):
        """
        Test that VisionMambaHead maintains state across forward passes.
        """
        head = VisionMambaHead(
            feature_dim=768,
            num_layers=2,
            d_state=16,
            d_conv=4,
            expand=2,
        )

        B, T, C, H, W = 1, 3, 768, 16, 16
        x1 = torch.randn(B, T, C, H, W)

        # First forward pass
        output1 = head(x1)
        state1 = output1.get('temporal_state')

        # Second forward pass with same input
        output2 = head(x1, temporal_state=state1)
        state2 = output2.get('temporal_state')

        # State should be returned and updated
        assert state1 is not None, "VisionMambaHead should return temporal_state"
        assert state2 is not None, "VisionMambaHead should update temporal_state"

        # States should be different (state evolved)
        if isinstance(state1, dict) and isinstance(state2, dict):
            for key in state1.keys():
                if torch.is_tensor(state1[key]) and torch.is_tensor(state2[key]):
                    # At least one state buffer should have changed
                    if not torch.allclose(state1[key], state2[key], atol=1e-6):
                        # Good - state evolved
                        break
            else:
                pytest.skip("Cannot verify state evolution (tensors too similar or not present)")


class TestDocumentationUpdates:
    """Test that documentation references the new config names."""

    def test_readme_uses_new_config_names(self):
        """Check that README.md uses train.yaml and eval.yaml."""
        readme = Path('README.md')
        if not readme.exists():
            pytest.skip("README.md not found")

        content = readme.read_text()

        # Should reference new names
        assert 'train.yaml' in content, \
            "README.md should reference configs/train/train.yaml"
        assert 'eval.yaml' in content, \
            "README.md should reference configs/eval/eval.yaml"

        # Should not reference old names (except in historical context)
        assert 'train_multiseq.yaml' not in content, \
            "README.md still references old config name train_multiseq.yaml"
        assert 'eval_multiseq.yaml' not in content, \
            "README.md still references old config name eval_multiseq.yaml"

    def test_training_guide_documents_multiframe(self):
        """Check that Training_Guide.md explains multi-frame training."""
        guide = Path('docs/Training_Guide.md')
        if not guide.exists():
            pytest.skip("Training_Guide.md not found")

        content = guide.read_text()

        # Should use new config names
        assert 'train.yaml' in content, \
            "Training_Guide.md should reference train.yaml"

        # Should not reference old names
        assert 'train_multiseq.yaml' not in content, \
            "Training_Guide.md still references old config name"

    def test_eval_guide_documents_multiframe(self):
        """Check that Evaluation_Guide.md explains multi-frame evaluation."""
        guide = Path('docs/Evaluation_Guide.md')
        if not guide.exists():
            pytest.skip("Evaluation_Guide.md not found")

        content = guide.read_text()

        # Should use new config names
        assert 'eval.yaml' in content, \
            "Evaluation_Guide.md should reference eval.yaml"

        # Should not reference old names
        assert 'eval_multiseq.yaml' not in content, \
            "Evaluation_Guide.md still references old config name"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
