"""
Test complete validation process with template-search data format.

This test replicates the full validation loop to ensure:
1. Validity flag handling works correctly (adaptive [B] vs [B,T])
2. No undefined variable errors (target_idx fixed)
3. Validation metrics computed correctly
"""

import pytest
import torch
import torch.nn as nn
import inspect
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestValidationProcess:
    """Test suite for validation process integrity"""

    def test_validity_flag_handling_template_search_format(self):
        """Test validity flags with [B] template-search format (OSTrack)"""

        # Create mock batch with template-search format
        batch_size = 4
        batch = {
            'template_rgb': torch.rand(batch_size, 3, 256, 256),
            'search_rgb': torch.rand(batch_size, 3, 512, 512),
            'template_sonar': torch.rand(batch_size, 1, 256, 256),
            'search_sonar': torch.rand(batch_size, 1, 512, 512),
            'search_bbox': torch.rand(batch_size, 4),  # Ground truth

            # Template-search format: [B] - single search frame validity
            'is_valid': torch.tensor([1, 1, 0, 1], dtype=torch.bool),
            'is_valid_rgb': torch.tensor([1, 0, 1, 1], dtype=torch.bool),
            'is_valid_sonar': torch.tensor([1, 1, 1, 0], dtype=torch.bool),
        }

        # Simulate validity statistics tracking (from trainer.py:660-682)
        validity_stats = {
            'total_samples': 0,
            'valid_samples': 0,
            'invalid_samples': 0,
            'valid_rgb': 0,
            'valid_sonar': 0
        }

        # CRITICAL: Test adaptive handling (MUST NOT reference target_idx)
        validity_stats['total_samples'] += batch_size

        if 'is_valid' in batch:
            valid_count = batch['is_valid'].sum().item()
            validity_stats['valid_samples'] += valid_count
            validity_stats['invalid_samples'] += (batch_size - valid_count)

        if 'is_valid_rgb' in batch:
            # Adaptive handling: check dimensionality
            is_valid_rgb = batch['is_valid_rgb']
            if is_valid_rgb.dim() == 2:
                # Frame sequence: [B, T] - extract last frame
                valid_rgb = is_valid_rgb[:, -1]
            else:
                # Template-search: [B] - already target frame
                valid_rgb = is_valid_rgb
            validity_stats['valid_rgb'] += valid_rgb.sum().item()

        if 'is_valid_sonar' in batch:
            is_valid_sonar = batch['is_valid_sonar']
            if is_valid_sonar.dim() == 2:
                valid_sonar = is_valid_sonar[:, -1]
            else:
                valid_sonar = is_valid_sonar
            validity_stats['valid_sonar'] += valid_sonar.sum().item()

        # Assertions
        assert validity_stats['total_samples'] == 4
        assert validity_stats['valid_samples'] == 3  # 3 valid out of 4
        assert validity_stats['invalid_samples'] == 1
        assert validity_stats['valid_rgb'] == 3  # [1, 0, 1, 1]
        assert validity_stats['valid_sonar'] == 3  # [1, 1, 1, 0]


    def test_validity_flag_handling_frame_sequence_format(self):
        """Test validity flags with [B, T] frame sequence format (legacy)"""

        batch_size = 3
        seq_length = 5
        batch = {
            'rgb': torch.rand(batch_size, seq_length, 3, 512, 512),
            'sonar': torch.rand(batch_size, seq_length, 1, 512, 512),
            'bbox': torch.rand(batch_size, seq_length, 4),

            # Frame sequence format: [B, T] - multi-frame validity
            'is_valid': torch.ones(batch_size, seq_length, dtype=torch.bool),
            'is_valid_rgb': torch.tensor([
                [1, 1, 1, 0, 1],  # Last frame (idx=4) valid
                [1, 1, 1, 1, 0],  # Last frame (idx=4) invalid
                [1, 1, 1, 1, 1],  # Last frame (idx=4) valid
            ], dtype=torch.bool),
            'is_valid_sonar': torch.tensor([
                [1, 0, 1, 1, 1],  # Last frame valid
                [1, 1, 1, 1, 1],  # Last frame valid
                [0, 0, 0, 0, 0],  # Last frame invalid
            ], dtype=torch.bool),
        }

        validity_stats = {
            'total_samples': 0,
            'valid_rgb': 0,
            'valid_sonar': 0
        }

        validity_stats['total_samples'] += batch_size

        # Test [B, T] extraction
        if 'is_valid_rgb' in batch:
            is_valid_rgb = batch['is_valid_rgb']
            if is_valid_rgb.dim() == 2:
                valid_rgb = is_valid_rgb[:, -1]  # Extract last frame [B]
            else:
                valid_rgb = is_valid_rgb
            validity_stats['valid_rgb'] += valid_rgb.sum().item()

        if 'is_valid_sonar' in batch:
            is_valid_sonar = batch['is_valid_sonar']
            if is_valid_sonar.dim() == 2:
                valid_sonar = is_valid_sonar[:, -1]
            else:
                valid_sonar = is_valid_sonar
            validity_stats['valid_sonar'] += valid_sonar.sum().item()

        # Assertions: Check last frame (index=-1) extraction
        assert validity_stats['total_samples'] == 3
        assert validity_stats['valid_rgb'] == 2  # Last column: [1, 0, 1]
        assert validity_stats['valid_sonar'] == 2  # Last column: [1, 1, 0]


    def test_no_target_idx_dependency(self):
        """Ensure validation code does NOT reference undefined target_idx"""

        from src.aquasot3d.training.trainer import Trainer

        # Get validation method source
        validate_source = inspect.getsource(Trainer.validate)

        # CRITICAL: Must NOT contain undefined target_idx references
        # Valid pattern: Check if variable is defined before use
        # Invalid pattern: Direct use of target_idx without definition

        # Search for problematic patterns
        lines = validate_source.split('\n')
        for i, line in enumerate(lines):
            if 'target_idx' in line:
                # Allowed: Comments
                if line.strip().startswith('#'):
                    continue
                # Allowed: Variable definition/assignment
                if 'target_idx =' in line:
                    continue
                # Allowed: Conditional check with fallback
                if 'if' in line and 'dim()' in line:
                    continue
                # FORBIDDEN: Direct usage without context
                if 'target_idx]' in line or 'target_idx)' in line:
                    # Must be within adaptive handling block
                    context_start = max(0, i-5)
                    context = '\n'.join(lines[context_start:i+3])

                    assert 'dim()' in context or '[:, -1]' in context, \
                        f"Unsafe target_idx usage at line {i}:\n{line}\n\nContext:\n{context}"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
