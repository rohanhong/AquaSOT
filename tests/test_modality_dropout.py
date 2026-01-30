"""
Tests for modality dropout utility (testing-time only).

Tests the apply_modality_dropout function for:
1. Correct behavior when disabled (backward compatibility)
2. Correct behavior for RGB-only ablation (drop sonar)
3. Correct behavior for Sonar-only ablation (drop RGB)
4. Device and dtype preservation
5. Integration with ablation config files
"""

import torch
import pytest
from easydict import EasyDict as edict

from lib.utils.modality_dropout import apply_modality_dropout


class TestModalityDropoutDisabled:
    """Test modality dropout when ENABLED=False (default configuration)."""

    @pytest.fixture
    def cfg_disabled(self):
        """Config with dropout disabled (default)."""
        cfg = edict()
        cfg.DATA = edict()
        cfg.DATA.MODALITY_DROPOUT = edict()
        cfg.DATA.MODALITY_DROPOUT.ENABLED = False
        cfg.DATA.MODALITY_DROPOUT.P_DROP_RGB = 0.0
        cfg.DATA.MODALITY_DROPOUT.P_DROP_SONAR = 0.0
        return cfg

    @pytest.fixture
    def sample_tensors(self):
        """Create sample RGB and Sonar tensors."""
        rgb = torch.randn(2, 3, 256, 256)  # [B, C, H, W]
        sonar = torch.randn(2, 1, 256, 256)
        return rgb, sonar

    def test_disabled_returns_identical_tensors(self, sample_tensors, cfg_disabled):
        """When disabled, tensors should be unchanged (backward compatible)."""
        rgb, sonar = sample_tensors
        rgb_out, sonar_out, info = apply_modality_dropout(rgb, sonar, cfg_disabled)

        assert torch.equal(rgb, rgb_out), "RGB should be unchanged when disabled"
        assert torch.equal(sonar, sonar_out), "Sonar should be unchanged when disabled"

    def test_disabled_returns_correct_info(self, sample_tensors, cfg_disabled):
        """When disabled, dropout_info should indicate nothing dropped."""
        rgb, sonar = sample_tensors
        _, _, info = apply_modality_dropout(rgb, sonar, cfg_disabled)

        assert info['rgb_dropped'] is False
        assert info['sonar_dropped'] is False

    def test_disabled_works_with_different_shapes(self, cfg_disabled):
        """Disabled dropout should work with various tensor shapes."""
        # Single sample
        rgb = torch.randn(1, 3, 128, 128)
        sonar = torch.randn(1, 1, 128, 128)
        rgb_out, sonar_out, _ = apply_modality_dropout(rgb, sonar, cfg_disabled)
        assert torch.equal(rgb, rgb_out)
        assert torch.equal(sonar, sonar_out)

        # Larger batch
        rgb = torch.randn(8, 3, 256, 256)
        sonar = torch.randn(8, 1, 256, 256)
        rgb_out, sonar_out, _ = apply_modality_dropout(rgb, sonar, cfg_disabled)
        assert torch.equal(rgb, rgb_out)
        assert torch.equal(sonar, sonar_out)


class TestRGBOnlyAblation:
    """Test RGB-only ablation (drop sonar, P_DROP_SONAR=1.0)."""

    @pytest.fixture
    def cfg_rgb_only(self):
        """Config for RGB-only ablation."""
        cfg = edict()
        cfg.DATA = edict()
        cfg.DATA.MODALITY_DROPOUT = edict()
        cfg.DATA.MODALITY_DROPOUT.ENABLED = True
        cfg.DATA.MODALITY_DROPOUT.P_DROP_RGB = 0.0
        cfg.DATA.MODALITY_DROPOUT.P_DROP_SONAR = 1.0
        return cfg

    @pytest.fixture
    def sample_tensors(self):
        """Create sample RGB and Sonar tensors with non-zero values."""
        rgb = torch.randn(2, 3, 256, 256) + 1.0  # Ensure non-zero
        sonar = torch.randn(2, 1, 256, 256) + 1.0
        return rgb, sonar

    def test_rgb_only_keeps_rgb_unchanged(self, sample_tensors, cfg_rgb_only):
        """RGB-only mode should keep RGB tensor unchanged."""
        rgb, sonar = sample_tensors
        rgb_out, _, _ = apply_modality_dropout(rgb, sonar, cfg_rgb_only)

        assert torch.equal(rgb, rgb_out), "RGB should be unchanged in RGB-only mode"

    def test_rgb_only_zeros_sonar(self, sample_tensors, cfg_rgb_only):
        """RGB-only mode should zero out sonar tensor."""
        rgb, sonar = sample_tensors
        _, sonar_out, _ = apply_modality_dropout(rgb, sonar, cfg_rgb_only)

        assert torch.all(sonar_out == 0), "Sonar should be all zeros in RGB-only mode"
        assert sonar_out.shape == sonar.shape, "Sonar shape should be preserved"

    def test_rgb_only_returns_correct_info(self, sample_tensors, cfg_rgb_only):
        """RGB-only mode should return correct dropout info."""
        rgb, sonar = sample_tensors
        _, _, info = apply_modality_dropout(rgb, sonar, cfg_rgb_only)

        assert info['rgb_dropped'] is False
        assert info['sonar_dropped'] is True

    def test_rgb_only_deterministic(self, sample_tensors, cfg_rgb_only):
        """RGB-only ablation should be deterministic across multiple calls."""
        rgb, sonar = sample_tensors

        # Run multiple times - should always produce same result
        for _ in range(10):
            _, sonar_out, info = apply_modality_dropout(rgb, sonar, cfg_rgb_only)
            assert torch.all(sonar_out == 0), "Should always zero sonar"
            assert info['sonar_dropped'] is True


class TestSonarOnlyAblation:
    """Test Sonar-only ablation (drop RGB, P_DROP_RGB=1.0)."""

    @pytest.fixture
    def cfg_sonar_only(self):
        """Config for Sonar-only ablation."""
        cfg = edict()
        cfg.DATA = edict()
        cfg.DATA.MODALITY_DROPOUT = edict()
        cfg.DATA.MODALITY_DROPOUT.ENABLED = True
        cfg.DATA.MODALITY_DROPOUT.P_DROP_RGB = 1.0
        cfg.DATA.MODALITY_DROPOUT.P_DROP_SONAR = 0.0
        return cfg

    @pytest.fixture
    def sample_tensors(self):
        """Create sample RGB and Sonar tensors with non-zero values."""
        rgb = torch.randn(2, 3, 256, 256) + 1.0
        sonar = torch.randn(2, 1, 256, 256) + 1.0
        return rgb, sonar

    def test_sonar_only_keeps_sonar_unchanged(self, sample_tensors, cfg_sonar_only):
        """Sonar-only mode should keep Sonar tensor unchanged."""
        rgb, sonar = sample_tensors
        _, sonar_out, _ = apply_modality_dropout(rgb, sonar, cfg_sonar_only)

        assert torch.equal(sonar, sonar_out), "Sonar should be unchanged in Sonar-only mode"

    def test_sonar_only_zeros_rgb(self, sample_tensors, cfg_sonar_only):
        """Sonar-only mode should zero out RGB tensor."""
        rgb, sonar = sample_tensors
        rgb_out, _, _ = apply_modality_dropout(rgb, sonar, cfg_sonar_only)

        assert torch.all(rgb_out == 0), "RGB should be all zeros in Sonar-only mode"
        assert rgb_out.shape == rgb.shape, "RGB shape should be preserved"

    def test_sonar_only_returns_correct_info(self, sample_tensors, cfg_sonar_only):
        """Sonar-only mode should return correct dropout info."""
        rgb, sonar = sample_tensors
        _, _, info = apply_modality_dropout(rgb, sonar, cfg_sonar_only)

        assert info['rgb_dropped'] is True
        assert info['sonar_dropped'] is False

    def test_sonar_only_deterministic(self, sample_tensors, cfg_sonar_only):
        """Sonar-only ablation should be deterministic across multiple calls."""
        rgb, sonar = sample_tensors

        # Run multiple times - should always produce same result
        for _ in range(10):
            rgb_out, _, info = apply_modality_dropout(rgb, sonar, cfg_sonar_only)
            assert torch.all(rgb_out == 0), "Should always zero RGB"
            assert info['rgb_dropped'] is True


class TestDeviceAndDtype:
    """Test that device and dtype are preserved."""

    @pytest.fixture
    def cfg_rgb_only(self):
        """Config for RGB-only ablation."""
        cfg = edict()
        cfg.DATA = edict()
        cfg.DATA.MODALITY_DROPOUT = edict()
        cfg.DATA.MODALITY_DROPOUT.ENABLED = True
        cfg.DATA.MODALITY_DROPOUT.P_DROP_RGB = 0.0
        cfg.DATA.MODALITY_DROPOUT.P_DROP_SONAR = 1.0
        return cfg

    def test_preserves_dtype_float32(self, cfg_rgb_only):
        """Should preserve float32 dtype."""
        rgb = torch.randn(2, 3, 256, 256, dtype=torch.float32)
        sonar = torch.randn(2, 1, 256, 256, dtype=torch.float32)

        rgb_out, sonar_out, _ = apply_modality_dropout(rgb, sonar, cfg_rgb_only)

        assert rgb_out.dtype == torch.float32
        assert sonar_out.dtype == torch.float32

    def test_preserves_dtype_float16(self, cfg_rgb_only):
        """Should preserve float16 dtype."""
        rgb = torch.randn(2, 3, 256, 256, dtype=torch.float16)
        sonar = torch.randn(2, 1, 256, 256, dtype=torch.float16)

        rgb_out, sonar_out, _ = apply_modality_dropout(rgb, sonar, cfg_rgb_only)

        assert rgb_out.dtype == torch.float16
        assert sonar_out.dtype == torch.float16

    @pytest.mark.gpu
    def test_preserves_device_cuda(self, cfg_rgb_only):
        """Should preserve CUDA device."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        rgb = torch.randn(2, 3, 256, 256, device='cuda')
        sonar = torch.randn(2, 1, 256, 256, device='cuda')

        rgb_out, sonar_out, _ = apply_modality_dropout(rgb, sonar, cfg_rgb_only)

        assert rgb_out.device.type == 'cuda'
        assert sonar_out.device.type == 'cuda'


class TestBackwardCompatibility:
    """Test backward compatibility with existing checkpoints."""

    def test_default_config_no_change(self):
        """Default config (from config.py) should result in no change."""
        from lib.config.aquasot.config import cfg

        rgb = torch.randn(2, 3, 256, 256)
        sonar = torch.randn(2, 1, 256, 256)

        rgb_out, sonar_out, info = apply_modality_dropout(rgb, sonar, cfg)

        # Default config has ENABLED=False
        assert torch.equal(rgb, rgb_out), "Default config should not change RGB"
        assert torch.equal(sonar, sonar_out), "Default config should not change Sonar"
        assert info['rgb_dropped'] is False
        assert info['sonar_dropped'] is False

    def test_default_config_enabled_is_false(self):
        """Verify default config has MODALITY_DROPOUT.ENABLED=False."""
        from lib.config.aquasot.config import cfg

        assert cfg.DATA.MODALITY_DROPOUT.ENABLED is False, \
            "Default config must have MODALITY_DROPOUT.ENABLED=False for backward compatibility"
