"""
Tests for OSTrack-style configuration system.

This module tests the new configuration system following OSTrack patterns:
- Default values defined in Python (config.py)
- YAML files only override existing parameters
- Validation at load time for undefined parameters
"""

import pytest
import tempfile
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'lib'))

from lib.config.aquasot import cfg, update_config_from_file, get_config, gen_config


class TestConfigDefaults:
    """Test that all default values are properly defined."""

    def test_model_backbone_rgb_defaults(self):
        """Test RGB backbone default configuration."""
        assert cfg.MODEL.BACKBONE.RGB.TYPE == "vit_base_patch16_224"
        assert cfg.MODEL.BACKBONE.RGB.STRIDE == 16
        assert cfg.MODEL.BACKBONE.RGB.IN_CHANNELS == 3
        assert cfg.MODEL.BACKBONE.RGB.EMBED_DIM == 768

    def test_model_backbone_sonar_defaults(self):
        """Test Sonar backbone default configuration."""
        assert cfg.MODEL.BACKBONE.SONAR.TYPE == "vit_base_patch16_224"
        assert cfg.MODEL.BACKBONE.SONAR.IN_CHANNELS == 1
        assert cfg.MODEL.BACKBONE.SONAR.EMBED_DIM == 768

    def test_model_fusion_defaults(self):
        """Test fusion module default configuration."""
        assert cfg.MODEL.FUSION.TYPE == "dcma"
        assert cfg.MODEL.FUSION.NUM_HEADS == 8
        assert cfg.MODEL.FUSION.NUM_LAYERS == 2

    def test_model_head_defaults(self):
        """Test head default configuration."""
        assert cfg.MODEL.HEAD.TYPE == "CENTER"
        assert cfg.MODEL.HEAD.NUM_CHANNELS == 256

    def test_train_defaults(self):
        """Test training default configuration."""
        assert cfg.TRAIN.LR == 0.0001
        assert cfg.TRAIN.BATCH_SIZE == 2
        assert cfg.TRAIN.EPOCH == 100
        assert cfg.TRAIN.GRAD_CLIP_NORM == 0.1
        assert cfg.TRAIN.BACKBONE_MULTIPLIER == 0.1

    def test_train_loss_weights(self):
        """Test loss weight defaults."""
        assert cfg.TRAIN.GIOU_WEIGHT == 2.0
        assert cfg.TRAIN.L1_WEIGHT == 5.0
        assert cfg.TRAIN.FOCAL_WEIGHT == 1.0

    def test_data_defaults(self):
        """Test data default configuration."""
        assert cfg.DATA.ROOT == "data"
        assert len(cfg.DATA.TRAIN.SEQUENCES) == 9
        assert len(cfg.DATA.VAL.SEQUENCES) == 3
        assert cfg.DATA.SEARCH.SIZE == 256
        assert cfg.DATA.TEMPLATE.SIZE == 128

    def test_data_normalization(self):
        """Test normalization defaults."""
        assert cfg.DATA.RGB_MEAN == [0.485, 0.456, 0.406]
        assert cfg.DATA.RGB_STD == [0.229, 0.224, 0.225]
        assert cfg.DATA.SONAR_MEAN == [0.5]
        assert cfg.DATA.SONAR_STD == [0.5]

    def test_test_defaults(self):
        """Test evaluation default configuration."""
        assert cfg.TEST.TEMPLATE_SIZE == 128
        assert cfg.TEST.SEARCH_SIZE == 256
        assert cfg.TEST.TEMPLATE_FACTOR == 2.0
        assert cfg.TEST.SEARCH_FACTOR == 4.0

    def test_system_defaults(self):
        """Test system default configuration."""
        assert cfg.SYSTEM.SEED == 42
        assert cfg.SYSTEM.DEVICE == "cuda"


class TestConfigUpdate:
    """Test configuration update from YAML files."""

    def test_update_simple_value(self):
        """Test updating a simple value from YAML."""
        yaml_content = """
TRAIN:
  LR: 0.001
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()

            # Store original value
            original_lr = cfg.TRAIN.LR

            try:
                update_config_from_file(f.name)
                assert cfg.TRAIN.LR == 0.001
            finally:
                # Restore original value
                cfg.TRAIN.LR = original_lr
                os.unlink(f.name)

    def test_update_nested_value(self):
        """Test updating a nested value from YAML."""
        yaml_content = """
MODEL:
  BACKBONE:
    RGB:
      EMBED_DIM: 512
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()

            original_dim = cfg.MODEL.BACKBONE.RGB.EMBED_DIM

            try:
                update_config_from_file(f.name)
                assert cfg.MODEL.BACKBONE.RGB.EMBED_DIM == 512
            finally:
                cfg.MODEL.BACKBONE.RGB.EMBED_DIM = original_dim
                os.unlink(f.name)

    def test_undefined_key_raises_error(self):
        """Test that undefined keys raise ValueError."""
        yaml_content = """
UNDEFINED_KEY: value
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()

            try:
                with pytest.raises(ValueError) as exc_info:
                    update_config_from_file(f.name)
                assert "does not exist in config.py" in str(exc_info.value)
            finally:
                os.unlink(f.name)

    def test_undefined_nested_key_raises_error(self):
        """Test that undefined nested keys raise ValueError."""
        yaml_content = """
MODEL:
  UNDEFINED_SECTION:
    KEY: value
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()

            try:
                with pytest.raises(ValueError) as exc_info:
                    update_config_from_file(f.name)
                assert "does not exist in config.py" in str(exc_info.value)
            finally:
                os.unlink(f.name)


class TestConfigGeneration:
    """Test configuration file generation."""

    def test_gen_config_creates_valid_yaml(self):
        """Test that gen_config creates a valid YAML file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            gen_config(f.name)

            # Read back and verify it's valid YAML
            import yaml
            with open(f.name) as rf:
                loaded = yaml.safe_load(rf)

            assert 'MODEL' in loaded
            assert 'TRAIN' in loaded
            assert 'DATA' in loaded
            assert loaded['MODEL']['BACKBONE']['RGB']['TYPE'] == cfg.MODEL.BACKBONE.RGB.TYPE

            os.unlink(f.name)


class TestActualConfigFiles:
    """Test loading actual experiment config files."""

    @pytest.fixture
    def experiments_dir(self):
        return project_root / 'experiments' / 'aquasot'

    def test_load_main_config(self, experiments_dir):
        """Test loading the main training config."""
        config_path = experiments_dir / 'vitb_256_dual.yaml'
        if config_path.exists():
            # Make a copy of original values
            original_lr = cfg.TRAIN.LR
            original_epoch = cfg.TRAIN.EPOCH

            try:
                update_config_from_file(str(config_path))
                # Config should load without errors
                assert True
            finally:
                # Restore
                cfg.TRAIN.LR = original_lr
                cfg.TRAIN.EPOCH = original_epoch
        else:
            pytest.skip(f"Config file not found: {config_path}")

    def test_load_ablation_rgb_only(self, experiments_dir):
        """Test loading RGB-only ablation config."""
        config_path = experiments_dir / 'ablation_rgb_only.yaml'
        if config_path.exists():
            update_config_from_file(str(config_path))
            assert cfg.DATA.MODALITY_DROPOUT.ENABLED == True
            assert cfg.DATA.MODALITY_DROPOUT.P_DROP_SONAR == 1.0
        else:
            pytest.skip(f"Config file not found: {config_path}")


class TestConfigIntegrity:
    """Test configuration integrity and consistency."""

    def test_template_smaller_than_search(self):
        """Template size should be smaller than search size."""
        assert cfg.DATA.TEMPLATE.SIZE < cfg.DATA.SEARCH.SIZE

    def test_factor_relationship(self):
        """Template factor should be smaller than search factor."""
        assert cfg.DATA.TEMPLATE.FACTOR < cfg.DATA.SEARCH.FACTOR

    def test_all_train_sequences_defined(self):
        """All training sequences should be defined."""
        expected_train = ["uuv1", "uuv2", "uuv6", "uuv7", "uuv8", "uuv9", "uuv10", "uuv11", "uuv12"]
        assert set(cfg.DATA.TRAIN.SEQUENCES) == set(expected_train)

    def test_all_val_sequences_defined(self):
        """All validation sequences should be defined."""
        expected_val = ["uuv3", "uuv4", "uuv5"]
        assert set(cfg.DATA.VAL.SEQUENCES) == set(expected_val)

    def test_no_sequence_overlap(self):
        """Train and val sequences should not overlap."""
        train_set = set(cfg.DATA.TRAIN.SEQUENCES)
        val_set = set(cfg.DATA.VAL.SEQUENCES)
        assert len(train_set & val_set) == 0, "Train and val sequences should not overlap"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
