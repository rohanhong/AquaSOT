"""
Unit tests for core/config.py

Tests the Config and ConfigDict classes for configuration management.
"""

import pytest
import yaml
from pathlib import Path
from aquasot3d.core.config import Config, ConfigDict


class TestConfigDict:
    """Test cases for ConfigDict class."""

    def test_configdict_creation(self):
        """Test ConfigDict initialization."""
        cfg = ConfigDict({'key1': 'value1', 'key2': 2})
        assert cfg['key1'] == 'value1'
        assert cfg['key2'] == 2

    def test_attribute_access(self):
        """Test attribute-style access."""
        cfg = ConfigDict({'model': {'backbone': 'ResNet'}})
        assert cfg.model.backbone == 'ResNet'

    def test_dictionary_access(self):
        """Test dictionary-style access."""
        cfg = ConfigDict({'model': {'backbone': 'ResNet'}})
        assert cfg['model']['backbone'] == 'ResNet'

    def test_nested_dict_conversion(self):
        """Test that nested dicts are converted to ConfigDict."""
        cfg = ConfigDict({'level1': {'level2': {'level3': 'value'}}})
        assert isinstance(cfg.level1, ConfigDict)
        assert isinstance(cfg.level1.level2, ConfigDict)
        assert cfg.level1.level2.level3 == 'value'

    def test_list_with_dicts(self):
        """Test that dicts in lists are converted to ConfigDict."""
        cfg = ConfigDict({'data_items': [{'name': 'item1'}, {'name': 'item2'}]})
        assert isinstance(cfg.data_items[0], ConfigDict)
        assert cfg.data_items[0].name == 'item1'

    def test_set_attribute(self):
        """Test setting attribute."""
        cfg = ConfigDict()
        cfg.new_key = 'new_value'
        assert cfg.new_key == 'new_value'
        assert cfg['new_key'] == 'new_value'

    def test_set_item(self):
        """Test setting item."""
        cfg = ConfigDict()
        cfg['new_key'] = 'new_value'
        assert cfg.new_key == 'new_value'

    def test_delete_attribute(self):
        """Test deleting attribute."""
        cfg = ConfigDict({'key': 'value'})
        del cfg.key
        assert 'key' not in cfg

    def test_merge(self):
        """Test merging dictionaries."""
        cfg1 = ConfigDict({'a': 1, 'b': {'c': 2}})
        cfg2 = {'b': {'d': 3}, 'e': 4}
        cfg1.merge(cfg2)

        assert cfg1.a == 1
        assert cfg1.b.c == 2
        assert cfg1.b.d == 3
        assert cfg1.e == 4

    def test_to_dict(self):
        """Test conversion to regular dict."""
        cfg = ConfigDict({'level1': {'level2': 'value'}})
        regular_dict = cfg.to_dict()

        assert isinstance(regular_dict, dict)
        assert not isinstance(regular_dict, ConfigDict)
        assert isinstance(regular_dict['level1'], dict)
        assert regular_dict['level1']['level2'] == 'value'

    def test_copy(self):
        """Test deep copy."""
        cfg1 = ConfigDict({'a': 1, 'b': {'c': 2}})
        cfg2 = cfg1.copy()

        cfg2.b.c = 3
        assert cfg1.b.c == 2  # Original unchanged
        assert cfg2.b.c == 3

    def test_attribute_error(self):
        """Test AttributeError for non-existent key."""
        cfg = ConfigDict({'key': 'value'})
        with pytest.raises(AttributeError):
            _ = cfg.nonexistent


class TestConfig:
    """Test cases for Config class."""

    def test_config_creation(self):
        """Test Config initialization."""
        cfg_dict = {'model': {'backbone': 'ResNet'}}
        cfg = Config(cfg_dict)

        assert cfg.model.backbone == 'ResNet'

    def test_config_attribute_access(self):
        """Test attribute access."""
        cfg = Config({'model': {'backbone': 'ResNet'}})
        assert cfg.model.backbone == 'ResNet'

    def test_config_item_access(self):
        """Test item access."""
        cfg = Config({'model': {'backbone': 'ResNet'}})
        assert cfg['model']['backbone'] == 'ResNet'

    def test_config_get_with_dot_notation(self):
        """Test get method with dot notation."""
        cfg = Config({'model': {'backbone': {'type': 'ResNet'}}})
        assert cfg.get('model.backbone.type') == 'ResNet'
        assert cfg.get('model.nonexistent', 'default') == 'default'

    def test_config_set_with_dot_notation(self):
        """Test set method with dot notation."""
        cfg = Config({})
        cfg.set('model.backbone.type', 'ResNet')
        assert cfg.model.backbone.type == 'ResNet'

    def test_config_contains(self):
        """Test 'in' operator."""
        cfg = Config({'model': {'backbone': 'ResNet'}})
        assert 'model' in cfg
        assert 'nonexistent' not in cfg

    def test_merge_from_dict(self):
        """Test merging from dictionary."""
        cfg = Config({'a': 1, 'b': {'c': 2}})
        cfg.merge_from_dict({'b': {'d': 3}, 'e': 4})

        assert cfg.a == 1
        assert cfg.b.c == 2
        assert cfg.b.d == 3
        assert cfg.e == 4

    def test_to_dict(self):
        """Test conversion to dictionary."""
        cfg = Config({'model': {'backbone': 'ResNet'}})
        cfg_dict = cfg.to_dict()

        assert isinstance(cfg_dict, dict)
        assert cfg_dict['model']['backbone'] == 'ResNet'

    def test_config_copy(self):
        """Test deep copy."""
        cfg1 = Config({'model': {'depth': 50}})
        cfg2 = cfg1.copy()

        cfg2.model.depth = 101
        assert cfg1.model.depth == 50  # Original unchanged
        assert cfg2.model.depth == 101

    def test_from_file(self, temp_config_file):
        """Test loading config from file."""
        cfg = Config.from_file(temp_config_file)

        assert cfg.model.backbone.type == 'ViT'
        assert cfg.training.batch_size == 32

    def test_from_file_nonexistent(self):
        """Test loading from non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            Config.from_file('nonexistent_config.yaml')

    def test_dump(self, temp_dir):
        """Test saving config to file."""
        cfg = Config({'model': {'backbone': 'ResNet'}})
        output_file = temp_dir / 'output_config.yaml'
        cfg.dump(output_file)

        assert output_file.exists()

        # Load and verify
        loaded_cfg = Config.from_file(output_file)
        assert loaded_cfg.model.backbone == 'ResNet'

    def test_config_inheritance(self, temp_dir):
        """Test config inheritance with _base_ key."""
        # Create base config
        base_cfg = {'model': {'backbone': 'ResNet', 'depth': 50}}
        base_file = temp_dir / 'base.yaml'
        with open(base_file, 'w') as f:
            yaml.dump(base_cfg, f)

        # Create derived config
        derived_cfg = {
            '_base_': 'base.yaml',
            'model': {'depth': 101},  # Override depth
            'training': {'lr': 0.01}  # Add new key
        }
        derived_file = temp_dir / 'derived.yaml'
        with open(derived_file, 'w') as f:
            yaml.dump(derived_cfg, f)

        # Load derived config
        cfg = Config.from_file(derived_file)

        assert cfg.model.backbone == 'ResNet'  # Inherited
        assert cfg.model.depth == 101  # Overridden
        assert cfg.training.lr == 0.01  # New key

    def test_config_str(self):
        """Test string representation."""
        cfg = Config({'model': {'backbone': 'ResNet'}})
        cfg_str = str(cfg)

        assert 'model' in cfg_str
        assert 'backbone' in cfg_str

    def test_config_repr(self):
        """Test repr."""
        cfg = Config({'model': {'backbone': 'ResNet'}}, filename='test.yaml')
        repr_str = repr(cfg)

        assert 'Config' in repr_str
        assert 'test.yaml' in repr_str


class TestConfigIntegration:
    """Integration tests for Config system."""

    def test_complex_config_workflow(self, temp_dir):
        """Test complete workflow with complex config."""
        # Create config
        cfg_dict = {
            'model': {
                'backbone': {
                    'type': 'ViT',
                    'depth': 12,
                    'embed_dim': 768
                },
                'fusion': {
                    'type': 'G2DCMA',
                    'hidden_dim': 256,
                    'num_heads': 8
                },
                'heads': [
                    {'type': 'BoxHead', 'num_classes': 1},
                    {'type': 'FusionHead', 'hidden_dim': 256}
                ]
            },
            'training': {
                'optimizer': {
                    'type': 'AdamW',
                    'lr': 2e-4,
                    'weight_decay': 0.01
                },
                'scheduler': {
                    'type': 'CosineAnnealingLR',
                    'T_max': 60
                }
            }
        }

        cfg = Config(cfg_dict)

        # Test access
        assert cfg.model.backbone.type == 'ViT'
        assert cfg.model.heads[0].type == 'BoxHead'
        assert cfg.training.optimizer.lr == 2e-4

        # Test modification
        cfg.set('training.optimizer.lr', 1e-4)
        assert cfg.training.optimizer.lr == 1e-4

        # Test save and load
        config_file = temp_dir / 'complex_config.yaml'
        cfg.dump(config_file)
        loaded_cfg = Config.from_file(config_file)

        assert loaded_cfg.model.fusion.num_heads == 8
        assert loaded_cfg.training.scheduler.T_max == 60


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
