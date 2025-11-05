"""
Unit tests for core/registry.py

Tests the Registry system for component registration and instantiation.
"""

import pytest
from aquasot3d.core.registry import (
    Registry,
    BACKBONES,
    FUSION_MODULES,
    build_from_cfg
)


class TestRegistry:
    """Test cases for Registry class."""

    def test_registry_creation(self):
        """Test registry initialization."""
        reg = Registry('test_registry')
        assert reg.name == 'test_registry'
        assert len(reg) == 0
        assert isinstance(reg.module_dict, dict)

    def test_register_module_decorator(self):
        """Test registering module using decorator."""
        reg = Registry('test')

        @reg.register_module()
        class DummyModule:
            def __init__(self, param1=10):
                self.param1 = param1

        assert 'DummyModule' in reg
        assert reg.get('DummyModule') == DummyModule

    def test_register_module_with_name(self):
        """Test registering module with custom name."""
        reg = Registry('test')

        @reg.register_module(name='custom_name')
        class DummyModule:
            pass

        assert 'custom_name' in reg
        assert reg.get('custom_name') == DummyModule

    def test_register_module_direct(self):
        """Test direct module registration."""
        reg = Registry('test')

        class DummyModule:
            pass

        reg.register_module(module=DummyModule)
        assert 'DummyModule' in reg

    def test_register_duplicate_raises_error(self):
        """Test that registering duplicate name raises error."""
        reg = Registry('test')

        @reg.register_module()
        class DummyModule:
            pass

        with pytest.raises(KeyError):
            @reg.register_module()
            class DummyModule:  # Same name
                pass

    def test_register_duplicate_with_force(self):
        """Test that force=True allows overriding."""
        reg = Registry('test')

        @reg.register_module()
        class DummyModule:
            value = 1

        @reg.register_module(force=True)
        class DummyModule:  # Override
            value = 2

        module_class = reg.get('DummyModule')
        assert module_class.value == 2

    def test_build_module_from_cfg(self):
        """Test building module from config dict."""
        reg = Registry('test')

        @reg.register_module()
        class DummyModule:
            def __init__(self, param1, param2=20):
                self.param1 = param1
                self.param2 = param2

        cfg = dict(type='DummyModule', param1=10, param2=30)
        module = reg.build(cfg)

        assert isinstance(module, DummyModule)
        assert module.param1 == 10
        assert module.param2 == 30

    def test_build_with_kwargs(self):
        """Test building module with additional kwargs."""
        reg = Registry('test')

        @reg.register_module()
        class DummyModule:
            def __init__(self, param1, param2=20):
                self.param1 = param1
                self.param2 = param2

        cfg = dict(type='DummyModule', param1=10)
        module = reg.build(cfg, param2=40)

        assert module.param1 == 10
        assert module.param2 == 40  # kwargs override cfg

    def test_build_with_class_type(self):
        """Test building with class directly instead of string."""
        reg = Registry('test')

        class DummyModule:
            def __init__(self, param1):
                self.param1 = param1

        cfg = dict(type=DummyModule, param1=10)
        module = reg.build(cfg)

        assert isinstance(module, DummyModule)
        assert module.param1 == 10

    def test_build_unregistered_raises_error(self):
        """Test that building unregistered module raises error."""
        reg = Registry('test')
        cfg = dict(type='NonExistentModule')

        with pytest.raises(KeyError):
            reg.build(cfg)

    def test_build_without_type_raises_error(self):
        """Test that config without 'type' raises error."""
        reg = Registry('test')
        cfg = dict(param1=10)  # Missing 'type'

        with pytest.raises(KeyError):
            reg.build(cfg)

    def test_build_invalid_cfg_type_raises_error(self):
        """Test that non-dict config raises error."""
        reg = Registry('test')

        with pytest.raises(TypeError):
            reg.build("invalid_config")

    def test_registry_repr(self):
        """Test registry string representation."""
        reg = Registry('test')

        @reg.register_module()
        class Module1:
            pass

        @reg.register_module()
        class Module2:
            pass

        repr_str = repr(reg)
        assert 'test' in repr_str
        assert 'Module1' in repr_str
        assert 'Module2' in repr_str


class TestPredefinedRegistries:
    """Test predefined registry instances."""

    def test_backbones_registry_exists(self):
        """Test that BACKBONES registry is available."""
        assert BACKBONES.name == 'backbone'

    def test_fusion_modules_registry_exists(self):
        """Test that FUSION_MODULES registry is available."""
        assert FUSION_MODULES.name == 'fusion'

    def test_registries_are_independent(self):
        """Test that different registries don't interfere."""
        @BACKBONES.register_module()
        class TestBackbone:
            pass

        @FUSION_MODULES.register_module()
        class TestFusion:
            pass

        assert 'TestBackbone' in BACKBONES
        assert 'TestBackbone' not in FUSION_MODULES
        assert 'TestFusion' in FUSION_MODULES
        assert 'TestFusion' not in BACKBONES


class TestBuildFromCfg:
    """Test build_from_cfg helper function."""

    def test_build_from_cfg_basic(self):
        """Test basic build_from_cfg usage."""
        reg = Registry('test')

        @reg.register_module()
        class DummyModule:
            def __init__(self, param1):
                self.param1 = param1

        cfg = dict(type='DummyModule', param1=10)
        module = build_from_cfg(cfg, reg)

        assert isinstance(module, DummyModule)
        assert module.param1 == 10

    def test_build_from_cfg_with_default_args(self):
        """Test build_from_cfg with default arguments."""
        reg = Registry('test')

        @reg.register_module()
        class DummyModule:
            def __init__(self, param1, param2=20):
                self.param1 = param1
                self.param2 = param2

        cfg = dict(type='DummyModule', param1=10)
        default_args = dict(param2=30)
        module = build_from_cfg(cfg, reg, default_args)

        assert module.param1 == 10
        assert module.param2 == 30

    def test_build_from_cfg_invalid_registry(self):
        """Test build_from_cfg with invalid registry type."""
        cfg = dict(type='DummyModule')

        with pytest.raises(TypeError):
            build_from_cfg(cfg, "not_a_registry")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
