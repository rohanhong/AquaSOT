"""
Test TemplateSearchAugmentation

Tests for bbox-aware template-search pair augmentation pipeline.
Verifies coordinated transforms and bbox-image synchronization.

Author: AquaSOT-3D Team
Date: 2025-11-19
"""

import unittest
import numpy as np
import torch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from aquasot3d.dataloader.datasets.augmentation import (
    TemplateSearchAugmentation,
    TemplateSearchAugmentationBboxAware,  # Test backward compatibility
)


class TestTemplateSearchAugmentation(unittest.TestCase):
    """Test template-search augmentation pipeline."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_data = {
            'template_rgb': np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8),
            'search_rgb': np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8),
            'template_bbox_rgb': np.array([100, 50, 80, 60], dtype=np.float32),
            'search_bbox_rgb': np.array([200, 100, 90, 70], dtype=np.float32),
            'template_sonar': np.random.randint(0, 256, (376, 682, 1), dtype=np.uint8),
            'search_sonar': np.random.randint(0, 256, (376, 682, 1), dtype=np.uint8),
            'template_bbox_sonar': np.array([150, 80, 70, 50], dtype=np.float32),
            'search_bbox_sonar': np.array([250, 120, 80, 60], dtype=np.float32),
        }

    def test_basic_pipeline_execution(self):
        """Test basic template-search augmentation execution."""
        aug = TemplateSearchAugmentation(
            is_train=True,
            template_config={
                'rgb': {'enable_color_jitter': False},
                'sonar': {'enable_speckle': False}
            },
            search_config={
                'rgb': {'enable_color_jitter': True, 'enable_blur': True},
                'sonar': {'enable_speckle': True}
            },
            enable_flip=True,
            use_logpolar=False,
        )

        result = aug(self.test_data)

        # Check all outputs exist
        self.assertIn('template_rgb', result)
        self.assertIn('search_rgb', result)
        self.assertIn('template_sonar', result)
        self.assertIn('search_sonar', result)
        self.assertIn('template_bbox_rgb', result)
        self.assertIn('search_bbox_rgb', result)
        self.assertIn('template_bbox_sonar', result)
        self.assertIn('search_bbox_sonar', result)

        # Check output types
        self.assertIsInstance(result['template_rgb'], torch.Tensor)
        self.assertIsInstance(result['search_rgb'], torch.Tensor)

    def test_with_logpolar_transform(self):
        """Test template-search augmentation with log-polar transform."""
        aug = TemplateSearchAugmentation(
            is_train=True,
            template_config={'rgb': {}, 'sonar': {}},
            search_config={'rgb': {}, 'sonar': {}},
            enable_flip=False,
            use_logpolar=True,
            logpolar_config={
                'sonar_height': 376,
                'sonar_width': 682,
                'output_height': 256,
                'output_width': 256,
            }
        )

        result = aug(self.test_data)

        # Sonar images should be transformed to 256x256
        self.assertEqual(result['template_sonar'].shape, (1, 256, 256))
        self.assertEqual(result['search_sonar'].shape, (1, 256, 256))

    def test_coordinated_flip(self):
        """Test coordinated horizontal flip between template and search."""
        # Use seed for reproducibility
        aug = TemplateSearchAugmentation(
            is_train=True,
            template_config={'rgb': {}, 'sonar': {}},
            search_config={'rgb': {}, 'sonar': {}},
            enable_flip=True,
            use_logpolar=False,
            seed=42,
        )

        # Run augmentation multiple times with same seed
        result1 = aug(self.test_data.copy())

        # Reset seed
        aug2 = TemplateSearchAugmentation(
            is_train=True,
            template_config={'rgb': {}, 'sonar': {}},
            search_config={'rgb': {}, 'sonar': {}},
            enable_flip=True,
            use_logpolar=False,
            seed=42,
        )
        result2 = aug2(self.test_data.copy())

        # Results should be identical with same seed
        self.assertTrue(torch.allclose(result1['template_rgb'], result2['template_rgb']))
        self.assertTrue(torch.allclose(result1['search_rgb'], result2['search_rgb']))

    def test_different_augmentation_configs(self):
        """Test that template and search can have different augmentation levels."""
        # Template with minimal augmentation
        template_config = {
            'rgb': {
                'enable_color_jitter': False,
                'enable_blur': False,
                'enable_horizontal_flip': False,
            },
            'sonar': {
                'enable_speckle': False,
                'enable_radial_jitter': False,
            }
        }

        # Search with aggressive augmentation
        search_config = {
            'rgb': {
                'enable_color_jitter': True,
                'enable_blur': True,
                'enable_horizontal_flip': True,
            },
            'sonar': {
                'enable_speckle': True,
                'enable_radial_jitter': True,
            }
        }

        aug = TemplateSearchAugmentation(
            is_train=True,
            template_config=template_config,
            search_config=search_config,
            enable_flip=False,  # Disable coordinated flip
            use_logpolar=False,
        )

        result = aug(self.test_data)

        # Should complete without error
        self.assertIn('template_rgb', result)
        self.assertIn('search_rgb', result)

    def test_evaluation_mode(self):
        """Test evaluation mode (is_train=False)."""
        aug = TemplateSearchAugmentation(
            is_train=False,
            template_config={'rgb': {}, 'sonar': {}},
            search_config={'rgb': {}, 'sonar': {}},
            enable_flip=True,  # Should be disabled in eval mode
            use_logpolar=False,
        )

        result = aug(self.test_data)

        # Should still produce valid outputs
        self.assertIn('template_rgb', result)
        self.assertIn('search_rgb', result)
        self.assertIsInstance(result['template_rgb'], torch.Tensor)
        self.assertIsInstance(result['search_rgb'], torch.Tensor)

    def test_bbox_format_consistency(self):
        """Test bbox format remains xywh throughout pipeline."""
        aug = TemplateSearchAugmentation(
            is_train=True,
            template_config={'rgb': {}, 'sonar': {}},
            search_config={'rgb': {}, 'sonar': {}},
            enable_flip=True,
            use_logpolar=True,
        )

        result = aug(self.test_data)

        # All bboxes should have 4 elements (xywh format)
        for key in ['template_bbox_rgb', 'search_bbox_rgb',
                    'template_bbox_sonar', 'search_bbox_sonar']:
            self.assertEqual(len(result[key]), 4)

    def test_backward_compatibility_alias(self):
        """Test backward compatibility with BboxAware suffix."""
        # Should be able to import using old name
        aug = TemplateSearchAugmentationBboxAware(
            is_train=True,
            template_config={'rgb': {}, 'sonar': {}},
            search_config={'rgb': {}, 'sonar': {}},
            use_logpolar=False,
        )

        result = aug(self.test_data)

        # Should work identically
        self.assertIn('template_rgb', result)
        self.assertIn('search_rgb', result)

    def test_none_handling(self):
        """Test handling of None values in data."""
        data_with_none = {
            'template_rgb': self.test_data['template_rgb'],
            'search_rgb': None,  # None search RGB
            'template_bbox_rgb': self.test_data['template_bbox_rgb'],
            'search_bbox_rgb': np.array([0, 0, 0, 0], dtype=np.float32),
            'template_sonar': self.test_data['template_sonar'],
            'search_sonar': self.test_data['search_sonar'],
            'template_bbox_sonar': self.test_data['template_bbox_sonar'],
            'search_bbox_sonar': self.test_data['search_bbox_sonar'],
        }

        aug = TemplateSearchAugmentation(
            is_train=True,
            template_config={'rgb': {}, 'sonar': {}},
            search_config={'rgb': {}, 'sonar': {}},
            enable_flip=False,
            use_logpolar=False,
        )

        result = aug(data_with_none)

        # Should handle None gracefully
        self.assertIsNotNone(result['template_rgb'])
        self.assertIsNone(result['search_rgb'])


class TestTemplateSearchAugmentationIntegration(unittest.TestCase):
    """Integration tests for template-search augmentation."""

    def test_full_pipeline_with_all_features(self):
        """Test complete pipeline with all features enabled."""
        data = {
            'template_rgb': np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8),
            'search_rgb': np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8),
            'template_bbox_rgb': np.array([100, 50, 80, 60], dtype=np.float32),
            'search_bbox_rgb': np.array([200, 100, 90, 70], dtype=np.float32),
            'template_sonar': np.random.randint(0, 256, (376, 682, 1), dtype=np.uint8),
            'search_sonar': np.random.randint(0, 256, (376, 682, 1), dtype=np.uint8),
            'template_bbox_sonar': np.array([150, 80, 70, 50], dtype=np.float32),
            'search_bbox_sonar': np.array([250, 120, 80, 60], dtype=np.float32),
        }

        aug = TemplateSearchAugmentation(
            is_train=True,
            template_config={
                'rgb': {
                    'enable_color_jitter': True,
                    'enable_blur': False,
                },
                'sonar': {
                    'enable_speckle': True,
                    'enable_radial_jitter': False,
                }
            },
            search_config={
                'rgb': {
                    'enable_color_jitter': True,
                    'enable_blur': True,
                },
                'sonar': {
                    'enable_speckle': True,
                    'enable_radial_jitter': True,
                }
            },
            enable_flip=True,
            use_logpolar=True,
            logpolar_config={
                'sonar_height': 376,
                'sonar_width': 682,
                'output_height': 256,
                'output_width': 256,
                'fov_angle': 130.0,
            },
        )

        result = aug(data)

        # Verify all outputs
        self.assertIsInstance(result['template_rgb'], torch.Tensor)
        self.assertIsInstance(result['search_rgb'], torch.Tensor)
        self.assertIsInstance(result['template_sonar'], torch.Tensor)
        self.assertIsInstance(result['search_sonar'], torch.Tensor)

        # Verify shapes
        self.assertEqual(result['template_rgb'].shape[0], 3)  # C, H, W
        self.assertEqual(result['search_rgb'].shape[0], 3)
        self.assertEqual(result['template_sonar'].shape, (1, 256, 256))
        self.assertEqual(result['search_sonar'].shape, (1, 256, 256))

        # Verify bbox format
        for key in ['template_bbox_rgb', 'search_bbox_rgb',
                    'template_bbox_sonar', 'search_bbox_sonar']:
            self.assertEqual(len(result[key]), 4)
            self.assertIsInstance(result[key], (torch.Tensor, np.ndarray))


if __name__ == '__main__':
    unittest.main()
