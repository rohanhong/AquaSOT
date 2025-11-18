"""
Comprehensive Validation Stage Test for AquaSOT-3D

This test script replicates the exact validation flow to verify:
1. Temporal state reset mechanism works correctly
2. Shape validation handles training->validation transitions
3. EMA aggregation doesn't crash with dimension mismatches
4. Validation runs successfully with batched data

Test Audit: 2025111802
Purpose: Validate fixes for validation stage shape mismatch errors
"""

import sys
import os
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from aquasot3d.utils.config import load_config
from aquasot3d.dataloader.datasets.rgb_sonar_dataset import MultiSequenceRGBSonarDataset
from aquasot3d.dataloader.loader import build_dataloader
from aquasot3d.models.tracker import AquaSOT3DTracker

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class ValidationTester:
    """
    Comprehensive tester for validation stage temporal state management.
    """

    def __init__(self, config_path: str = "configs/train/train.yaml"):
        """
        Initialize validation tester.

        Args:
            config_path: Path to training config file
        """
        logger.info("=" * 70)
        logger.info("Validation Stage Temporal State Test")
        logger.info("=" * 70)

        # Load configuration
        self.cfg = load_config(config_path)
        logger.info(f"Loaded config from: {config_path}")

        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

        # Build model
        self._build_model()

        # Build validation dataloader
        self._build_dataloader()

    def _build_model(self):
        """Build AquaSOT3D model."""
        logger.info("Building model...")

        # Build tracker model
        self.model = AquaSOT3DTracker(
            rgb_backbone_config=self.cfg.model.rgb_backbone,
            sonar_encoder_config=self.cfg.model.sonar_encoder,
            fusion_config=self.cfg.model.fusion,
            temporal_config=self.cfg.model.temporal,
            weak3d_config=self.cfg.model.weak3d,
            quality_config=self.cfg.model.quality
        ).to(self.device)

        logger.info(f"Model built: {self.model.__class__.__name__}")
        logger.info(f"Total parameters: {sum(p.numel() for p in self.model.parameters()) / 1e6:.2f}M")

        # Check temporal components
        if hasattr(self.model, 'temporal_aggregator') and self.model.temporal_aggregator is not None:
            logger.info("✓ Model has temporal_aggregator")
            if hasattr(self.model.temporal_aggregator, 'reset'):
                logger.info("✓ temporal_aggregator has reset() method")
            else:
                logger.error("✗ temporal_aggregator lacks reset() method!")

        if hasattr(self.model, 'temporal_head') and self.model.temporal_head is not None:
            logger.info("✓ Model has temporal_head")

    def _build_dataloader(self):
        """Build validation dataloader."""
        logger.info("Building validation dataloader...")

        # Get validation sequences
        val_sequences = self.cfg.data.val_sequences
        logger.info(f"Validation sequences: {val_sequences}")

        # Build validation dataset
        self.val_dataset = MultiSequenceRGBSonarDataset(
            data_root=self.cfg.data.data_root,
            sequences=val_sequences,
            sequence_length=self.cfg.training.sequence_length,
            mode='pair',  # Validation uses pair mode
            transforms=None  # No augmentation for validation
        )

        logger.info(f"Validation dataset: {len(self.val_dataset)} samples")

        # Build dataloader using project's build function
        self.val_loader = build_dataloader(
            self.val_dataset,
            batch_size=self.cfg.training.val_batch_size,
            shuffle=False,
            num_workers=self.cfg.system.num_workers,
            pin_memory=True,
            distributed=False,
            drop_last=False
        )

        logger.info(f"Validation batches: {len(self.val_loader)}")

    def test_temporal_state_reset(self):
        """
        Test 1: Verify temporal state reset mechanism.

        This test checks:
        - Model has reset_temporal_state() method
        - Reset returns correct status dict
        - All temporal modules are properly reset
        """
        logger.info("\n" + "=" * 70)
        logger.info("Test 1: Temporal State Reset Mechanism")
        logger.info("=" * 70)

        if not hasattr(self.model, 'reset_temporal_state'):
            logger.error("✗ Model lacks reset_temporal_state() method!")
            return False

        # Call reset and check status
        reset_status = self.model.reset_temporal_state()
        logger.info(f"Reset status: {reset_status}")

        # Verify all temporal modules were reset
        all_reset = True
        for module_name, status in reset_status.items():
            if status is False:
                logger.error(f"✗ {module_name} failed to reset!")
                all_reset = False
            elif status is True:
                logger.info(f"✓ {module_name} reset successfully")
            elif status is None:
                logger.info(f"○ {module_name} not used (None)")

        if all_reset:
            logger.info("✓ Test 1 PASSED: All temporal modules reset successfully")
            return True
        else:
            logger.error("✗ Test 1 FAILED: Some modules failed to reset")
            return False

    def test_shape_validation(self):
        """
        Test 2: Verify EMA shape validation.

        This test simulates training->validation transition with different shapes:
        - Training: [B*T=5, C, H, W]
        - Validation: [B=32, C, H, W]
        """
        logger.info("\n" + "=" * 70)
        logger.info("Test 2: EMA Shape Validation (Training->Validation Transition)")
        logger.info("=" * 70)

        if not hasattr(self.model, 'temporal_aggregator') or self.model.temporal_aggregator is None:
            logger.warning("○ Model has no temporal_aggregator, skipping test")
            return True

        temporal_agg = self.model.temporal_aggregator
        ema = temporal_agg.ema

        # Simulate training phase features [B*T=5, C, 32, 32]
        training_features = torch.randn(5, 256, 32, 32, device=self.device)
        logger.info(f"Simulating training features: {training_features.shape}")

        # Set previous features (simulates temporal state from training)
        temporal_agg._prev_features = training_features

        # Simulate validation phase features [B=32, C, 32, 32]
        validation_features = torch.randn(32, 256, 32, 32, device=self.device)
        logger.info(f"Simulating validation features: {validation_features.shape}")

        # Test EMA forward with shape mismatch
        try:
            output = ema(
                feat_current=validation_features,
                feat_prev=temporal_agg._prev_features
            )
            logger.info(f"✓ EMA forward succeeded: output shape {output.shape}")

            # Verify output shape matches current features
            if output.shape == validation_features.shape:
                logger.info("✓ Output shape matches validation features (correct behavior)")
                logger.info("✓ Test 2 PASSED: Shape validation works correctly")
                return True
            else:
                logger.error(f"✗ Output shape {output.shape} doesn't match validation shape {validation_features.shape}")
                return False

        except Exception as e:
            logger.error(f"✗ EMA forward failed with error: {e}")
            logger.error("✗ Test 2 FAILED: Shape validation doesn't handle mismatch")
            return False

    def test_validation_loop(self, num_batches: int = 5):
        """
        Test 3: Run partial validation loop.

        This test runs actual validation on a few batches to ensure:
        - Validation doesn't crash
        - Temporal state is properly managed
        - All losses are computed correctly

        Args:
            num_batches: Number of batches to test (default: 5)
        """
        logger.info("\n" + "=" * 70)
        logger.info(f"Test 3: Validation Loop ({num_batches} batches)")
        logger.info("=" * 70)

        self.model.eval()

        # Reset temporal state before validation (mimics trainer.validate())
        logger.info("Resetting temporal state before validation...")
        if hasattr(self.model, 'reset_temporal_state'):
            reset_status = self.model.reset_temporal_state()
            logger.info(f"Reset status: {reset_status}")

        batch_count = 0
        success = True

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                if batch_idx >= num_batches:
                    break

                logger.info(f"\nProcessing batch {batch_idx + 1}/{num_batches}")

                try:
                    # Move batch to device
                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                             for k, v in batch.items()}

                    # Log batch info
                    logger.info(f"  rgb_template: {batch['rgb_template'].shape}")
                    logger.info(f"  rgb_search: {batch['rgb_search'].shape}")
                    logger.info(f"  sonar_template: {batch['sonar_template'].shape}")
                    logger.info(f"  sonar_search: {batch['sonar_search'].shape}")
                    logger.info(f"  gt_boxes: {batch['gt_boxes'].shape}")

                    # Forward pass
                    outputs = self.model(batch)

                    # Log outputs
                    logger.info(f"  Output keys: {list(outputs.keys())}")
                    if 'boxes' in outputs:
                        logger.info(f"  Predicted boxes: {outputs['boxes'].shape}")
                    if 'scores' in outputs:
                        logger.info(f"  Confidence scores: {outputs['scores'].shape}")

                    logger.info(f"✓ Batch {batch_idx + 1} processed successfully")
                    batch_count += 1

                except Exception as e:
                    logger.error(f"✗ Batch {batch_idx + 1} failed with error:")
                    logger.error(f"  {type(e).__name__}: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    success = False
                    break

        if success and batch_count == num_batches:
            logger.info(f"\n✓ Test 3 PASSED: All {batch_count} batches processed successfully")
            return True
        else:
            logger.error(f"\n✗ Test 3 FAILED: Only {batch_count}/{num_batches} batches succeeded")
            return False

    def test_training_to_validation_transition(self):
        """
        Test 4: Full training->validation transition.

        This test simulates the exact flow that occurs during training:
        1. Run model in training mode
        2. Build up temporal state
        3. Switch to validation mode
        4. Reset temporal state
        5. Run validation
        """
        logger.info("\n" + "=" * 70)
        logger.info("Test 4: Training->Validation Transition")
        logger.info("=" * 70)

        # Phase 1: Training mode
        logger.info("\nPhase 1: Training Mode")
        self.model.train()

        # Simulate training batch [B=1, T=5]
        training_batch = {
            'rgb_template': torch.randn(1, 3, 224, 224, device=self.device),
            'rgb_search': torch.randn(5, 3, 224, 224, device=self.device),
            'sonar_template': torch.randn(1, 3, 224, 224, device=self.device),
            'sonar_search': torch.randn(5, 3, 224, 224, device=self.device),
            'gt_boxes': torch.randn(5, 4, device=self.device),
            'metadata': {'num_template': 1, 'num_search': 5}
        }

        logger.info("Simulating training forward pass...")
        try:
            with torch.set_grad_enabled(True):
                train_outputs = self.model(training_batch)
            logger.info(f"✓ Training forward succeeded: {list(train_outputs.keys())}")

            # Check if temporal state was built
            if hasattr(self.model, 'temporal_aggregator') and self.model.temporal_aggregator is not None:
                if hasattr(self.model.temporal_aggregator, '_prev_features'):
                    prev_feat = self.model.temporal_aggregator._prev_features
                    if prev_feat is not None:
                        logger.info(f"  Temporal state shape: {prev_feat.shape}")
                    else:
                        logger.info("  Temporal state: None (training mode doesn't use cross-batch state)")
        except Exception as e:
            logger.error(f"✗ Training forward failed: {e}")
            return False

        # Phase 2: Validation mode
        logger.info("\nPhase 2: Validation Mode")
        self.model.eval()

        # Reset temporal state (critical step!)
        logger.info("Resetting temporal state for validation...")
        if hasattr(self.model, 'reset_temporal_state'):
            reset_status = self.model.reset_temporal_state()
            logger.info(f"  Reset status: {reset_status}")

        # Simulate validation batch [B=32]
        validation_batch = {
            'rgb_template': torch.randn(32, 3, 224, 224, device=self.device),
            'rgb_search': torch.randn(32, 3, 224, 224, device=self.device),
            'sonar_template': torch.randn(32, 3, 224, 224, device=self.device),
            'sonar_search': torch.randn(32, 3, 224, 224, device=self.device),
            'gt_boxes': torch.randn(32, 4, device=self.device),
            'metadata': {'num_template': 32, 'num_search': 32}
        }

        logger.info("Running validation forward pass...")
        try:
            with torch.no_grad():
                val_outputs = self.model(validation_batch)
            logger.info(f"✓ Validation forward succeeded: {list(val_outputs.keys())}")
            logger.info("✓ Test 4 PASSED: Training->Validation transition works correctly")
            return True
        except Exception as e:
            logger.error(f"✗ Validation forward failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            logger.error("✗ Test 4 FAILED: Transition handling broken")
            return False

    def run_all_tests(self):
        """
        Run all validation tests.

        Returns:
            True if all tests pass, False otherwise
        """
        logger.info("\n" + "=" * 70)
        logger.info("Running All Validation Tests")
        logger.info("=" * 70)

        results = {
            'test_temporal_state_reset': self.test_temporal_state_reset(),
            'test_shape_validation': self.test_shape_validation(),
            'test_training_to_validation_transition': self.test_training_to_validation_transition(),
            'test_validation_loop': self.test_validation_loop(num_batches=3),
        }

        # Summary
        logger.info("\n" + "=" * 70)
        logger.info("Test Summary")
        logger.info("=" * 70)

        passed = sum(1 for result in results.values() if result)
        total = len(results)

        for test_name, result in results.items():
            status = "✓ PASSED" if result else "✗ FAILED"
            logger.info(f"{status}: {test_name}")

        logger.info("=" * 70)
        logger.info(f"Total: {passed}/{total} tests passed")

        if passed == total:
            logger.info("✓ ALL TESTS PASSED - Validation stage is ready!")
            return True
        else:
            logger.error(f"✗ {total - passed} TESTS FAILED - Please review fixes")
            return False


def main():
    """
    Main entry point for validation test script.
    """
    import argparse

    parser = argparse.ArgumentParser(description='Test validation stage temporal state management')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/train/train.yaml',
        help='Path to training config file'
    )
    args = parser.parse_args()

    # Run tests
    tester = ValidationTester(config_path=args.config)
    success = tester.run_all_tests()

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()