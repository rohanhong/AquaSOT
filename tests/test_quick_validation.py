"""
Quick Validation Test for Temporal State Fixes

Tests the two critical fixes:
1. TemporalAggregator.reset() method exists
2. EMA shape validation handles training->validation transitions
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
from aquasot3d.models.temporal.temporal_aggregator import TemporalAggregator

def test_reset_method():
    """Test 1: Verify reset() method exists and works"""
    print("\n" + "="*70)
    print("Test 1: TemporalAggregator.reset() Method")
    print("="*70)

    # Create temporal aggregator
    temporal_agg = TemporalAggregator(feature_dim=256)

    # Check if reset() method exists
    if not hasattr(temporal_agg, 'reset'):
        print("FAILED: TemporalAggregator lacks reset() method!")
        return False

    print("PASSED: TemporalAggregator has reset() method")

    # Test calling reset
    try:
        temporal_agg.reset()
        print("PASSED: reset() method executes without error")
        return True
    except Exception as e:
        print(f"FAILED: reset() method raised error: {e}")
        return False


def test_shape_validation():
    """Test 2: Verify EMA shape validation"""
    print("\n" + "="*70)
    print("Test 2: EMA Shape Validation (Training->Validation Transition)")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create temporal aggregator
    temporal_agg = TemporalAggregator(feature_dim=256).to(device)
    ema = temporal_agg.ema

    # Simulate training phase: [B*T=5, C, 32, 32]
    training_features = torch.randn(5, 256, 32, 32, device=device)
    print(f"Training features shape: {training_features.shape}")

    # Set as previous features (simulate temporal state)
    temporal_agg._prev_features = training_features

    # Simulate validation phase: [B=32, C, 32, 32]
    validation_features = torch.randn(32, 256, 32, 32, device=device)
    print(f"Validation features shape: {validation_features.shape}")

    # Test EMA with shape mismatch
    try:
        output = ema(
            feat_current=validation_features,
            feat_prev=temporal_agg._prev_features
        )
        print(f"PASSED: EMA handled shape mismatch without crash")
        print(f"Output shape: {output.shape}")

        # Verify output matches current features
        if output.shape == validation_features.shape:
            print("PASSED: Output shape correctly matches validation features")
            return True
        else:
            print(f"FAILED: Output shape {output.shape} doesn't match validation shape {validation_features.shape}")
            return False

    except Exception as e:
        print(f"FAILED: EMA crashed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_transition():
    """Test 3: Full training->validation transition"""
    print("\n" + "="*70)
    print("Test 3: Full Training->Validation Transition")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create temporal aggregator
    temporal_agg = TemporalAggregator(feature_dim=256).to(device)

    # Phase 1: Training mode
    print("\nPhase 1: Training Mode")
    temporal_agg.train()

    # Simulate training forward
    training_features = torch.randn(5, 256, 32, 32, device=device)
    aggregated, state = temporal_agg(training_features)
    print(f"Training output shape: {aggregated.shape}")

    # Phase 2: Reset for validation
    print("\nPhase 2: Reset Temporal State")
    try:
        temporal_agg.reset()
        print("PASSED: reset() executed successfully")
    except Exception as e:
        print(f"FAILED: reset() failed: {e}")
        return False

    # Phase 3: Validation mode
    print("\nPhase 3: Validation Mode")
    temporal_agg.eval()

    # Simulate validation forward with different batch size
    validation_features = torch.randn(32, 256, 32, 32, device=device)
    try:
        with torch.no_grad():
            aggregated, state = temporal_agg(validation_features)
        print(f"PASSED: Validation forward succeeded")
        print(f"Validation output shape: {aggregated.shape}")
        return True
    except Exception as e:
        print(f"FAILED: Validation forward failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "="*70)
    print("QUICK VALIDATION TEST - Temporal State Fixes")
    print("="*70)

    results = {
        'test_reset_method': test_reset_method(),
        'test_shape_validation': test_shape_validation(),
        'test_full_transition': test_full_transition()
    }

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    passed = sum(1 for result in results.values() if result)
    total = len(results)

    for test_name, result in results.items():
        status = "PASSED" if result else "FAILED"
        print(f"[{status}] {test_name}")

    print("="*70)
    print(f"Result: {passed}/{total} tests passed")

    if passed == total:
        print("\nSUCCESS: All validation fixes are working correctly!")
        return 0
    else:
        print(f"\nFAILURE: {total - passed} test(s) failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())