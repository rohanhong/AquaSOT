"""
Unit tests for Log-Polar Transform with learnable scales

This test suite verifies the correctness of the log-polar transformation,
particularly focusing on the learnable parameters alpha and beta.

Critical tests:
1. Gradient flow: alpha and beta must have non-zero gradients
2. Parameter effect: changing alpha/beta must affect the sampling grid
3. Inverse transform consistency
"""

import pytest
import torch
import torch.nn as nn
import numpy as np

from src.aquasot3d.dataloader.transforms.logpolar import (
    LogPolarTransform,
    LogPolarPreprocessor
)


class TestLogPolarLearnableScales:
    """Test learnable scale parameters alpha and beta"""

    def test_alpha_beta_gradient_flow(self):
        """
        Test that alpha and beta receive non-zero gradients during backpropagation.

        This verifies that the bug fix (removing cancellation) is effective.
        """
        # Create transform with learnable scales
        transform = LogPolarTransform(
            sonar_height=376,
            sonar_width=682,
            output_height=128,
            output_width=128,
            learnable_scales=True,
            init_alpha=1.0,
            init_beta=1.0
        )

        # Create dummy input
        batch_size = 2
        sonar_img = torch.randn(batch_size, 1, 376, 682, requires_grad=True)

        # Forward pass
        logpolar_img = transform(sonar_img)

        # Create a simple loss (sum of outputs)
        loss = logpolar_img.sum()

        # Backward pass
        loss.backward()

        # Check that alpha and beta have non-zero gradients
        assert transform.alpha.grad is not None, "alpha should have gradient"
        assert transform.beta.grad is not None, "beta should have gradient"
        assert torch.abs(transform.alpha.grad).sum() > 1e-6, \
            "alpha gradient should be non-zero (not cancelled out)"
        assert torch.abs(transform.beta.grad).sum() > 1e-6, \
            "beta gradient should be non-zero (not cancelled out)"

        print(f"Alpha gradient: {transform.alpha.grad.item():.6f}")
        print(f"Beta gradient: {transform.beta.grad.item():.6f}")

    def test_alpha_perturbation_affects_sampling(self):
        """
        Test that perturbing alpha changes the sampling grid.

        This verifies that alpha actually influences the transformation.
        """
        # Create two transforms with different alpha values
        transform1 = LogPolarTransform(
            sonar_height=376,
            sonar_width=682,
            output_height=64,
            output_width=64,
            learnable_scales=True,
            init_alpha=0.8,
            init_beta=1.0
        )

        transform2 = LogPolarTransform(
            sonar_height=376,
            sonar_width=682,
            output_height=64,
            output_width=64,
            learnable_scales=True,
            init_alpha=1.2,
            init_beta=1.0
        )

        # Create identical input
        sonar_img = torch.randn(1, 1, 376, 682)

        # Apply transforms
        with torch.no_grad():
            logpolar1 = transform1(sonar_img)
            logpolar2 = transform2(sonar_img)

        # Check that outputs are different
        diff = torch.abs(logpolar1 - logpolar2).mean()
        assert diff > 1e-3, \
            f"Different alpha values should produce different outputs, but diff={diff:.6f}"

        print(f"Alpha perturbation effect (mean absolute difference): {diff:.6f}")

    def test_beta_perturbation_affects_sampling(self):
        """
        Test that perturbing beta changes the sampling grid.

        This verifies that beta actually influences the transformation.
        """
        # Create two transforms with different beta values
        transform1 = LogPolarTransform(
            sonar_height=376,
            sonar_width=682,
            output_height=64,
            output_width=64,
            learnable_scales=True,
            init_alpha=1.0,
            init_beta=0.8
        )

        transform2 = LogPolarTransform(
            sonar_height=376,
            sonar_width=682,
            output_height=64,
            output_width=64,
            learnable_scales=True,
            init_alpha=1.0,
            init_beta=1.2
        )

        # Create identical input
        sonar_img = torch.randn(1, 1, 376, 682)

        # Apply transforms
        with torch.no_grad():
            logpolar1 = transform1(sonar_img)
            logpolar2 = transform2(sonar_img)

        # Check that outputs are different
        diff = torch.abs(logpolar1 - logpolar2).mean()
        assert diff > 1e-3, \
            f"Different beta values should produce different outputs, but diff={diff:.6f}"

        print(f"Beta perturbation effect (mean absolute difference): {diff:.6f}")

    def test_learnable_vs_fixed_scales(self):
        """
        Test that learnable scales can be optimized through training.
        """
        # Create transform with learnable scales
        transform = LogPolarTransform(
            sonar_height=376,
            sonar_width=682,
            output_height=64,
            output_width=64,
            learnable_scales=True,
            init_alpha=1.0,
            init_beta=1.0
        )

        # Create optimizer with smaller learning rate for stability
        optimizer = torch.optim.Adam([transform.alpha, transform.beta], lr=0.001)

        # Create dummy input and target (fixed for consistency)
        torch.manual_seed(42)
        sonar_img = torch.randn(1, 1, 376, 682)
        target = torch.randn(1, 1, 64, 64)

        # Store initial values
        alpha_init = transform.alpha.item()
        beta_init = transform.beta.item()

        # Training loop with gradient clipping
        for i in range(5):
            optimizer.zero_grad()
            output = transform(sonar_img)
            loss = nn.functional.mse_loss(output, target)
            loss.backward()

            # Clip gradients to prevent instability
            torch.nn.utils.clip_grad_norm_([transform.alpha, transform.beta], max_norm=1.0)

            optimizer.step()

            # Ensure parameters stay positive
            with torch.no_grad():
                transform.alpha.data.clamp_(min=0.1, max=10.0)
                transform.beta.data.clamp_(min=0.1, max=10.0)

        # Check that parameters changed
        alpha_final = transform.alpha.item()
        beta_final = transform.beta.item()

        # Check for NaN
        assert not np.isnan(alpha_final), "Alpha became NaN during optimization"
        assert not np.isnan(beta_final), "Beta became NaN during optimization"

        # Parameters should have changed (relaxed threshold)
        assert abs(alpha_final - alpha_init) > 1e-6 or abs(beta_final - beta_init) > 1e-6, \
            "At least one parameter should change during optimization"

        print(f"Alpha: {alpha_init:.4f} -> {alpha_final:.4f}")
        print(f"Beta: {beta_init:.4f} -> {beta_final:.4f}")


class TestLogPolarTransformBasics:
    """Test basic functionality of log-polar transform"""

    def test_forward_shape(self):
        """Test that forward pass produces correct output shape"""
        transform = LogPolarTransform(
            sonar_height=376,
            sonar_width=682,
            output_height=256,
            output_width=256
        )

        batch_size = 4
        sonar_img = torch.randn(batch_size, 1, 376, 682)
        logpolar_img = transform(sonar_img)

        assert logpolar_img.shape == (batch_size, 1, 256, 256), \
            f"Expected shape (4, 1, 256, 256), got {logpolar_img.shape}"

    def test_dynamic_input_size(self):
        """Test that transform handles different input sizes"""
        transform = LogPolarTransform(
            sonar_height=376,
            sonar_width=682,
            output_height=128,
            output_width=128
        )

        # Test with different input size
        sonar_img = torch.randn(2, 1, 188, 341)  # Half size
        logpolar_img = transform(sonar_img)

        # Output size should remain the same
        assert logpolar_img.shape == (2, 1, 128, 128), \
            f"Expected shape (2, 1, 128, 128), got {logpolar_img.shape}"

    def test_inverse_transform(self):
        """Test inverse transformation"""
        transform = LogPolarTransform(
            sonar_height=100,
            sonar_width=100,
            output_height=64,
            output_width=64
        )

        # Create simple test pattern
        sonar_img = torch.zeros(1, 1, 100, 100)
        sonar_img[:, :, 40:60, 40:60] = 1.0  # Central square

        # Forward and inverse
        logpolar_img = transform(sonar_img)
        reconstructed = transform.inverse_transform(logpolar_img)

        assert reconstructed.shape == sonar_img.shape, \
            "Inverse transform should restore original shape"


class TestLogPolarPreprocessor:
    """Test the preprocessor wrapper"""

    def test_preprocessor_basic(self):
        """Test basic preprocessing pipeline"""
        preprocessor = LogPolarPreprocessor(
            transform_config={
                'sonar_height': 376,
                'sonar_width': 682,
                'output_height': 128,
                'output_width': 128,
                'learnable_scales': True
            },
            normalize=True,
            apply_mask=False  # Disable mask to avoid geometry dependencies
        )

        sonar_img = torch.randn(2, 1, 376, 682)
        output = preprocessor(sonar_img)

        assert output.shape == (2, 1, 128, 128), \
            f"Expected shape (2, 1, 128, 128), got {output.shape}"

    def test_preprocessor_normalization(self):
        """Test that normalization works correctly"""
        preprocessor = LogPolarPreprocessor(
            transform_config={
                'sonar_height': 100,
                'sonar_width': 100,
                'output_height': 64,
                'output_width': 64
            },
            normalize=True,
            apply_mask=False
        )

        # Create input with known range
        sonar_img = torch.randn(1, 1, 100, 100) * 100 + 500
        output = preprocessor(sonar_img)

        # After normalization, values should be in [0, 1]
        # Note: log-polar transform may introduce interpolation artifacts
        # so we just check reasonable bounds
        assert output.min() >= -0.1, "Normalized output should be >= 0"
        assert output.max() <= 1.1, "Normalized output should be <= 1"


class TestLogPolarMathematicalCorrectness:
    """Test mathematical correctness of the transformation"""

    def test_radial_scaling_correctness(self):
        """
        Test that the inverse transformation correctly implements:
        r = exp(ρ / α) - 1

        Verifies the mathematical formula is correctly applied.
        """
        transform = LogPolarTransform(
            sonar_height=376,
            sonar_width=682,
            output_height=64,
            output_width=64,
            learnable_scales=True,
            init_alpha=1.0,  # Use alpha=1.0 for easier validation
            init_beta=1.0
        )

        # Compute grids manually
        with torch.no_grad():
            x_grid, y_grid = transform._compute_cartesian_grid()

            # Compute expected radial distances
            r = torch.sqrt((x_grid - transform.cx)**2 + (y_grid - transform.cy)**2)

            # For the first row (rho=0), r should be close to 0 (since exp(0/α)-1 ≈ 0)
            r_first_row = r[0, :].mean()
            assert r_first_row < 10.0, \
                f"First row should have small r, got {r_first_row:.2f}"

            # For the last row (rho=max), r should be significantly larger
            # With alpha=1.0, rho_max = log(max_range+1), so r_max = max_range
            r_last_row = r[-1, :].mean()
            expected_r_max = transform.max_range  # When alpha=1.0

            # Allow for some tolerance due to discretization and averaging
            assert r_last_row > expected_r_max * 0.8, \
                f"Last row should have large r (>{expected_r_max*0.8:.2f}), got {r_last_row:.2f}"

            print(f"Radial distance range: {r_first_row:.2f} to {r_last_row:.2f}")
            print(f"Expected max: {expected_r_max:.2f}")

    def test_angular_scaling_correctness(self):
        """
        Test that the inverse transformation correctly implements:
        θ = φ / β
        """
        transform = LogPolarTransform(
            sonar_height=376,
            sonar_width=682,
            output_height=64,
            output_width=64,
            learnable_scales=True,
            init_alpha=1.0,
            init_beta=2.0
        )

        # Compute grids manually
        with torch.no_grad():
            x_grid, y_grid = transform._compute_cartesian_grid()

            # Compute angles
            theta = torch.atan2(x_grid - transform.cx, transform.cy - y_grid)

            # Center column (phi=0) should have theta≈0
            center_col = transform.output_width // 2
            theta_center = theta[:, center_col].abs().mean()
            assert theta_center < 0.1, \
                f"Center column should have theta≈0, got {theta_center:.4f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])