import os
import sys
import unittest
import numpy as np
import torch

# Ensure project root is in path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.neural_uq import GPRUncertaintyModel, VariationalLSTM, ProbabilisticNBeats, gaussian_nll_loss


class TestNeuralUQLayer(unittest.TestCase):
    """
    Automated verification suite for Step 2: Non-Parametric & Neural Uncertainty.
    Verifies GPR baseline, Variational LSTM with MC Dropout, and Probabilistic N-BEATS.
    """

    @classmethod
    def setUpClass(cls):
        """Generate deterministic synthetic time-series data for reproducible testing."""
        np.random.seed(42)
        torch.manual_seed(42)
        cls.X_train = np.linspace(0, 10, 30).reshape(-1, 1)
        # Synthetic upward trend with sine oscillation and noise
        cls.y_train = 50.0 + 2.0 * cls.X_train.ravel() + 5.0 * np.sin(cls.X_train.ravel()) + np.random.normal(0, 0.5, size=30)

    def test_01_gpr_uncertainty_model(self):
        """Verify Gaussian Process Regression fitting, prediction, and positive std bounds."""
        print("\n[Test 1/3] Testing GPRUncertaintyModel...")
        gpr = GPRUncertaintyModel(alpha=1e-5)
        gpr.fit(self.X_train, self.y_train)

        X_test = np.linspace(10, 12, 10).reshape(-1, 1)
        mean_pred, std_pred = gpr.predict(X_test, return_std=True)

        self.assertEqual(mean_pred.shape, (10,), "GPR mean prediction shape mismatch.")
        self.assertEqual(std_pred.shape, (10,), "GPR std prediction shape mismatch.")
        self.assertTrue((std_pred > 0).all(), "GPR standard deviation must be strictly positive.")
        
        # Test out-of-sample uncertainty expansion (std should generally grow further from observed data)
        self.assertGreater(std_pred[-1], std_pred[0], "GPR uncertainty should increase as projection horizon extends.")
        print(f" -> GPR verification passed! Final Horizon Mean: {mean_pred[-1]:.2f} +/- {std_pred[-1]:.2f}")

    def test_02_variational_lstm_mc_dropout(self):
        """Verify Variational LSTM forward pass and MC Dropout uncertainty decomposition."""
        print("\n[Test 2/3] Testing VariationalLSTM with MC Dropout...")
        batch_size = 5
        seq_len = 8
        input_size = 1
        hidden_size = 16

        model = VariationalLSTM(input_size=input_size, hidden_size=hidden_size, num_layers=2, dropout_rate=0.2)
        dummy_input = torch.randn(batch_size, seq_len, input_size)

        # 1. Check standard forward pass
        mean, logvar = model(dummy_input)
        self.assertEqual(mean.shape, (batch_size, 1), "LSTM forward mean shape mismatch.")
        self.assertEqual(logvar.shape, (batch_size, 1), "LSTM forward logvar shape mismatch.")

        # 2. Check MC Dropout inference decomposition
        overall_mean, total_std, aleatoric_std, epistemic_std = model.predict_with_uncertainty(dummy_input, n_samples=25)
        
        self.assertEqual(overall_mean.shape, (batch_size, 1), "Overall mean shape mismatch.")
        self.assertEqual(total_std.shape, (batch_size, 1), "Total std shape mismatch.")
        self.assertTrue((total_std > 0).all(), "Total uncertainty must be strictly positive.")
        self.assertTrue((aleatoric_std > 0).all(), "Aleatoric uncertainty must be strictly positive.")
        self.assertTrue((epistemic_std >= 0).all(), "Epistemic uncertainty must be non-negative.")
        print(f" -> Variational LSTM verification passed! Sample Total Std: {total_std[0, 0]:.4f} (Aleatoric: {aleatoric_std[0, 0]:.4f}, Epistemic: {epistemic_std[0, 0]:.4f})")

    def test_03_nbeats_probabilistic_uq(self):
        """Verify Probabilistic N-BEATS dual-head outputs and Gaussian NLL optimization."""
        print("\n[Test 3/3] Testing ProbabilisticNBeats...")
        backcast_len = 10
        forecast_len = 3
        batch_size = 6

        model = ProbabilisticNBeats(backcast_length=backcast_len, forecast_length=forecast_len, num_blocks=2, hidden_dim=32)
        dummy_backcast = torch.randn(batch_size, backcast_len)
        dummy_target = torch.randn(batch_size, forecast_len)

        # 1. Forward prediction test
        mean, std = model(dummy_backcast)
        self.assertEqual(mean.shape, (batch_size, forecast_len), "N-BEATS mean output shape mismatch.")
        self.assertEqual(std.shape, (batch_size, forecast_len), "N-BEATS std output shape mismatch.")
        self.assertTrue((std > 0).all(), "N-BEATS standard deviation must be strictly positive (enforced via Softplus).")

        # 2. Loss computation & backpropagation test
        loss = gaussian_nll_loss(mean, std, dummy_target)
        self.assertFalse(torch.isnan(loss), "Gaussian NLL loss returned NaN.")
        self.assertFalse(torch.isinf(loss), "Gaussian NLL loss returned Inf.")
        
        loss.backward()
        # Verify gradient flow
        has_gradients = any(param.grad is not None and torch.norm(param.grad) > 0 for param in model.parameters())
        self.assertTrue(has_gradients, "Gradients failed to propagate backward through N-BEATS model.")
        print(f" -> Probabilistic N-BEATS verification passed! Computed NLL Loss: {loss.item():.4f}")


if __name__ == '__main__':
    print("="*70)
    print(" EXECUTING STEP 2 NON-PARAMETRIC & NEURAL UQ VERIFICATION SUITE")
    print("="*70)
    unittest.main(verbosity=2)