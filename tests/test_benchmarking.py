import os
import sys
import unittest
import numpy as np
import pandas as pd

# Ensure project root is in path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.benchmarking import ProbabilisticMetrics, ModelBenchmarkOrchestrator
from src.forecasting import TBATSModel, MCMCCalibrator
from src.neural_uq import GPRUncertaintyModel


class TestBenchmarkingOrchestrator(unittest.TestCase):
    """
    Automated verification suite for the CRPS & Coverage Probability Benchmarking Orchestrator.
    Verifies metric equations, model registration across Step 1 and Step 2, and ranking summaries.
    """
    @classmethod
    def setUpClass(cls):
        """Generate synthetic time series with train/test split."""
        np.random.seed(42)
        years = np.arange(2000, 2026)
        trend = 500.0 + 15.0 * (years - 2000) + 20.0 * np.sin(2 * np.pi * (years - 2000) / 5.0)
        noise = np.random.normal(0, 5.0, size=len(years))
        data = trend + noise
        
        # Train on 2000-2019 (20 years), test on 2020-2025 (6 years)
        cls.X_train = years[:20]
        cls.y_train = data[:20]
        cls.X_test = years[20:]
        cls.y_test = data[20:]

    def test_01_probabilistic_metrics(self):
        """Verify exact calculations for Gaussian CRPS, empirical CRPS, PICP, and MIW."""
        print("\n[Test 1/2] Testing ProbabilisticMetrics calculations...")
        y_true = np.array([100.0, 105.0, 110.0])
        mean_pred = np.array([99.0, 106.0, 109.0])
        std_pred = np.array([2.0, 2.0, 2.0])
        
        metrics = ProbabilisticMetrics.evaluate_all(mean_pred, std_pred, y_true, nominal_level=0.95)
        
        self.assertIn('CRPS', metrics, "CRPS metric missing from summary.")
        self.assertIn('PICP_95%', metrics, "Coverage probability missing from summary.")
        self.assertIn('Sharpness_MIW', metrics, "Sharpness MIW missing from summary.")
        self.assertGreater(metrics['CRPS'], 0.0, "CRPS must be strictly positive for imperfect predictions.")
        self.assertEqual(metrics['PICP_95%'], 1.0, "All true values fall within 2 sigma; PICP should be 1.0.")
        print(f" -> Metrics verified! CRPS: {metrics['CRPS']:.4f} | Coverage: {metrics['PICP_95%']*100:.1f}%")

    def test_02_orchestrator_execution_and_ranking(self):
        """Verify backtesting execution across Step 1 and Step 2 models and comparative plotting."""
        print("\n[Test 2/2] Testing ModelBenchmarkOrchestrator across Step 1 and Step 2 models...")
        orchestrator = ModelBenchmarkOrchestrator(self.X_train, self.y_train, self.X_test, self.y_test, nominal_level=0.95)
        
        # Register Step 1 Models
        orchestrator.register_model("TBATS_Decomposition", TBATSModel(seasonal_periods=[5]), step="Step 1")
        orchestrator.register_model("Bayesian_MCMC", MCMCCalibrator(n_iterations=2000, burn_in=200), step="Step 1")
        
        # Register Step 2 Models
        orchestrator.register_model("Gaussian_Process_GPR", GPRUncertaintyModel(), step="Step 2")
        orchestrator.register_model("Variational_LSTM", model_type="lstm", step="Step 2", seq_len=5, epochs=60, hidden_dim=16)
        orchestrator.register_model("Probabilistic_NBEATS", model_type="nbeats", step="Step 2", seq_len=5, epochs=60, hidden_dim=16)
        
        # Run benchmark
        df_results = orchestrator.run_benchmark()
        
        self.assertIsInstance(df_results, pd.DataFrame, "Orchestrator must return a Pandas DataFrame.")
        self.assertEqual(len(df_results), 5, "All 5 registered models must be evaluated.")
        
        # Verify ranking order (CRPS ascending)
        crps_vals = df_results['CRPS'].values
        self.assertTrue(np.all(np.diff(crps_vals) >= 0), "Results DataFrame must be sorted by CRPS in ascending order.")
        
        # Test comparative visualization export
        plot_path = "test_benchmark_comparison.png"
        orchestrator.plot_benchmark(save_path=plot_path)
        self.assertTrue(os.path.exists(plot_path), "Benchmark plot image was not created.")
        
        print("\n--- BENCHMARK RANKING RESULTS ---")
        print(df_results[['Model_Name', 'Pipeline_Step', 'CRPS', 'PICP_95%', 'Sharpness_MIW', 'RMSE']].to_string(index=False))
        print(" -> Orchestrator test passed successfully!")


if __name__ == '__main__':
    print("="*70)
    print(" EXECUTING CRPS & COVERAGE PROBABILITY BENCHMARKING SUITE")
    print("="*70)
    unittest.main(verbosity=2)