import os
import sys
import unittest
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(PROJECT_ROOT)

try:
    from src.forecasting import TBATSModel, MCMCCalibrator
    from src.ingestion.schema import UniversalSchemaValidator
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import forecasting modules. {e}")
    sys.exit(1)

class TestClassicalForecastingLayer(unittest.TestCase):
    """
    Unit test suite verifying TBATS decomposition and Bayesian MCMC parameter calibration.
    """
    @classmethod
    def setUpClass(cls):
        """Generate a synthetic 15-year historical trade series with upward drift and 5-year seasonality."""
        np.random.seed(42)
        years = np.arange(2010, 2025)
        # Synthetic copper trade: Base 50,000 tons + 2,000/yr growth + 5,000 sine wave cycle + Gaussian noise
        base_trend = 50000 + 2000 * (years - 2010)
        seasonality = 5000 * np.sin(2 * np.pi * (years - 2010) / 5.0)
        noise = np.random.normal(0, 1500, size=len(years))
        
        cls.synthetic_df = pd.DataFrame({
            'Year': years,
            'Published_Mean': base_trend + seasonality + noise
        })

    def test_01_tbats_decomposition_and_schema_export(self):
        """Verify TBATS fitting, future forecasting, and export to Universal MFA Schema."""
        print("\n[Test 1/2] Testing TBATSModel...")
        model = TBATSModel(seasonal_periods=[5], confidence_level=0.95)
        model.fit(self.synthetic_df, time_col='Year', value_col='Published_Mean')
        
        # Predict 5 years into the future
        forecast_df = model.predict_intervals(horizon=5)
        self.assertEqual(len(forecast_df), 5, "Forecast horizon must output exactly 5 years.")
        self.assertTrue((forecast_df['Upper_Bound'] > forecast_df['Lower_Bound']).all(), "Upper bounds must strictly exceed lower bounds.")
        
        # Verify export to universal schema
        univ_df = model.to_universal_uncertainty(
            parameter_prefix="PRED_TBATS", source_node="ISO_152", target_node="ISO_276", material="Copper"
        )
        self.assertEqual(len(univ_df), 5, "Universal schema export must match forecast horizon length.")
        
        # Validate through universal schema validator
        validated_df = UniversalSchemaValidator.validate(univ_df)
        self.assertTrue((validated_df['Uncertainty_Class'] == 'aleatory').all(), "Classical forecasting must export as Aleatory uncertainty.")
        print(f" -> TBATS test passed! 2025–2029 Mean Trajectory: {forecast_df['Mean_Forecast'].values.round(1)}")

    def test_02_mcmc_bayesian_calibration(self):
        """Verify Bayesian MCMC parameter sampling, R-hat convergence, and credible intervals."""
        print("\n[Test 2/2] Testing MCMCCalibrator...")
        mcmc = MCMCCalibrator(n_iterations=3000, burn_in=500, confidence_level=0.95)
        mcmc.fit(self.synthetic_df, time_col='Year', value_col='Published_Mean')
        
        # Assert mathematical convergence (R-hat below 1.05)
        self.assertLess(mcmc.r_hat, 1.05, f"MCMC chains failed to converge. R-hat: {mcmc.r_hat:.4f}")
        
        # Project credible intervals 3 years forward
        forecast_df = mcmc.predict_intervals(horizon=3)
        self.assertEqual(len(forecast_df), 3, "MCMC forecast must match requested 3-year horizon.")
        self.assertTrue((forecast_df['Std_Dev'] > 0).all(), "MCMC posterior projections must exhibit non-zero variance.")
        print(f" -> MCMC test passed! Gelman-Rubin R-hat: {mcmc.r_hat:.4f}")

if __name__ == '__main__':
    print("="*70)
    print(" EXECUTING STEP 1 FORECASTING VERIFICATION SUITE")
    print("="*70)
    unittest.main(verbosity=2)