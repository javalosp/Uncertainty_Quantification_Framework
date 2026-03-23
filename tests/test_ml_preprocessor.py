import unittest
import pandas as pd
import numpy as np
import sys
import os
import warnings

# Ensure the test can find the 'src' module
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(PROJECT_ROOT)

from src.preprocess_dynamic import EmpiricalDataProcessor

class TestEmpiricalDataProcessor(unittest.TestCase):
    """
    Unit tests for the Machine Learning and Empirical Data Preprocessor.
    """

    def setup(self):
        """
        Suppress the convergence warnings from scikit-learn during automated 
        testing, considering that tests intentionally use edge-case dummy data.
        """
        warnings.filterwarnings("ignore")
        
        # A dummy processor for methods that don't require complex initialisation
        self.dummy_processor = EmpiricalDataProcessor(
            raw_historical_data=pd.DataFrame(),
            simulation_start_year=2024,
            simulation_end_year=2050
        )

    def test_algorithmic_dqi_penalties(self):
        """
        Tests if the algorithmic Data Quality Indicators (DQI) correctly penalise 
        datasets with severe missing records (triggering Epistemic classification).
        """
        # Create 100 years of data, where exactly 35 years are missing (35% gaps)
        # End the series at 2024 to align with the default simulation start year
        valid_data = [10.0] * 65
        missing_data = [np.nan] * 35
        combined_data = valid_data + missing_data
        
        # Shuffle to simulate random gaps in historical data
        np.random.seed(42)
        np.random.shuffle(combined_data)
        
        test_series = pd.Series(combined_data, index=np.arange(1925, 2025))
        
        # Run the DQI evaluator
        dqis = self.dummy_processor._calculate_empirical_dqis(test_series)
        
        # Rule: missing_ratio > 0.30 MUST trigger a Completeness (Comp) penalty of 5
        self.assertEqual(
            dqis['Comp'], 5,
            f"Failed: 35% missing data should yield a Comp score of 5, got {dqis['Comp']}"
        )
        print("\n [PASSED] Algorithmic DQI penalty logic.")

    def test_fuzzy_delay_mass_conservation(self):
        """
        Tests if the scrap generation function conserves mass.
        Exactly 100% of an input cohort must eventually exit the system.
        """
        # Calculate fractions for a product living roughly 15 years
        min_frac, mode_frac, max_frac = self.dummy_processor._calculate_fuzzy_delay_distribution(
            min_life=10, mode_life=15, max_life=20, std_dev=2.0
        )
        
        # The sum of the probability density function array must equal exactly 1.0
        self.assertAlmostEqual(np.sum(min_frac), 1.0, places=5, msg="Min Life distribution leaked mass.")
        self.assertAlmostEqual(np.sum(mode_frac), 1.0, places=5, msg="Mode Life distribution leaked mass.")
        self.assertAlmostEqual(np.sum(max_frac), 1.0, places=5, msg="Max Life distribution leaked mass.")
        
        print("\n [PASSED] Fuzzy delay mass conservation.")

    def test_non_negative_physical_constraints(self):
        """
        Tests if the Machine Learning proxy ensembles obey the laws of physics
        by truncating forecasts to 0.0 when models predict negative mass.
        """
        hist_years = np.arange(2000, 2010)
        target_years = np.arange(2000, 2051)
        
        # SABOTAGE THE DATA: 
        # The flow is violently dropping to 0, while the proxy is exploding upwards.
        # A purely linear model will forecast massive negative mass by 2050.
        flow_data = pd.Series(np.linspace(100, 0, 10), index=hist_years) 
        proxy_series = np.linspace(100, 1000, len(target_years))
        proxy_df = pd.DataFrame({'GDP': proxy_series}, index=target_years)
        
        processor = EmpiricalDataProcessor(
            raw_historical_data=pd.DataFrame({'Flow': flow_data}),
            proxy_data=proxy_df,
            simulation_start_year=2024,
            simulation_end_year=2050
        )
        
        # Execute the Wang ML Ensemble 
        core, bound_min, bound_max = processor._evaluate_proxy_ensembles(flow_data, 'GDP')
        
        # Validate that no value in any array drops below zero
        self.assertTrue(np.all(core >= 0.0), "Physics Violation: Median forecast dropped below 0.0")
        self.assertTrue(np.all(bound_min >= 0.0), "Physics Violation: Minimum bound dropped below 0.0")
        self.assertTrue(np.all(bound_max >= 0.0), "Physics Violation: Maximum bound dropped below 0.0")
        
        print("\n [PASSED] Non-Negative physical constraints (Machine Learning).")

if __name__ == '__main__':
    unittest.main(verbosity=2)