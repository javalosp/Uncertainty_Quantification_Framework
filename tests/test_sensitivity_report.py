import unittest
import pandas as pd
import numpy as np
import sys
import os

# Ensure the test can find the 'src' module
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(PROJECT_ROOT)

from src.sensitivity import SensitivityAnalyser
from src.report import RobustnessReporter

class TestSensitivityAndReporting(unittest.TestCase):
    """
    Unit tests for the Global Sensitivity Analysis (GSA) and Metric Reporting modules.
    """

    def test_analytical_sensitivity_math(self):
        """
        Tests if the SensitivityAnalyser correctly normalizes and combines 
        Epistemic Widths and Aleatory Variances into the S_Combined score.
        """
        # 1. Setup Mock Data
        mock_data = pd.DataFrame([
            {
                'Flow_Name': 'Epistemic_Flow',
                'Type': 'Epistemic',
                'Params_Aleatory': None,
                # The epistemic absolute width is exactly 10.0 (20 - 10)
                'Params_Epistemic': {'min': 10.0, 'max': 20.0} 
            },
            {
                'Flow_Name': 'Aleatory_Flow',
                'Type': 'Aleatory',
                # A lognormal distribution with these params yields a specific variance
                'Params_Aleatory': {'mu_ln': 0.0, 'sigma_ln': 0.1}, 
                'Params_Epistemic': None
            }
        ])

        k_map = {'Epistemic_Flow': 1.0, 'Aleatory_Flow': 1.0}
        
        # 2. Run the Analyser
        analyser = SensitivityAnalyser(mock_data, k_map)
        results = analyser.run_analysis()
        
        epi_row = results[results['Flow_Name'] == 'Epistemic_Flow'].iloc[0]
        ale_row = results[results['Flow_Name'] == 'Aleatory_Flow'].iloc[0]

        # 3. Expected Mathematical Derivations
        # Aleatory Range calculation mapped from src/sensitivity.py logic:
        # var = exp(2*0 + 0.01) * (exp(0.01) - 1) = 1.01005 * 0.01005 = ~0.01015
        # std_dev = sqrt(0.01015) = ~0.10075
        # range_contribution = std_dev * 3.29 = ~0.33147
        expected_aleatory_range = 0.33147
        expected_epistemic_width = 10.0
        
        # Combined magnitude = sqrt(Epi^2 + Ale^2)
        total_magnitude = np.sqrt(expected_epistemic_width**2 + 0) + np.sqrt(0 + expected_aleatory_range**2)
        
        expected_s_combined_epi = expected_epistemic_width / total_magnitude
        expected_s_combined_ale = expected_aleatory_range / total_magnitude

        # 4. Assertions
        # Verify 100% of Epistemic ignorance is assigned to the epistemic flow
        self.assertEqual(epi_row['S_Epistemic'], 1.0)
        # Verify 100% of Aleatory variance is assigned to the aleatory flow
        self.assertEqual(ale_row['S_Aleatory'], 1.0)
        
        # Verify the combined magnitude weighting is mathematically exact
        self.assertAlmostEqual(epi_row['S_Combined'], expected_s_combined_epi, places=3,
                               msg="Combined sensitivity score for Epistemic Flow is incorrect.")
        self.assertAlmostEqual(ale_row['S_Combined'], expected_s_combined_ale, places=3,
                               msg="Combined sensitivity score for Aleatory Flow is incorrect.")
        
        print("\n [PASSED] Analytical sensitivity maths.")

    def test_metric_extraction(self):
        """
        Tests if the RobustnessReporter correctly extracts the P05, P50, and P95 
        percentiles from the simulation arrays to calculate the final Executive Report metrics.
        """
        # 1. Setup predictable linear distributions
        # np.linspace(0, 100, 101) creates an array [0, 1, 2... 100]
        # In this array, the 5th percentile is exactly 5.0, 50th is 50.0, 95th is 95.0
        y_min = np.linspace(0, 100, 101)
        y_max = np.linspace(10, 110, 101) # Shifted up by 10
        
        mock_results_map = {
            0.0: pd.DataFrame({'Y_Min_Estimation': y_min, 'Y_Max_Estimation': y_max}),
            # The core (Alpha=1.0) isn't strictly used for the safety buffer, but required to instantiate
            1.0: pd.DataFrame({'Y_Min_Estimation': y_min + 5, 'Y_Max_Estimation': y_max - 5}) 
        }

        # 2. Run Reporter
        reporter = RobustnessReporter(mock_results_map)
        metrics = reporter.get_metrics_dictionary()

        # 3. Expected Mathematical Derivations based on y_max (Conservative limit)
        # P05 of y_max = 15.0
        # P50 (Median) of y_max = 60.0
        # P95 of y_max = 105.0
        
        # 4. Assertions
        # Safety Buffer = P95(y_max) - Median(y_max) -> 105.0 - 60.0 = 45.0
        self.assertEqual(metrics['SAFETY BUFFER REQUIRED'], 45.0, 
                         "Safety Buffer calculation failed.")
        
        # Ignorance Penalty = P95(y_max) - P95(y_min) -> 105.0 - 95.0 = 10.0
        self.assertEqual(metrics['IGNORANCE PENALTY'], 10.0, 
                         "Ignorance Penalty calculation failed.")
        
        # Limits check
        self.assertEqual(metrics['Limit if Data Accurate'], 95.0)
        self.assertEqual(metrics['Limit if Data Flawed'], 105.0)

        print("\n [PASSED] Metric extraction percentiles (P05/P50/P95).")

if __name__ == '__main__':
    unittest.main(verbosity=2)