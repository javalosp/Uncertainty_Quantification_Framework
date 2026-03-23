import unittest
import pandas as pd
import numpy as np
import sys
import os

# Ensure the test can find the 'src' module
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(PROJECT_ROOT)

from src.classify import LCIDataManager

class TestLCIDataManager(unittest.TestCase):
    """
    Unit tests for the LCA Module (Data classification and characterisation).
    """
    
    def setup(self):
        """
        Creates a mock dataframe for testing.
        The data is intentionally defined to trigger specific if/else logic paths in the engine.
        """
        self.mock_df = pd.DataFrame({
            'Flow_Name': ['Flow_A_High_Rel', 'Flow_B_Aleatory', 'Flow_C_High_Epi'],
            'Mean': [10.0, 50.0, 100.0],
            'GSD': [1.2, 1.1, 1.5],
            # Flow A: High reliability penalty (Rel=5). Triggers automatic Epistemic override.
            'Rel': [5, 1, 3],
            # Flow B: Low penalties, high temporal aleatory noise (Temp=4). Should be Aleatory.
            'Comp': [1, 1, 3],
            'Tech': [1, 1, 3],
            'Temp': [1, 4, 1],
            'Geo': [1, 1, 1],
            'Contrib_Header': ['Map_A', 'Map_B', 'Map_C']
        })
        
        # Instantiate the manager without a file path and manually inject the mock data
        self.manager = LCIDataManager(file_path=None)
        self.manager.data = self.mock_df

    def test_pedigree_classification_logic(self):
        """
        Tests if flows are properly categorised based on the explicit Pedigree logic rules.
        """
        classified_df = self.manager.classify_uncertainty()
        
        # Rule 1: Flow A has Rel=5, so it MUST be Epistemic regardless of other scores.
        self.assertEqual(
            classified_df.loc[0, 'Uncertainty_Type'], 'Epistemic',
            "Failed: Rel >= 5 did not trigger Epistemic classification."
        )
        
        # Rule 2: Flow B has Epi_Score=3, Ale_Score=5. Because Epi < 8, it MUST be Aleatory.
        self.assertEqual(
            classified_df.loc[1, 'Uncertainty_Type'], 'Aleatory',
            "Failed: Flow B should have been classified as Aleatory."
        )
        
        # Rule 3: Flow C has Epi_Score=9 (which is >= 8), and Ale_Score=2 (Epi > Ale). 
        # It MUST be Epistemic.
        self.assertEqual(
            classified_df.loc[2, 'Uncertainty_Type'], 'Epistemic',
            "Failed: Dominant Epistemic score did not trigger Epistemic classification."
        )
        
        print("\n [PASSED] Pedigree classification logic.")

    def test_fuzzy_and_lognormal_characterisation(self):
        """
        Tests if the mathematical transformations (Min/Max and Lognormal params) are correctly calculated.
        """
        # Run classification first to setup the types
        self.manager.classify_uncertainty()
        params_df = self.manager.characterise_variables()
        
        # Test Epistemic maths
        # Flow A is Epistemic. Mean=10, GSD=1.2
        flow_a_params = params_df.iloc[0]['Params_Epistemic']
        expected_min = 10.0 / (1.2 ** 2) # 6.9444...
        expected_max = 10.0 * (1.2 ** 2) # 14.400...
        
        # Use assertAlmostEqual to prevent floating-point precision errors from failing the test
        self.assertAlmostEqual(flow_a_params['min'], expected_min, places=4, 
                               msg="Fuzzy Min bound calculated incorrectly.")
        self.assertAlmostEqual(flow_a_params['max'], expected_max, places=4, 
                               msg="Fuzzy Max bound calculated incorrectly.")
        self.assertAlmostEqual(flow_a_params['mode'], 10.0, places=4, 
                               msg="Fuzzy Mode does not match the mean.")
        
        # Test Aleatory maths
        # Flow B is Aleatory. Mean=50, GSD=1.1
        flow_b_params = params_df.iloc[1]['Params_Aleatory']
        
        # The mathematical formulas from the classify.py engine
        expected_sigma_ln = np.log(1.1)
        expected_mu_ln = np.log(50.0) - 0.5 * (expected_sigma_ln ** 2)
        
        self.assertAlmostEqual(flow_b_params['sigma_ln'], expected_sigma_ln, places=5, 
                               msg="Lognormal sigma_ln calculated incorrectly.")
        self.assertAlmostEqual(flow_b_params['mu_ln'], expected_mu_ln, places=5, 
                               msg="Lognormal mu_ln calculated incorrectly.")
        
        print("\n [PASSED] Fuzzy and Lognormal characterisation maths.")


if __name__ == '__main__':
    unittest.main(verbosity=2)