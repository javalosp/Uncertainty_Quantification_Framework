import unittest
import pandas as pd
import numpy as np
import sys
import os

# Ensure the test can find the 'src' module
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(PROJECT_ROOT)

from src.propagate import HybridPropagationEngine, DynamicPropagationEngine

class TestPropagationEngines(unittest.TestCase):
    """
    Unit tests for the Hybrid and Dynamic Propagation Engines.
    """

    def test_alpha_cut_envelope_behavior(self):
        """
        Tests if the HybridPropagationEngine correctly respects alpha-cuts.
        At alpha=1.0 (The Core), Epistemic bounds should collapse to a single value.
        At alpha=0.0 (The Envelope), Epistemic bounds should be at their maximum width.
        """
        # 1. Setup Mock Data (One Aleatory flow, One Epistemic flow)
        mock_data = pd.DataFrame([
            {
                'Flow_Name': 'Epistemic_Flow',
                'Type': 'Epistemic',
                'Params_Aleatory': None,
                # At alpha=0.0 bounds are [10, 30]. At alpha=1.0 bounds collapse to 20.
                'Params_Epistemic': {'min': 10.0, 'mode': 20.0, 'max': 30.0} 
            },
            {
                'Flow_Name': 'Aleatory_Flow',
                'Type': 'Aleatory',
                'Params_Aleatory': {'mu_ln': 0.0, 'sigma_ln': 0.1},
                'Params_Epistemic': None
            }
        ])

        engine = HybridPropagationEngine(mock_data)
        
        # Both flows contribute 1:1 to the final impact
        engine.define_impact_model(specific_k={'Epistemic_Flow': 1.0, 'Aleatory_Flow': 1.0})

        # 2. Run simulation for the two extremes of possibility
        results = engine.run_simulation(n_iterations=100, seed=42, alpha_cuts=[0.0, 1.0])

        df_alpha_0 = results[0.0]
        df_alpha_1 = results[1.0]

        # 3. Assertions for Alpha = 1.0 (The Core)
        # Because Epistemic width collapses to 0, Y_Min and Y_Max should be identical for every iteration
        # (The remaining variance is purely the shared Aleatory Monte Carlo noise)
        np.testing.assert_array_almost_equal(
            df_alpha_1['Y_Min_Estimation'].values, 
            df_alpha_1['Y_Max_Estimation'].values,
            err_msg="Failed: At alpha=1.0, Y_Min and Y_Max should be identical."
        )

        # 4. Assertions for Alpha = 0.0 (The Envelope)
        # Epistemic width is at its maximum (10 vs 30). Y_Max must be strictly greater than Y_Min.
        self.assertTrue(
            np.all(df_alpha_0['Y_Max_Estimation'] > df_alpha_0['Y_Min_Estimation']),
            "Failed: At alpha=0.0, Y_Max must be strictly greater than Y_Min."
        )
        
        print("\n [PASSED] Alpha-Cut envelope behavior.")

    def test_dynamic_temporal_accumulation(self):
        """
        Tests if the DynamicPropagationEngine correctly accumulates mass over time.
        Formula: Stock(t) = Stock(t-1) + Inflows(t) - Outflows(t).
        """
        n_steps = 3 # 3 Years: t=0, t=1, t=2
        
        # 1. Setup Mock Time-Series Data
        # We use purely static Epistemic arrays to easily track the exact math
        mock_dyn_data = pd.DataFrame([
            {
                'Flow_Name': 'Inflow',
                'Type': 'Epistemic',
                'Params_Aleatory_TS': None,
                # Consistently adds 10 units every year
                'Params_Epistemic_TS': {
                    'min': np.array([10.0, 10.0, 10.0]),
                    'mode': np.array([10.0, 10.0, 10.0]),
                    'max': np.array([10.0, 10.0, 10.0])
                }
            },
            {
                'Flow_Name': 'Outflow',
                'Type': 'Epistemic',
                'Params_Aleatory_TS': None,
                # Consistently removes 2 units every year
                'Params_Epistemic_TS': {
                    'min': np.array([2.0, 2.0, 2.0]),
                    'mode': np.array([2.0, 2.0, 2.0]),
                    'max': np.array([2.0, 2.0, 2.0])
                }
            }
        ])

        # 2. Instantiate Dynamic Engine
        start_year = 2020
        end_year = 2022
        engine = DynamicPropagationEngine(mock_dyn_data, start_year, end_year)
        
        # K-Factors: 1.0 = positive accumulation, -1.0 = subtraction
        engine.define_impact_model(specific_k={'Inflow': 1.0, 'Outflow': -1.0})

        # 3. Run Dynamic Simulation
        results = engine.run_dynamic_simulation(n_iterations=10, seed=42, alpha_cuts=[1.0])
        
        # 4. Assertions on the Stock Accumulation Matrix
        # Shape is (iterations, time_steps)
        stock_matrix = results[1.0]['Stock_Max_TS'] 
        
        # Extract a single iteration's trajectory over time
        trajectory = stock_matrix[0] 
        
        # Expected Math: Net flow is +8 per year (10 in, 2 out)
        # t=0 -> 8
        # t=1 -> 8 + 8 = 16
        # t=2 -> 16 + 8 = 24
        expected_trajectory = np.array([8.0, 16.0, 24.0])
        
        np.testing.assert_array_almost_equal(
            trajectory, 
            expected_trajectory,
            err_msg="Failed: Dynamic engine failed to accumulate mass correctly over time."
        )
        
        print("\n [PASSED] Dynamic temporal accumulation.")

if __name__ == '__main__':
    unittest.main(verbosity=2)