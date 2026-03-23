import unittest
import sys
import os

# Ensure the test can find the 'src' module
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(PROJECT_ROOT)

from src.propagate import StaticReconciliationEngine

class TestStaticReconciliationEngine(unittest.TestCase):
    """
    Unit tests for the algebraic mass-balance solver.
    """

    def test_under_determined_system_raises_error(self):
        """
        Tests if the solver correctly identifies and rejects a network 
        where a node has more than one unknown (calculated) flow.
        """
        # Setup an unsolvable mock network
        # Node_A receives 100 units, but splits into TWO unknown flows going to the environment.
        # Because the environment is unconstrained, there is no mathematical way to 
        # know how the 100 units are split between Out_1 and Out_2.
        mock_network = {
            'nodes': ['Environment', 'Node_A'],
            'edges': [
                {'id': 'Flow_In', 'source': 'Environment', 'target': 'Node_A', 'type': 'Flow'},
                {'id': 'Unknown_Out_1', 'source': 'Node_A', 'target': 'Environment', 'type': 'Flow'},
                {'id': 'Unknown_Out_2', 'source': 'Node_A', 'target': 'Environment', 'type': 'Flow'}
            ],
            'calculated': {
                'Unknown_Out_1': {},
                'Unknown_Out_2': {}
            }
        }

        # Instantiate the solver
        solver = StaticReconciliationEngine(mock_network)
        
        # Provide the known iteration values (Flow_In = 100)
        iteration_values = {'Flow_In': 100.0}

        # Assert that running the solver throws a ValueError
        with self.assertRaises(ValueError) as context:
            solver.resolve_mass_balance(iteration_values)

        # Verify the error message contains the expected warning
        self.assertTrue("Under-determined System" in str(context.exception))
        
        print("\n [PASSED] Rejection of under-determined MFA system.")

    def test_valid_system_resolves_correctly(self):
        """
        Tests if the solver correctly calculates missing flows in a fully 
        determined, multi-node mass-balance network.
        """
        # Setup a valid, solvable mock network
        # 'Node A' receives 'Flow_In = 100' units from the Environment.
        # 'Node A' sends 'Flow_Waste = 20' units to Waste in the Environment.
        # 'Node A' sends the rest 'Flow_internal' to Node B (Flow_Internal -> Unknown).
        # 'Node B' processes it and sends everything out as 'Flow_Product' to the Environment (Flow_Product -> Unknown).
        mock_network = {
            'nodes': ['Environment', 'Node_A', 'Node_B'],
            'edges': [
                {'id': 'Flow_In', 'source': 'Environment', 'target': 'Node_A', 'type': 'Flow'},
                {'id': 'Flow_Waste', 'source': 'Node_A', 'target': 'Environment', 'type': 'Flow'},
                {'id': 'Flow_Internal', 'source': 'Node_A', 'target': 'Node_B', 'type': 'Flow'},
                {'id': 'Flow_Product', 'source': 'Node_B', 'target': 'Environment', 'type': 'Flow'}
            ],
            'calculated': {
                'Flow_Internal': {},
                'Flow_Product': {}
            }
        }

        # Instantiate the solver
        solver = StaticReconciliationEngine(mock_network)

        # Provide the known iteration values
        iteration_values = {
            'Flow_In': 100.0,
            'Flow_Waste': 20.0
        }

        # Given the mock values, the expected maths:
        # 'Node A' will be solved first: 100 - 20 = 80. Therefore, Flow_Internal must be 80.
        # The solver will carry that 80 to Node B.
        # Since there are no other inputs or outputs at Node B, Flow_Product must also be 80.

        # Run the solver
        resolved_state = solver.resolve_mass_balance(iteration_values)

        # Assert the calculated values are mathematically correct
        self.assertEqual(resolved_state['Flow_Internal'], 80.0)
        self.assertEqual(resolved_state['Flow_Product'], 80.0)
        
        print("\n [PASSED] Balance MFA valid system.")

if __name__ == '__main__':
    unittest.main(verbosity=2)