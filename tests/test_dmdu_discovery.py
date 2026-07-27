import os
import sys
import unittest
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.dmdu import BaseDiscovery, PRIMAnalyser, CARTAnalyser


class TestDMDUDiscoverySuite(unittest.TestCase):
    """Automated verification suite for Step 3: Scenario Discovery & Vulnerability Analytics (DMDU)."""
    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        cls.N = 1000
        cls.X_df = pd.DataFrame({
            'X1': np.random.uniform(0, 1, cls.N),
            'X2': np.random.uniform(0, 1, cls.N),
            'X3': np.random.uniform(0, 1, cls.N),
            'X4': np.random.normal(50, 10, cls.N)
        })
        cls.y_bin = ((cls.X_df['X1'] < 0.25) & (cls.X_df['X2'] > 0.75)).astype(int)
        cls.y_cont = 50.0 * cls.X_df['X1'] + 50.0 * (1.0 - cls.X_df['X2'])

    def test_01_target_preparation_and_evaluation(self):
        prim = PRIMAnalyser(threshold=25.0, threshold_type='less')
        y_prepared = prim._prepare_target(self.y_cont)
        self.assertEqual(len(y_prepared), self.N)
        self.assertTrue(set(np.unique(y_prepared)).issubset({0, 1}))

    def test_02_prim_analyser_peeling_and_trajectory(self):
        prim = PRIMAnalyser(threshold=1, threshold_type='binary', peel_alpha=0.05, min_support=0.04, target_density=0.85)
        prim.fit(self.X_df, self.y_bin)
        boxes_df = prim.to_dataframe()
        self.assertFalse(boxes_df.empty)
        self.assertGreaterEqual(prim.best_box_['density'], 0.85)

    def test_03_cart_analyser_rule_extraction_and_importance(self):
        cart = CARTAnalyser(threshold=1, threshold_type='binary', max_depth=3, min_samples_leaf=0.05)
        cart.fit(self.X_df, self.y_bin)
        boxes_df = cart.to_dataframe()
        self.assertFalse(boxes_df.empty)
        imp_df = cart.get_feature_importances()
        self.assertEqual(set(imp_df['Feature'].head(2)), {'X1', 'X2'})


if __name__ == '__main__':
    unittest.main(verbosity=2)