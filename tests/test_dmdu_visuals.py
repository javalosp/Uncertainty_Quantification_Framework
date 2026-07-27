import os
import sys
import unittest
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.dmdu import PRIMAnalyser, CARTAnalyser, DMDUVisualiser


class TestDMDUVisualsSuite(unittest.TestCase):
    """
    Automated verification suite for Step 3 Visual Analytics (DMDUVisualiser).
    Verifies generation of PRIM peeling trajectories, scenario bounding bars,
    dimensional trade-off matrices, and CART feature importance plots.
    """
    @classmethod
    def setUpClass(cls):
        """Generate a deterministic 4-variable Monte Carlo dataset."""
        np.random.seed(42)
        cls.N = 600
        cls.X_df = pd.DataFrame({
            'Rainfall_Trend': np.random.uniform(-10, 10, cls.N),
            'Demand_Growth': np.random.uniform(0.5, 3.5, cls.N),
            'Infrastructure_Cost': np.random.uniform(50, 150, cls.N),
            'Market_Volatility': np.random.normal(20, 5, cls.N)
        })
        cls.y_bin = ((cls.X_df['Rainfall_Trend'] < -2.0) & (cls.X_df['Demand_Growth'] > 2.0)).astype(int)
        
        cls.prim = PRIMAnalyser(peel_alpha=0.05, min_support=0.05, target_density=0.80).fit(cls.X_df, cls.y_bin)
        cls.cart = CARTAnalyser(max_depth=3, min_samples_leaf=0.05).fit(cls.X_df, cls.y_bin)

    def test_01_plot_prim_trajectory(self):
        """Verify generation of PRIM Coverage vs. Density trade-off curve."""
        print("\n[Test 1/4] Testing plot_prim_trajectory...")
        save_path = "test_prim_trajectory.png"
        fig = DMDUVisualiser.plot_prim_trajectory(self.prim, save_path=save_path)
        self.assertIsNotNone(fig, "plot_prim_trajectory returned None.")
        self.assertTrue(os.path.exists(save_path), f"Plot image {save_path} was not saved.")
        #if os.path.exists(save_path): os.remove(save_path)
        print(" -> PRIM trajectory plot verified!")

    def test_02_plot_box_boundaries(self):
        """Verify generation of normalized scenario range bar chart."""
        print("\n[Test 2/4] Testing plot_box_boundaries...")
        save_path = "test_box_boundaries.png"
        fig = DMDUVisualiser.plot_box_boundaries(self.prim, box_index=0, save_path=save_path)
        self.assertIsNotNone(fig, "plot_box_boundaries returned None.")
        self.assertTrue(os.path.exists(save_path), f"Plot image {save_path} was not saved.")
        #if os.path.exists(save_path): os.remove(save_path)
        print(" -> Scenario box boundary plot verified!")

    def test_03_plot_dimensional_tradeoff_matrix(self):
        """Verify generation of 2D scatter pair trade-off matrix with bounding boxes."""
        print("\n[Test 3/4] Testing plot_dimensional_tradeoff_matrix...")
        save_path = "test_tradeoff_matrix.png"
        fig = DMDUVisualiser.plot_dimensional_tradeoff_matrix(self.prim, box_index=0, top_n_features=3, save_path=save_path)
        self.assertIsNotNone(fig, "plot_dimensional_tradeoff_matrix returned None.")
        self.assertTrue(os.path.exists(save_path), f"Plot image {save_path} was not saved.")
        #if os.path.exists(save_path): os.remove(save_path)
        print(" -> Dimensional trade-off matrix plot verified!")

    def test_04_plot_cart_feature_importances(self):
        """Verify generation of CART Gini feature importance bar chart."""
        print("\n[Test 4/4] Testing plot_cart_feature_importances...")
        save_path = "test_cart_importances.png"
        fig = DMDUVisualiser.plot_cart_feature_importances(self.cart, top_n=4, save_path=save_path)
        self.assertIsNotNone(fig, "plot_cart_feature_importances returned None.")
        self.assertTrue(os.path.exists(save_path), f"Plot image {save_path} was not saved.")
        #if os.path.exists(save_path): os.remove(save_path)
        print(" -> CART feature importance plot verified!")


if __name__ == '__main__':
    print("="*70)
    print(" EXECUTING STEP 3 VISUAL ANALYTICS VERIFICATION SUITE")
    print("="*70)
    unittest.main(verbosity=2)