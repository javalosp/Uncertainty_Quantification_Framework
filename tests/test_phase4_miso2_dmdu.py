import os
import sys
import unittest
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Import native framework classes
from src.dmdu import CARTAnalyser, PRIMAnalyser, DMDUVisualiser


class TestPhase4MISO2DMDUScenarioDiscovery(unittest.TestCase):
    """
    Automated Phase 4 Verification Suite: DMDU Scenario Discovery & Visual Analytics.
    Ingests the Phase 3 experimental design matrix, executes native PRIM parameter peeling/pasting,
    induces CART decision trees, extracts bounding boxes, and exports diagnostic charts
    """

    @classmethod
    def setUpClass(cls):
        """Load the Phase 3 DMDU Experimental Design Matrix from disk."""
        matrix_path = os.path.join(PROJECT_ROOT, "data/dmdu_ready/phase3_miso2_dmdu_matrix.csv")
        if not os.path.exists(matrix_path):
            raise FileNotFoundError(
                f"[Phase 4 Setup] DMDU matrix not found at '{matrix_path}'. Execute Phase 3 test first!"
            )
            
        df = pd.read_csv(matrix_path)
        cls.y = df['Tipping_Point_Failure'].values
        cls.X = df.drop(columns=['Tipping_Point_Failure'])
        
        print(f"\n[Phase 4 Setup] Loaded DMDU Matrix: {len(cls.X)} trajectories, {cls.X.shape[1]} features.")
        print(f"[Phase 4 Setup] Baseline Tipping Point Failure Rate: {np.mean(cls.y)*100:.1f}%")

    def test_01_execute_native_cart_analysis_and_box_extraction(self):
        """Run native CARTAnalyser to extract feature importances and leaf bounding boxes."""
        print("\n[Phase 4 / Step 1] Executing Native CART Decision Tree Scenario Discovery...")
        
        # Instantiate and fit native CARTAnalyser
        cart = CARTAnalyser(max_depth=3, min_samples_leaf=0.05)
        cart.fit(self.X, self.y)
        
        # Extract Gini feature importances
        imp_df = cart.get_feature_importances()
        self.assertIsInstance(imp_df, pd.DataFrame, "Importances must return a DataFrame.")
        self.assertFalse(imp_df.empty, "Feature importances cannot be empty.")
        
        print("\n" + "="*80)
        print(" CART PARAMETER IMPORTANCE RANKING")
        print("="*80)
        for _, row in imp_df.iterrows():
            print(f" * {row['Feature']:<30} | Relative Importance: {row['Importance']*100:.1f}%")
        print("="*80)
        
        # Verify box extraction from tree leaves
        boxes = cart.boxes_
        self.assertGreater(len(boxes), 0, "CART failed to extract bounding boxes from leaves.")
        top_box = boxes[0]
        print(f"\n [*] Top CART Leaf Box Density:  {top_box['density']*100:.1f}%")
        print(f" [*] Top CART Leaf Box Coverage: {top_box['coverage']*100:.1f}%")
        
        # Generate native diagnostic charts
        os.makedirs("test_dmdu_plots", exist_ok=True)
        DMDUVisualiser.plot_cart_feature_importances(cart, save_path="test_dmdu_plots/native_cart_importances.png")
        DMDUVisualiser.plot_box_boundaries(cart, box_index=0, save_path="test_dmdu_plots/native_cart_box_bounds.png")
        print(" -> Saved native CART diagnostic charts to 'test_dmdu_plots/'.")

    def test_02_execute_native_prim_peeling_and_trajectory_analytics(self):
        """Run native PRIMAnalyser to isolate vulnerability boundaries and plot trade-off matrices."""
        print("\n[Phase 4 / Step 2] Executing Native PRIM Peeling & Pasting Trajectory Analytics...")
        
        # Instantiate and fit native PRIMAnalyser
        prim = PRIMAnalyser(peel_alpha=0.05, paste_alpha=0.05, min_support=0.10, target_density=0.75)
        prim.fit(self.X, self.y)
        
        # Extract trajectory and optimal box
        traj_df = prim.get_trajectory()
        self.assertIsInstance(traj_df, pd.DataFrame, "Trajectory must return a DataFrame.")
        self.assertIsNotNone(prim.best_box_, "PRIM failed to isolate an optimal box.")
        
        best_box = prim.best_box_
        print("\n" + "="*80)
        print(" PRIM VULNERABILITY BOX SUMMARY")
        print("="*80)
        print(f" [*] Optimal Trajectory Step:      Step {best_box['step']}")
        print(f" [*] Discovered Box Density:       {best_box['density']*100:.1f}%")
        print(f" [*] Discovered Box Coverage:      {best_box['coverage']*100:.1f}%")
        print(f" [*] Parameter Support:            {best_box['support']*100:.1f}%")
        print("\n--- MULTI-DIMENSIONAL PARAMETER BOUNDARIES ---")
        
        for col in self.X.columns:
            low, high = best_box['box_min'][col], best_box['box_max'][col]
            glb_min, glb_max = self.X[col].min(), self.X[col].max()
            constrained = "<-- CRITICAL BOUNDARY" if (low > glb_min * 1.01 or high < glb_max * 0.99) else ""
            print(f" * {col:<28}: [{low:.3f}  to  {high:.3f}] {constrained}")
        print("="*80)
        
        # Generate native diagnostic visualisations
        DMDUVisualiser.plot_prim_trajectory(prim, save_path="test_dmdu_plots/native_prim_trajectory.png")
        DMDUVisualiser.plot_box_boundaries(prim, box_index=0, save_path="test_dmdu_plots/native_prim_box_bounds.png")
        DMDUVisualiser.plot_dimensional_tradeoff_matrix(prim, box_index=0, top_n_features=3, save_path="test_dmdu_plots/native_prim_tradeoff_matrix.png")
        print(" -> Saved native PRIM trajectory and trade-off matrices to 'test_dmdu_plots/'.")


if __name__ == '__main__':
    print("="*80)
    print(" EXECUTING PHASE 4: NATIVE DMDU SCENARIO DISCOVERY & VISUAL ANALYTICS")
    print("="*80)
    unittest.main(verbosity=2)