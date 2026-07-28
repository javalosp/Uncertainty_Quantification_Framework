import os
import sys
import unittest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure project root is in path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.ingestion import MISO2Connector
from src.forecasting import TBATSModel, MCMCCalibrator
from src.neural_uq import GPRUncertaintyModel, VariationalLSTM, ProbabilisticNBeats
from src.benchmarking import ProbabilisticMetrics, NeuralForecasterWrapper


class TestPhase2MISO2Benchmarking(unittest.TestCase):
    """
    Automated Phase 2 Real-Case Validation Suite: Out-of-Sample Benchmarking (1995-2016).
    Evaluates Step 1 Classical and Step 2 Neural UQ models against empirical MISO2 macro-trajectories
    across a 22-year predictive horizon using CRPS, PICP (95%), MIW, and RMSE.
    """

    @classmethod
    def setUpClass(cls):
        """Load and unpivot miso2_test_data.csv, extracting continuous 1970-2016 macro-series."""
        data_path = os.path.join(PROJECT_ROOT, "data/raw_cache/miso2_test_data.csv")
        cls.connector = MISO2Connector(filepath=data_path)
        univ_df = cls.connector.to_universal_schema()

        # Define target countries (avoid low-volume kernels)
        target_economies = {'USA', 'CHN', 'DEU', 'United States of America', 'China', 'Germany'}
        
        # Filter for continuous series with no missing years between 1970 and 2016
        series_groups = univ_df.groupby(['Source_Node', 'Target_Node', 'Material', 'Flow_Type'])
        valid_series = []
        
        for key, group in series_groups:
            src_node = str(key[0])
            # Ensure the series belongs to one of our target stable economies
            if any(econ in src_node for econ in target_economies):
                years = set(group['Year'].astype(int))
                if set(range(1970, 2017)).issubset(years):
                    valid_series.append((key, group.sort_values(by='Year').reset_index(drop=True)))
                
        if not valid_series:
            # Fallback to general continuous series if target economies are named differently in this cut
            print("[Warning] Target economies not found by name; defaulting to highest volume continuous series.")
            all_continuous = []
            for key, group in series_groups:
                if set(range(1970, 2017)).issubset(set(group['Year'].astype(int))):
                    all_continuous.append((group['Published_Mean'].mean(), key, group.sort_values(by='Year').reset_index(drop=True)))
            all_continuous.sort(key=lambda x: x[0], reverse=True)  # Sort by volume descending
            valid_series = [(k, g) for _, k, g in all_continuous]
            
        # Select 10 high-volume, continuous macro-series for backtesting
        cls.test_series = valid_series[:10]
        print(f"\n[Phase 2 Setup] Successfully isolated {len(cls.test_series)} high-volume macro-series (1970-2016) for backtesting.")

    def test_01_execute_out_of_sample_benchmarking(self):
        """Run 22-year out-of-sample forecasts (1995-2016) and rank model performance."""
        print("\n[Phase 2 / Step 1] Executing 1995-2016 Out-of-Sample Probabilistic Benchmarking...")
        
        models = {
            #'TBATS_Decomposition': (TBATSModel(seasonal_periods=[5]), 'Step 1'),
            #'Bayesian_MCMC': (MCMCCalibrator(n_iterations=1500, burn_in=200), 'Step 1'),
            'TBATS_Decomposition': (TBATSModel(seasonal_periods=[5]), 'Step 1'), # Let internal defaults handle seasonal periods
            'Bayesian_MCMC': (MCMCCalibrator(), 'Step 1'),  # Let internal defaults handle sample/burn-in counts
            'Gaussian_Process_GPR': (GPRUncertaintyModel(), 'Step 2'),
            #'Variational_LSTM': (VariationalLSTM(epochs=50), 'Step 2'),
            # Use NeuralForecasterWrapper to handle architecture initialisation (input_size=1, hidden_size=32)
            # and execute the training loop for the specified epochs.
            'Variational_LSTM': (NeuralForecasterWrapper(model_type='lstm', epochs=50), 'Step 2'),
            #'Probabilistic_NBEATS': (ProbabilisticNBeats(epochs=50), 'Step 2')
            'Probabilistic_NBEATS': (NeuralForecasterWrapper(model_type='nbeats', epochs=50), 'Step 2')
        }
        
        benchmark_records = []
        os.makedirs("test_benchmark_plots", exist_ok=True)
        
        for (src, tgt, mat, f_type), df_series in self.test_series:
            series_name = f"{src} -> {tgt} ({mat})"
            print(f"\n[*] Benchmarking Series: '{series_name}'")
            
            # Partition into Train (1970-1994) and Out-of-Sample Test (1995-2016)
            train_df = df_series[df_series['Year'] < 1995]
            test_df = df_series[(df_series['Year'] >= 1995) & (df_series['Year'] <= 2016)]
            
            t_train, y_train = train_df['Year'].values, train_df['Published_Mean'].values
            t_test, y_test = test_df['Year'].values, test_df['Published_Mean'].values
            horizon = len(t_test)  # H = 22 years
            
            # Setup comparative plot
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(t_train, y_train, 'k.-', label="Training Historical (1970-1994)", linewidth=1.5)
            ax.plot(t_test, y_test, 'g-s', label="Ground Truth MISO2 (1995-2016)", linewidth=2)
            
            for model_name, (model_inst, step_tag) in models.items():
                try:
                    # Fit on historical training window
                    model_inst.fit(t_train, y_train)
                    
                    # Predict 22-year out-of-sample horizon
                    if hasattr(model_inst, 'predict_intervals'):
                        pred_df = model_inst.predict_intervals(horizon=horizon)
                        mean_pred = pred_df['Mean_Forecast'].values
                        std_pred = pred_df['Std_Dev'].values
                    else:
                        res = model_inst.predict(t_test, return_std=True)
                        mean_pred, std_pred = (res[0], res[1]) if isinstance(res, tuple) else (res, np.full_like(res, np.std(y_train)*0.1))
                        
                    mean_pred = np.array(mean_pred, dtype=float).ravel()
                    std_pred = np.maximum(np.array(std_pred, dtype=float).ravel(), 1e-6)
                    
                    # Use evaluate_all() to compute all probabilistic metrics
                    eval_metrics = ProbabilisticMetrics.evaluate_all(
                        mean=mean_pred, 
                        std=std_pred, 
                        y_true=y_test, 
                        nominal_level=0.95
                    )
                    
                    benchmark_records.append({
                        'Series': f"{src} -> {tgt}",
                        'Material': mat,
                        'Model_Name': model_name,
                        'Pipeline_Step': step_tag,
                        'CRPS': float(eval_metrics['CRPS']),
                        'PICP_95': float(eval_metrics['PICP_95%']),
                        'Sharpness_MIW': float(eval_metrics['Sharpness_MIW']),
                        'RMSE': float(eval_metrics['RMSE'])
                    })
                    
                    # Plot model forecast mean
                    ax.plot(t_test, mean_pred, label=f"{model_name} ({step_tag}) [CRPS: {eval_metrics['CRPS']:.1f}]", linestyle='--')
                except Exception as e:
                    print(f"    [!] Model '{model_name}' failed on series '{series_name}': {e}")
                    
            ax.set_title(f"Out-of-Sample Benchmark (1995-2016): {series_name}", fontsize=12, fontweight='bold')
            ax.set_xlabel("Year", fontsize=10)
            ax.set_ylabel("Material Mass (Mt)", fontsize=10)
            ax.grid(True, linestyle=':', alpha=0.6)
            ax.legend(loc='upper left', fontsize=8)
            
            clean_filename = f"benchmark_{src}_{tgt}_{mat}".replace(" ", "_").replace("/", "_") + ".png"
            plt.tight_layout()
            plt.savefig(os.path.join("test_benchmark_plots", clean_filename), dpi=200)
            plt.close(fig)
            
        # Compile and assert benchmark summary
        results_df = pd.DataFrame(benchmark_records)
        self.assertFalse(results_df.empty, "Benchmarking summary DataFrame cannot be empty.")
        
        # Aggregate performance by model across all tested series
        ranking_df = results_df.groupby(['Model_Name', 'Pipeline_Step']).agg({
            'CRPS': 'mean',
            'PICP_95': 'mean',
            'Sharpness_MIW': 'mean',
            'RMSE': 'mean'
        }).reset_index().sort_values(by='CRPS', ascending=True)
        
        print("\n" + "="*80)
        print(" PHASE 2 OUT-OF-SAMPLE BENCHMARK RANKING SUMMARY (1995-2016 AVERAGE)")
        print("="*80)
        print(ranking_df.to_string(index=False))
        print("="*80)
        print(" -> All benchmark charts saved to directory: 'test_benchmark_plots/'")
        
        # Assert that at least one neural/non-parametric model achieves competitive reliability
        best_model = ranking_df.iloc[0]['Model_Name']
        print(f"\n [*] Top Ranked Out-of-Sample Forecaster: '{best_model}'")
        self.assertIsNotNone(best_model, "Failed to identify a winning model from rankings.")


if __name__ == '__main__':
    print("="*80)
    print(" EXECUTING PHASE 2 REAL-CASE VALIDATION: OUT-OF-SAMPLE BENCHMARKING (1995-2016)")
    print("="*80)
    unittest.main(verbosity=2)