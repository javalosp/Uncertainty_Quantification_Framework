import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any

class ProspectiveNetworkPropagator:
    """
    Executes prospective network propagation (2017-2035) under deep uncertainty.
    Propagates multi-horizon UQ forecasts through circular economy mass-balance equations
    and evaluates systemic tipping points to generate DMDU experimental datasets.
    """
    
    def __init__(self, start_year: int = 2017, end_year: int = 2035, n_samples: int = 1000):
        self.start_year = start_year
        self.end_year = end_year
        self.horizon = end_year - start_year + 1
        self.n_samples = n_samples
        self.years = np.arange(start_year, end_year + 1)
        
    def _generate_correlated_trajectories(self, mean_vec: np.ndarray, std_vec: np.ndarray, length_scale: float = 3.0) -> np.ndarray:
        """
        Generates N correlated Monte Carlo trajectories across the forecast horizon
        using an exponential decay covariance kernel to prevent aggregation bias.
        """
        H = len(mean_vec)
        t_i, t_j = np.meshgrid(np.arange(H), np.arange(H))
        corr_matrix = np.exp(-np.abs(t_i - t_j) / length_scale)
        
        # Construct covariance matrix: Sigma = D * R * D
        D = np.diag(np.maximum(std_vec, 1e-6))
        cov_matrix = D @ corr_matrix @ D
        
        # Add numerical jitter for positive definiteness
        cov_matrix += np.eye(H) * 1e-8
        
        # Draw multivariate Gaussian samples: Shape [n_samples, H]
        samples = np.random.multivariate_normal(mean_vec, cov_matrix, size=self.n_samples)
        return np.maximum(samples, 0.0)  # Enforce physical non-negativity

    def propagate_circular_economy_balance(
        self, 
        inflow_model: Any, 
        outflow_model: Any, 
        hist_t: np.ndarray, 
        hist_inflow: np.ndarray, 
        hist_outflow: np.ndarray,
        crit_scrap_ratio: float = 0.30
    ) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, Any]]:
        """
        Fits models on historical data, predicts 2017-2035 inflows and waste outflows,
        samples deep uncertainty policy levers, and evaluates circular economy tipping points.
        
        Returns:
            X_df (pd.DataFrame): Parameter feature matrix for DMDU discovery.
            y_bin (np.ndarray): Binary tipping point failure vector (1 for failure, 0 for safe).
            summary_stats (dict): Aggregate propagation diagnostics.
        """
        # 1. Fit models on full historical baseline (1970-2016)
        inflow_model.fit(hist_t, hist_inflow)
        outflow_model.fit(hist_t, hist_outflow)
        
        # 2. Predict out-of-sample prospective horizon (2017-2035)
        if hasattr(inflow_model, 'predict_intervals'):
            inf_pred = inflow_model.predict_intervals(horizon=self.horizon)
            inf_mean, inf_std = inf_pred['Mean_Forecast'].values, inf_pred['Std_Dev'].values
        else:
            res = inflow_model.predict(self.years, return_std=True)
            inf_mean, inf_std = (res[0], res[1]) if isinstance(res, tuple) else (res, np.full_like(res, np.std(hist_inflow)*0.1))
            
        if hasattr(outflow_model, 'predict_intervals'):
            out_pred = outflow_model.predict_intervals(horizon=self.horizon)
            out_mean, out_std = out_pred['Mean_Forecast'].values, out_pred['Std_Dev'].values
        else:
            res = outflow_model.predict(self.years, return_std=True)
            out_mean, out_std = (res[0], res[1]) if isinstance(res, tuple) else (res, np.full_like(res, np.std(hist_outflow)*0.1))
            
        # 3. Generate correlated Monte Carlo trajectories [n_samples, H]
        inflow_trajectories = self._generate_correlated_trajectories(inf_mean, inf_std)
        outflow_trajectories = self._generate_correlated_trajectories(out_mean, out_std)
        
        # 4. Sample Deep Uncertainty Policy & Environmental Levers (X Matrix)
        np.random.seed(42)
        recycling_efficiency = np.random.uniform(0.15, 0.65, size=self.n_samples)      # α: Secondary recovery rate
        demand_shock_factor = np.random.normal(1.0, 0.12, size=self.n_samples)          # γ: Unforeseen demand multiplier
        eol_generation_velocity = np.random.normal(1.0, 0.15, size=self.n_samples)      # δ: Scrap release velocity
        primary_import_constraint = np.random.uniform(0.40, 0.90, size=self.n_samples)  # ε: Primary supply restriction
        
        # 5. Execute Network Propagation across all M trajectories
        final_inflow_2035 = inflow_trajectories[:, -1] * demand_shock_factor
        final_outflow_2035 = outflow_trajectories[:, -1] * eol_generation_velocity
        
        # Calculate Secondary Scrap Recovery Ratio in 2035
        recovered_scrap_2035 = final_outflow_2035 * recycling_efficiency
        scrap_recovery_ratio_2035 = np.where(final_inflow_2035 > 0, recovered_scrap_2035 / final_inflow_2035, 0.0)
        
        # 6. Evaluate Tipping Point Gate (Failure if secondary scrap cannot satisfy critical demand ratio)
        y_bin = np.where(scrap_recovery_ratio_2035 < crit_scrap_ratio, 1, 0)
        
        # Assemble DMDU Feature Matrix (X)
        X_df = pd.DataFrame({
            'Recycling_Efficiency_Rate': recycling_efficiency,
            'Demand_Shock_Multiplier': demand_shock_factor,
            'EoL_Release_Velocity': eol_generation_velocity,
            'Primary_Import_Constraint': primary_import_constraint,
            'Projected_Inflow_2035_Mt': final_inflow_2035,
            'Projected_EoL_Waste_2035_Mt': final_outflow_2035
        })
        
        summary_stats = {
            'total_trajectories': self.n_samples,
            'tipping_point_failures': int(np.sum(y_bin)),
            'failure_probability': float(np.mean(y_bin)),
            'mean_scrap_ratio_2035': float(np.mean(scrap_recovery_ratio_2035)),
            'crit_threshold_applied': crit_scrap_ratio
        }
        
        return X_df, y_bin, summary_stats