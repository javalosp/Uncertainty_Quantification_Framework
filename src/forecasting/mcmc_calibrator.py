import numpy as np
import pandas as pd
from scipy.stats import norm


class MCMCCalibrator:
    """Bayesian MCMC calibrator for drift and volatility parameters."""
    def __init__(self, n_iterations=3000, burn_in=500, confidence_level=0.95):
        self.n_iterations = n_iterations
        self.burn_in = burn_in
        self.confidence_level = confidence_level
        self.posterior_samples = None
        self.last_val = None
        self.last_year = None
        self.r_hat = 1.0
        self.is_fitted = False
        self.forecast_df = None

    def _parse_inputs(self, arg1, arg2=None, time_col='Year', value_col='Published_Mean'):
        """Universal parser accepting either a Pandas DataFrame or (X, y) NumPy arrays."""
        if isinstance(arg1, pd.DataFrame):
            df_sorted = arg1.sort_values(by=time_col).dropna(subset=[value_col])
            X = df_sorted[time_col].values.astype(float)
            y = df_sorted[value_col].values.astype(float)
        else:
            X = np.array(arg1, dtype=float).ravel()
            y = np.array(arg2, dtype=float).ravel() if arg2 is not None else None
        return X, y

    def fit(self, X, y=None, time_col='Year', value_col='Published_Mean', **kwargs):
        """Runs Metropolis-Hastings MCMC sampling to calibrate drift and variance."""
        t_arr, y_arr = self._parse_inputs(X, y, time_col, value_col)
        if len(y_arr) < 4:
            raise ValueError("[MCMCCalibrator] Time series must contain at least 4 observations.")
            
        self.last_year = int(np.max(t_arr))
        self.last_val = y_arr[-1]
        returns = np.diff(np.log(np.maximum(y_arr, 1e-6)))
        
        def log_likelihood(mu, sigma):
            if sigma <= 0: return -np.inf
            return -0.5 * len(returns) * np.log(2 * np.pi * sigma**2) - np.sum((returns - mu)**2) / (2 * sigma**2)
            
        chains = []
        for _ in range(2):
            mu_curr = np.mean(returns) if len(returns) > 0 else 0.01
            sigma_curr = np.std(returns) if len(returns) > 0 and np.std(returns) > 0 else 0.05
            chain_samples = []
            
            for _ in range(self.n_iterations):
                mu_prop = mu_curr + np.random.normal(0, 0.005)
                sigma_prop = sigma_curr + np.random.normal(0, 0.002)
                if sigma_prop > 0:
                    if np.log(np.random.uniform(0, 1)) < (log_likelihood(mu_prop, sigma_prop) - log_likelihood(mu_curr, sigma_curr)):
                        mu_curr, sigma_curr = mu_prop, sigma_prop
                chain_samples.append((mu_curr, sigma_curr))
            chains.append(np.array(chain_samples)[self.burn_in:])
            
        c1, c2 = chains[0][:, 0], chains[1][:, 0]
        mean_chain = np.array([np.mean(c1), np.mean(c2)])
        b_val = len(c1) * np.var(mean_chain, ddof=1)
        w_val = 0.5 * (np.var(c1, ddof=1) + np.var(c2, ddof=1))
        var_plus = (len(c1) - 1) / len(c1) * w_val + b_val / len(c1)
        self.r_hat = np.sqrt(var_plus / w_val) if w_val > 0 else 1.0
        
        self.posterior_samples = np.vstack(chains)
        self.is_fitted = True
        return self

    def predict(self, X, return_std=True, **kwargs):
        """Projects geometric Brownian motion parameters forward."""
        if not self.is_fitted or self.posterior_samples is None:
            raise RuntimeError("[MCMCCalibrator] Cannot predict before calling fit().")
            
        steps = len(np.array(X).ravel())
        mu_mean = np.mean(self.posterior_samples[:, 0])
        sigma_mean = np.mean(self.posterior_samples[:, 1])
        
        forecast = self.last_val * np.exp(np.arange(1, steps + 1) * mu_mean)
        stds = forecast * sigma_mean * np.sqrt(np.arange(1, steps + 1))
        
        if return_std:
            return forecast, stds
        return forecast

    def predict_intervals(self, horizon: int, **kwargs) -> pd.DataFrame:
        """Helper method preserving compatibility with DataFrame tests."""
        future_years = np.arange(self.last_year + 1, self.last_year + horizon + 1)
        mean_f, std_f = self.predict(future_years, return_std=True)
        z_score = norm.ppf(1.0 - (1.0 - self.confidence_level) / 2.0)
        self.forecast_df = pd.DataFrame({
            'Year': future_years,
            'Mean_Forecast': mean_f,
            'Lower_Bound': np.maximum(0.0, mean_f - z_score * std_f),
            'Upper_Bound': mean_f + z_score * std_f,
            'Std_Dev': std_f
        })
        return self.forecast_df

    def to_universal_uncertainty(self, parameter_prefix: str, source_node: str, target_node: str, material: str = "Copper") -> pd.DataFrame:
        """Converts forecasts into the 12-column Universal MFA Schema structure."""
        if not self.is_fitted or self.forecast_df is None:
            raise RuntimeError("[MCMCCalibrator] Model must be fitted and predicted before exporting to universal schema.")
            
        records = []
        for _, row in self.forecast_df.iterrows():
            mean_val = float(row['Mean_Forecast'])
            std_dev = float(row['Std_Dev'])
            cv = (std_dev / mean_val) if (mean_val > 0 and std_dev > 0) else 0.05
            
            records.append({
                'Parameter_ID': f"{parameter_prefix}_{source_node}_to_{target_node}_{int(row['Year'])}",
                'Source_Node': source_node,
                'Target_Node': target_node,
                'Material': material,
                'Year': int(row['Year']),
                'Flow_Type': 'trade',
                'Uncertainty_Class': 'aleatory',
                'Published_Mean': mean_val,
                'CV_or_StdDev': cv,
                'Bound_Min': float(row['Lower_Bound']),
                'Bound_Max': float(row['Upper_Bound']),
                'Data_Pedigree_Score': 2.0
            })
        return pd.DataFrame(records)