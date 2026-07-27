import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize

try:
    from tbats import TBATS
    HAS_TBATS_LIB = True
except ImportError:
    HAS_TBATS_LIB = False


class TBATSModel:
    """TBATS / Trigonometric seasonal decomposition model exporting Gaussian uncertainty."""
    def __init__(self, seasonal_periods=[5], use_box_cox=True, confidence_level=0.95):
        self.seasonal_periods = seasonal_periods
        self.use_box_cox = use_box_cox
        self.confidence_level = confidence_level
        self.fitted_params = {}
        self.residuals_std = 0.0
        self.is_fitted = False
        self.last_year = None
        self.model = None
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
        """Fits TBATS or trigonometric decomposition to historical observations."""
        t_arr, y_arr = self._parse_inputs(X, y, time_col, value_col)
        if len(y_arr) < 4:
            raise ValueError("[TBATSModel] Time series must contain at least 4 observations.")
            
        self.last_year = int(np.max(t_arr))
        t_idx = np.arange(len(y_arr))
        period = self.seasonal_periods[0] if self.seasonal_periods else 5
        
        if HAS_TBATS_LIB:
            estimator = TBATS(seasonal_periods=self.seasonal_periods, use_box_cox=self.use_box_cox)
            self.model = estimator.fit(y_arr)
            self.residuals_std = np.std(self.model.resid)
        else:
            def loss_func(params):
                alpha, beta, a, b = params
                pred = alpha + beta * t_idx + a * np.sin(2 * np.pi * t_idx / period) + b * np.cos(2 * np.pi * t_idx / period)
                return np.sum((y_arr - pred)**2)
                
            res = minimize(loss_func, x0=[np.mean(y_arr), 0.0, 0.0, 0.0], method='Nelder-Mead')
            self.fitted_params = {'alpha': res.x[0], 'beta': res.x[1], 'a': res.x[2], 'b': res.x[3], 'period': period}
            
            t_pred = res.x[0] + res.x[1] * t_idx + res.x[2] * np.sin(2 * np.pi * t_idx / period) + res.x[3] * np.cos(2 * np.pi * t_idx / period)
            self.residuals_std = np.std(y_arr - t_pred) if len(y_arr) > 4 else np.std(y_arr) * 0.10
            
        self.is_fitted = True
        return self

    def predict(self, X, return_std=True, **kwargs):
        """Projects future mean values and analytical standard deviation bands."""
        if not self.is_fitted:
            raise RuntimeError("[TBATSModel] Cannot predict before calling fit().")
            
        t_future = np.array(X, dtype=float).ravel()
        steps = len(t_future)
        
        if HAS_TBATS_LIB and self.model is not None:
            forecast, conf_int = self.model.forecast(steps=steps, confidence_level=self.confidence_level)
            z_score = norm.ppf(1.0 - (1.0 - self.confidence_level) / 2.0)
            std_devs = (conf_int['upper_bound'] - conf_int['lower_bound']) / (2.0 * z_score)
        else:
            p = self.fitted_params
            forecast = p['alpha'] + p['beta'] * t_future + p['a'] * np.sin(2 * np.pi * t_future / p['period']) + p['b'] * np.cos(2 * np.pi * t_future / p['period'])
            std_devs = self.residuals_std * np.sqrt(np.arange(1, steps + 1))
            
        if return_std:
            return forecast, std_devs
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
            raise RuntimeError("[TBATSModel] Model must be fitted and predicted before exporting to universal schema.")
            
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