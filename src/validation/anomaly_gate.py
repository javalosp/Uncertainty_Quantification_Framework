import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


class AnomalyGate:
    """Statistical and heuristic anomaly detection gates for time series trajectories."""

    @staticmethod
    def detect_zscore_outliers(
        df: pd.DataFrame,
        value_col: str = 'Published_Mean',
        group_col: str = 'Parameter_ID',
        threshold: float = 3.0,
        use_modified: bool = True
    ) -> pd.DataFrame:
        """
        Identifies statistical outliers within each time series group.
        
        By default, applies the Modified Z-Score (using Median and Median Absolute Deviation - MAD),
        which is robust against outlier masking in small sample sizes (N <= 15).
        Formula: M_i = 0.6745 * |x_i - median| / MAD
        """
        def _calc_robust_z(series):
            n = len(series)
            if n == 0:
                return np.zeros_like(series)
                
            # Automatically apply Modified Z-score (MAD) for small samples (N < 15) or when enabled
            if use_modified or n < 15:
                med = series.median()
                mad = (series - med).abs().median()
                
                # Fallback to standard deviation if MAD is 0 (e.g., constant series except for the outlier)
                if mad == 0 or np.isnan(mad):
                    std = series.std()
                    return np.zeros_like(series) if (std == 0 or np.isnan(std)) else np.abs((series - series.mean()) / std)
                    
                return 0.6745 * np.abs(series - med) / mad
            else:
                std = series.std()
                return np.zeros_like(series) if (std == 0 or np.isnan(std)) else np.abs((series - series.mean()) / std)

        df_out = df.copy()
        if group_col in df_out.columns:
            df_out['Z_Score'] = df_out.groupby(group_col)[value_col].transform(_calc_robust_z)
        else:
            df_out['Z_Score'] = _calc_robust_z(df_out[value_col])
            
        outliers = df_out[df_out['Z_Score'] > threshold]
        return outliers

    @staticmethod
    def detect_iqr_outliers(
        df: pd.DataFrame,
        value_col: str = 'Published_Mean',
        group_col: str = 'Parameter_ID',
        multiplier: float = 1.5
    ) -> pd.DataFrame:
        """
        Identifies non-parametric outliers outside [Q1 - k*IQR, Q3 + k*IQR].
        Formula: IQR = Q3 - Q1
        """
        outlier_indices = []
        grouped = df.groupby(group_col) if group_col in df.columns else [('All', df)]
        
        for _, group in grouped:
            q1 = group[value_col].quantile(0.25)
            q3 = group[value_col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - (multiplier * iqr)
            upper_bound = q3 + (multiplier * iqr)
            
            sub_out = group[(group[value_col] < lower_bound) | (group[value_col] > upper_bound)]
            outlier_indices.extend(sub_out.index.tolist())
            
        return df.loc[outlier_indices]

    @staticmethod
    def detect_rate_of_change_spikes(
        df: pd.DataFrame,
        value_col: str = 'Published_Mean',
        time_col: str = 'Year',
        group_col: str = 'Parameter_ID',
        max_pct_change: float = 5.0
    ) -> pd.DataFrame:
        """
        Detects sudden year-over-year jump/drop spikes exceeding max_pct_change (e.g., 5.0 = 500% jump).
        """
        df_sorted = df.sort_values(by=[group_col, time_col]) if group_col in df.columns else df.sort_values(by=time_col)
        
        if group_col in df_sorted.columns:
            pct_change = df_sorted.groupby(group_col)[value_col].pct_change().abs()
        else:
            pct_change = df_sorted[value_col].pct_change().abs()
            
        df_sorted['Pct_Change_Spike'] = pct_change
        spikes = df_sorted[df_sorted['Pct_Change_Spike'] > max_pct_change]
        return spikes