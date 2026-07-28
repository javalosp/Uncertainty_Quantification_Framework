import pandas as pd
import numpy as np
from typing import Dict, Any, List

class DataProfiler:
    """Evaluates temporal continuity, uniqueness, and physical boundary constraints."""

    @staticmethod
    def profile(df: pd.DataFrame, flow_id_col: str = 'Parameter_ID', time_col: str = 'Year', value_col: str = 'Published_Mean') -> Dict[str, Any]:
        """
        Runs comprehensive profiling across all time series in the dataset.
        Returns a dictionary containing metrics, flags, and diagnostic warnings.
        """
        warnings = []
        metrics = {
            'total_records': len(df),
            'unique_flows': df[flow_id_col].nunique() if flow_id_col in df.columns else 1,
            'duplicate_keys': 0,
            'negative_flows': 0,
            'temporal_gaps': {}
        }

        # 1. Duplicate Key Check (Flow_ID + Year must be unique)
        if flow_id_col in df.columns and time_col in df.columns:
            dups = df.duplicated(subset=[flow_id_col, time_col], keep=False)
            metrics['duplicate_keys'] = int(dups.sum())
            if metrics['duplicate_keys'] > 0:
                warnings.append(f"Found {metrics['duplicate_keys']} duplicate records for ({flow_id_col}, {time_col}) pairs.")

        # 2. Physical Non-Negativity Check (Mass flows cannot be negative)
        if value_col in df.columns and pd.api.types.is_numeric_dtype(df[value_col]):
            neg_count = int((df[value_col] < 0.0).sum())
            metrics['negative_flows'] = neg_count
            if neg_count > 0:
                warnings.append(f"Physical boundary violation: {neg_count} records contain negative '{value_col}' values.")

        # 3. Temporal Continuity Check (Identify missing sequence years)
        if flow_id_col in df.columns and time_col in df.columns:
            for flow_id, group in df.groupby(flow_id_col):
                years = sorted(group[time_col].dropna().astype(int).unique())
                if len(years) > 1:
                    expected_years = set(range(min(years), max(years) + 1))
                    actual_years = set(years)
                    missing_years = sorted(expected_years - actual_years)
                    if missing_years:
                        metrics['temporal_gaps'][str(flow_id)] = missing_years
                        warnings.append(f"Flow '{flow_id}' has missing temporal steps in years: {missing_years}")

        metrics['warnings'] = warnings
        metrics['is_clean'] = len(warnings) == 0
        return metrics