from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

class BaseForecaster(ABC):
    """
    Abstract Base Class for classical time-series prediction and calibration models.
    Establishes a uniform interface for fitting models and exporting analytical uncertainty bounds.
    """
    def __init__(self, model_name: str, confidence_level: float = 0.95):
        """
        Args:
            model_name (str): Identifier for the forecasting algorithm (e.g., 'TBATS', 'Bayesian_MCMC').
            confidence_level (float): Target probability coverage for prediction intervals (default 95%).
        """
        self.model_name = model_name
        self.confidence_level = confidence_level
        self.is_fitted = False
        self.history_df = None
        self.forecast_df = None

    @abstractmethod
    def fit(self, df: pd.DataFrame, time_col: str = 'Year', value_col: str = 'Published_Mean', **kwargs):
        """
        Fits the forecasting algorithm to historical observations.
        
        Args:
            df (pd.DataFrame): Ingested time-series data from the Universal Ingestion Layer.
            time_col (str): Column name containing temporal indices.
            value_col (str): Column name containing observed historical values.
        """
        pass

    @abstractmethod
    def predict_intervals(self, horizon: int, **kwargs) -> pd.DataFrame:
        """
        Generates future predictions with analytical upper and lower confidence boundaries.
        
        Args:
            horizon (int): Number of future time steps (e.g., years) to project.
            
        Returns:
            pd.DataFrame: DataFrame containing columns ['Year', 'Mean_Forecast', 'Lower_Bound', 'Upper_Bound', 'Std_Dev'].
        """
        pass

    def to_universal_uncertainty(self, parameter_prefix: str, source_node: str, target_node: str, material: str = "Copper") -> pd.DataFrame:
        """
        Converts the forecast results into the 12-column Universal MFA Schema structure
        so they can be injected directly into the Hybrid IRS propagation engine.
        
        Args:
            parameter_prefix (str): Label prefix for Parameter_ID generation.
            source_node (str): Topological origin node name.
            target_node (str): Topological destination node name.
            material (str): Material identifier.
            
        Returns:
            pd.DataFrame: Standardised DataFrame ready for MFAAuditParser.
        """
        if not self.is_fitted or self.forecast_df is None:
            raise RuntimeError(f"[{self.model_name}] Model must be fitted and predicted before exporting to universal schema.")
            
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
                'Uncertainty_Class': 'aleatory',  # Classical statistical noise is modeled as Aleatory
                'Published_Mean': mean_val,
                'CV_or_StdDev': cv,
                'Bound_Min': float(row['Lower_Bound']),
                'Bound_Max': float(row['Upper_Bound']),
                'Data_Pedigree_Score': 2.0
            })
            
        return pd.DataFrame(records)