import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Union, Dict, List, Optional, Tuple


class BaseDiscovery(ABC):
    """
    Abstract Base Class for Scenario Discovery and Vulnerability Analytics (DMDU).
    Establishes a uniform interface for identifying failure boundaries and adaptation tipping points.
    """
    def __init__(self, threshold: Optional[float] = None, threshold_type: str = 'less', min_support: float = 0.05):
        """
        :param threshold: Numerical value defining vulnerability / adaptation tipping point.
        :param threshold_type: Comparison operator ('less', 'less_equal', 'greater', 'greater_equal', or 'binary').
        :param min_support: Minimum fraction of total observations required in a candidate box.
        """
        self.threshold = threshold
        self.threshold_type = threshold_type.lower()
        self.min_support = min_support
        self.feature_names = []
        self.X_data = None
        self.y_data = None
        self.y_binary = None
        self.is_fitted = False
        self.boxes_ = []

    def _prepare_target(self, y: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """Converts raw target values into a binary vulnerability indicator (1 = Vulnerable/Failure, 0 = Desired)."""
        y_arr = np.array(y, dtype=float).ravel()
        
        if self.threshold_type == 'binary' or self.threshold is None:
            # Assume y is already 0/1 binary indicator
            return (y_arr > 0.5).astype(int)
            
        if self.threshold_type in ['less', 'lt']:
            return (y_arr < self.threshold).astype(int)
        elif self.threshold_type in ['less_equal', 'le']:
            return (y_arr <= self.threshold).astype(int)
        elif self.threshold_type in ['greater', 'gt']:
            return (y_arr > self.threshold).astype(int)
        elif self.threshold_type in ['greater_equal', 'ge']:
            return (y_arr >= self.threshold).astype(int)
        else:
            raise ValueError(f"Unsupported threshold_type: '{self.threshold_type}'.")

    def _evaluate_box(self, box_min: Dict[str, float], box_max: Dict[str, float], X: Optional[pd.DataFrame] = None, y_bin: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Computes statistical performance metrics (Coverage, Density, Support, F1) for a hyper-rectangular box."""
        if X is None: X = self.X_data
        if y_bin is None: y_bin = self.y_binary
        
        mask = np.ones(len(X), dtype=bool)
        for col in X.columns:
            mask &= (X[col] >= box_min[col]) & (X[col] <= box_max[col])
            
        n_samples = int(mask.sum())
        n_total = len(y_bin)
        total_failures = int(y_bin.sum())
        
        support = n_samples / n_total if n_total > 0 else 0.0
        
        if n_samples > 0:
            box_failures = int(y_bin[mask].sum())
            density = box_failures / n_samples
            coverage = box_failures / total_failures if total_failures > 0 else 0.0
        else:
            density = 0.0
            coverage = 0.0
            
        # Calculate harmonic mean (F1 score) of Coverage and Density
        f1_score = (2.0 * density * coverage) / (density + coverage) if (density + coverage) > 0 else 0.0
        
        return {
            'support': float(support),
            'density': float(density),
            'coverage': float(coverage),
            'samples': int(n_samples),
            'f1_score': float(f1_score)
        }

    @abstractmethod
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray], **kwargs):
        """Fits the scenario discovery algorithm to the multi-dimensional parameter space."""
        pass

    @abstractmethod
    def find_boxes(self, **kwargs) -> List[Dict]:
        """Identifies and returns candidate vulnerability bounding boxes."""
        pass

    @abstractmethod
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predicts binary vulnerability status (1 = Vulnerable, 0 = Safe) for new parameter samples."""
        pass

    def to_dataframe(self) -> pd.DataFrame:
        """Exports all discovered scenario boxes and their boundary thresholds as a formatted DataFrame."""
        if not self.boxes_:
            raise RuntimeError("[BaseDiscovery] No boxes discovered. Ensure fit() and find_boxes() have been called.")
            
        records = []
        for idx, b in enumerate(self.boxes_):
            row = {
                'Box_ID': idx,
                'Support': b.get('support', 0.0),
                'Density': b.get('density', 0.0),
                'Coverage': b.get('coverage', 0.0),
                'Samples': b.get('samples', 0),
                'F1_Score': b.get('f1_score', 0.0)
            }
            # Flatten boundary limits
            for col in self.feature_names:
                row[f"{col}_min"] = b['box_min'].get(col, np.nan)
                row[f"{col}_max"] = b['box_max'].get(col, np.nan)
            records.append(row)
            
        return pd.DataFrame(records)