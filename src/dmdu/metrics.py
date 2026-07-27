import numpy as np
import pandas as pd

def calculate_density(y_true: np.ndarray, mask: np.ndarray) -> float:
    """Calculates the concentration of target cases (vulnerabilities) within a selected subspace."""
    if not np.any(mask):
        return 0.0
    return np.mean(y_true[mask])

def calculate_coverage(y_true: np.ndarray, mask: np.ndarray) -> float:
    """Calculates the proportion of total target cases captured within a selected subspace."""
    total_targets = np.sum(y_true)
    if total_targets == 0:
        return 0.0
    return np.sum(y_true[mask]) / total_targets

def evaluate_box(X: pd.DataFrame, y: pd.Series, box_limits: dict) -> dict:
    """Evaluates a hyper-rectangular box defined by feature limits."""
    mask = np.ones(len(X), dtype=bool)
    for feature, (low, high) in box_limits.items():
        mask &= (X[feature] >= low) & (X[feature] <= high)
        
    density = calculate_density(y.values, mask)
    coverage = calculate_coverage(y.values, mask)
    
    return {
        "density": density,
        "coverage": coverage,
        "box_size": np.sum(mask),
        "total_samples": len(X)
    }