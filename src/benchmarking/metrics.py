import numpy as np
from scipy.stats import norm

class ProbabilisticMetrics:
    """Collection of rigorous probabilistic evaluation metrics for UQ benchmarking."""
    
    @staticmethod
    def crps_gaussian(mean: np.ndarray, std: np.ndarray, y_true: np.ndarray) -> float:
        """Computes the analytical Continuous Ranked Probability Score (CRPS) for Gaussian distributions."""
        mean = np.array(mean, dtype=float).ravel()
        std = np.maximum(np.array(std, dtype=float).ravel(), 1e-8)
        y_true = np.array(y_true, dtype=float).ravel()
        
        z = (y_true - mean) / std
        crps_vals = std * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1.0 / np.sqrt(np.pi))
        return float(np.mean(crps_vals))

    @staticmethod
    def crps_empirical(samples: np.ndarray, y_true: np.ndarray) -> float:
        """Computes the empirical energy score CRPS for ensemble/sample-based predictions."""
        samples = np.array(samples, dtype=float)
        if samples.ndim == 1:
            samples = samples[:, None]
        y_true = np.array(y_true, dtype=float).ravel()
        
        M = samples.shape[0]
        term1 = np.mean(np.abs(samples - y_true), axis=0)
        
        sorted_samples = np.sort(samples, axis=0)
        diff = np.diff(sorted_samples, axis=0)
        weights = np.arange(1, M) * np.arange(M - 1, 0, -1)
        term2 = np.sum(diff * weights[:, None], axis=0) / (M ** 2)
        
        return float(np.mean(term1 - term2))

    @staticmethod
    def coverage_probability(mean: np.ndarray, std: np.ndarray, y_true: np.ndarray, nominal_level: float = 0.95) -> float:
        """Computes empirical Prediction Interval Coverage Probability (PICP)."""
        mean = np.array(mean, dtype=float).ravel()
        std = np.maximum(np.array(std, dtype=float).ravel(), 1e-8)
        y_true = np.array(y_true, dtype=float).ravel()
        
        z_score = norm.ppf(1.0 - (1.0 - nominal_level) / 2.0)
        lower = mean - z_score * std
        upper = mean + z_score * std
        
        covered = (y_true >= lower) & (y_true <= upper)
        return float(np.mean(covered))

    @staticmethod
    def mean_interval_width(std: np.ndarray, nominal_level: float = 0.95) -> float:
        """Computes the Mean Interval Width (MIW) to evaluate forecast sharpness."""
        std = np.maximum(np.array(std, dtype=float).ravel(), 1e-8)
        z_score = norm.ppf(1.0 - (1.0 - nominal_level) / 2.0)
        return float(np.mean(2.0 * z_score * std))

    @classmethod
    def evaluate_all(cls, mean: np.ndarray, std: np.ndarray, y_true: np.ndarray, nominal_level: float = 0.95, samples: np.ndarray = None) -> dict:
        """Computes a complete suite of probabilistic and point metrics."""
        mean = np.array(mean, dtype=float).ravel()
        y_true = np.array(y_true, dtype=float).ravel()
        
        mae = float(np.mean(np.abs(y_true - mean)))
        rmse = float(np.sqrt(np.mean((y_true - mean) ** 2)))
        picp = cls.coverage_probability(mean, std, y_true, nominal_level)
        miw = cls.mean_interval_width(std, nominal_level)
        
        if samples is not None:
            crps = cls.crps_empirical(samples, y_true)
        else:
            crps = cls.crps_gaussian(mean, std, y_true)
            
        return {
            'MAE': mae,
            'RMSE': rmse,
            'CRPS': crps,
            f'PICP_{int(nominal_level*100)}%': picp,
            'Sharpness_MIW': miw
        }