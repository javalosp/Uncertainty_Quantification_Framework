import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, WhiteKernel, Matern


class GPRUncertaintyModel:
    """Gaussian Process Regression model for probabilistic time-series forecasting."""

    def __init__(self, kernel=None, alpha=1e-10, n_restarts_optimizer=5):
        if kernel is None:
            kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, nu=1.5) + WhiteKernel(
                noise_level=1e-2
            )

        self.gpr = GaussianProcessRegressor(
            kernel=kernel,
            alpha=alpha,
            n_restarts_optimizer=n_restarts_optimizer,
            normalize_y=True,
            random_state=42,
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit Gaussian Process to training data, ensuring 2D feature matrix formatting."""
        X_arr = np.array(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        y_arr = np.array(y, dtype=float).ravel()
        self.gpr.fit(X_arr, y_arr)
        return self

    def predict(self, X: np.ndarray, return_std: bool = True):
        """Predict mean and uncertainty, ensuring 2D feature matrix formatting."""
        X_arr = np.array(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        if return_std:
            mean, std = self.gpr.predict(X_arr, return_std=True)
            return mean, std
        return self.gpr.predict(X_arr, return_std=False)