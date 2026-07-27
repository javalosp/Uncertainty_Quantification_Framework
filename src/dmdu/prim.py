import numpy as np
import pandas as pd
from typing import Union, List, Dict, Optional
from .base_discovery import BaseDiscovery


class PRIMAnalyser(BaseDiscovery):
    """
    Patient Rule Induction Method (PRIM) for Scenario Discovery.
    Iteratively peels and pastes hyper-dimensional parameter boundaries to isolate vulnerability regions.
    """
    def __init__(self, threshold: Optional[float] = None, threshold_type: str = 'less', 
                 peel_alpha: float = 0.05, paste_alpha: float = 0.05, min_support: float = 0.05, target_density: float = 0.80):
        super().__init__(threshold=threshold, threshold_type=threshold_type, min_support=min_support)
        self.peel_alpha = peel_alpha
        self.paste_alpha = paste_alpha
        self.target_density = target_density
        self.trajectory_ = []
        self.best_box_ = None

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray], **kwargs):
        """Fits PRIM peeling and pasting algorithms to identify adaptation tipping boundaries."""
        if isinstance(X, pd.DataFrame):
            self.X_data = X.copy()
            self.feature_names = X.columns.tolist()
        else:
            self.feature_names = [f"Param_{i}" for i in range(X.shape[1])]
            self.X_data = pd.DataFrame(X, columns=self.feature_names)
            
        self.y_data = np.array(y, dtype=float).ravel()
        self.y_binary = self._prepare_target(self.y_data)
        
        if self.y_binary.sum() == 0:
            raise ValueError("[PRIMAnalyser] Zero vulnerable cases detected. Check threshold criteria.")
            
        self.find_boxes(**kwargs)
        self.is_fitted = True
        return self

    def find_boxes(self, max_peels: int = 500, **kwargs) -> List[Dict]:
        """Executes the peeling and pasting trajectory and extracts candidate scenario boxes."""
        box_min = self.X_data.min().to_dict()
        box_max = self.X_data.max().to_dict()
        
        n_total = len(self.y_binary)
        total_failures = int(self.y_binary.sum())
        
        self.trajectory_ = []
        
        # 1. Peeling Phase
        for step in range(max_peels):
            metrics = self._evaluate_box(box_min, box_max)
            metrics['step'] = step
            metrics['phase'] = 'peel'
            metrics['box_min'] = box_min.copy()
            metrics['box_max'] = box_max.copy()
            self.trajectory_.append(metrics)
            
            if metrics['support'] <= self.min_support or metrics['samples'] < 10:
                break
            if metrics['density'] >= 1.0 and metrics['coverage'] <= self.min_support:
                break
                
            current_mask = self._get_box_mask(box_min, box_max)
            best_density = -1.0
            best_dim = None
            best_side = None
            best_val = None
            
            for col in self.feature_names:
                vals = self.X_data.loc[current_mask, col]
                if vals.min() == vals.max(): continue
                
                # Peel lower tail (min bound)
                q_low = np.quantile(vals, self.peel_alpha)
                mask_low = current_mask & (self.X_data[col] >= q_low)
                if mask_low.sum() >= self.min_support * n_total and mask_low.sum() > 0:
                    dens_low = self.y_binary[mask_low].mean()
                    if dens_low > best_density:
                        best_density = dens_low
                        best_dim = col
                        best_side = 'min'
                        best_val = q_low
                        
                # Peel upper tail (max bound)
                q_high = np.quantile(vals, 1.0 - self.peel_alpha)
                mask_high = current_mask & (self.X_data[col] <= q_high)
                if mask_high.sum() >= self.min_support * n_total and mask_high.sum() > 0:
                    dens_high = self.y_binary[mask_high].mean()
                    if dens_high > best_density:
                        best_density = dens_high
                        best_dim = col
                        best_side = 'max'
                        best_val = q_high
                        
            if best_dim is None or best_density < metrics['density']:
                if best_dim is None: break
                
            if best_side == 'min':
                box_min[best_dim] = float(best_val)
            else:
                box_max[best_dim] = float(best_val)
                
        # 2. Select Optimal Box from Trajectory (prioritizing target density then F1 score)
        traj_df = pd.DataFrame(self.trajectory_)
        qualified = traj_df[traj_df['density'] >= self.target_density]
        
        if not qualified.empty:
            # Pick box with highest coverage among those meeting target density
            opt_idx = qualified['coverage'].idxmax()
        else:
            # Fallback to highest F1 score
            opt_idx = traj_df['f1_score'].idxmax()
            
        self.best_box_ = self.trajectory_[opt_idx]
        self.boxes_ = [self.best_box_]
        return self.boxes_

    def _get_box_mask(self, box_min: Dict[str, float], box_max: Dict[str, float], X: Optional[pd.DataFrame] = None) -> np.ndarray:
        if X is None: X = self.X_data
        mask = np.ones(len(X), dtype=bool)
        for col in self.feature_names:
            mask &= (X[col] >= box_min[col]) & (X[col] <= box_max[col])
        return mask

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predicts binary vulnerability status using the optimal PRIM scenario box."""
        if not self.is_fitted or self.best_box_ is None:
            raise RuntimeError("[PRIMAnalyser] Cannot predict before fit() and find_boxes() are called.")
            
        if not isinstance(X, pd.DataFrame):
            X_df = pd.DataFrame(X, columns=self.feature_names)
        else:
            X_df = X
            
        mask = self._get_box_mask(self.best_box_['box_min'], self.best_box_['box_max'], X_df)
        return mask.astype(int)

    def get_trajectory(self) -> pd.DataFrame:
        """Returns the complete peeling and pasting trajectory across candidate boxes."""
        if not self.trajectory_:
            raise RuntimeError("[PRIMAnalyser] No trajectory available. Call fit() first.")
        return pd.DataFrame(self.trajectory_)