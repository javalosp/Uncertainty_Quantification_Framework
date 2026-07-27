import numpy as np
import pandas as pd
from typing import Union, List, Dict, Optional
from sklearn.tree import DecisionTreeClassifier, _tree
from .base_discovery import BaseDiscovery


class CARTAnalyser(BaseDiscovery):
    """
    Classification and Regression Trees (CART) for Scenario Discovery.
    Fits decision trees to separate vulnerable futures and extracts hyper-rectangular bounding rules.
    """
    def __init__(self, threshold: Optional[float] = None, threshold_type: str = 'less',
                 max_depth: int = 4, min_samples_leaf: Union[int, float] = 0.05, class_weight: Optional[str] = 'balanced'):
        super().__init__(threshold=threshold, threshold_type=threshold_type)
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.class_weight = class_weight
        self.tree_model = None

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray], **kwargs):
        """Fits the CART decision tree algorithm to identify vulnerability split thresholds."""
        if isinstance(X, pd.DataFrame):
            self.X_data = X.copy()
            self.feature_names = X.columns.tolist()
        else:
            self.feature_names = [f"Param_{i}" for i in range(X.shape[1])]
            self.X_data = pd.DataFrame(X, columns=self.feature_names)
            
        self.y_data = np.array(y, dtype=float).ravel()
        self.y_binary = self._prepare_target(self.y_data)
        
        if self.y_binary.sum() == 0:
            raise ValueError("[CARTAnalyser] Zero vulnerable cases detected. Check threshold criteria.")
            
        # Convert min_samples_leaf percentage to int if float
        min_leaf = int(self.min_samples_leaf * len(self.y_binary)) if isinstance(self.min_samples_leaf, float) and self.min_samples_leaf < 1.0 else int(self.min_samples_leaf)
        min_leaf = max(1, min_leaf)
        
        self.tree_model = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_leaf=min_leaf,
            class_weight=self.class_weight,
            random_state=42
        )
        self.tree_model.fit(self.X_data, self.y_binary)
        
        self.find_boxes(**kwargs)
        self.is_fitted = True
        return self

    def find_boxes(self, min_density: float = 0.50, **kwargs) -> List[Dict]:
        """Traverses the decision tree structure and extracts candidate bounding boxes from leaf nodes."""
        if self.tree_model is None:
            raise RuntimeError("[CARTAnalyser] Decision tree not fitted. Call fit() first.")
            
        tree_ = self.tree_model.tree_
        feature_names = [self.feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined" for i in tree_.feature]
        
        extracted_boxes = []
        
        def recurse(node, box_min, box_max):
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_names[node]
                threshold = tree_.threshold[node]
                
                # Left child: <= threshold
                left_max = box_max.copy()
                left_max[name] = min(left_max[name], threshold)
                recurse(tree_.children_left[node], box_min, left_max)
                
                # Right child: > threshold
                right_min = box_min.copy()
                right_min[name] = max(right_min[name], threshold)
                recurse(tree_.children_right[node], right_min, box_max)
            else:
                # Leaf node evaluation
                metrics = self._evaluate_box(box_min, box_max)
                metrics['node_id'] = int(node)
                metrics['box_min'] = box_min.copy()
                metrics['box_max'] = box_max.copy()
                
                if metrics['density'] >= min_density and metrics['samples'] > 0:
                    extracted_boxes.append(metrics)
                    
        recurse(0, self.X_data.min().to_dict(), self.X_data.max().to_dict())
        
        # Sort boxes by F1 score then Density descending
        self.boxes_ = sorted(extracted_boxes, key=lambda b: (b['f1_score'], b['density']), reverse=True)
        return self.boxes_

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predicts binary vulnerability status using the underlying CART decision tree."""
        if not self.is_fitted or self.tree_model is None:
            raise RuntimeError("[CARTAnalyser] Cannot predict before calling fit().")
            
        if isinstance(X, pd.DataFrame):
            X_arr = X[self.feature_names].values
        else:
            X_arr = np.array(X, dtype=float)
            
        return self.tree_model.predict(X_arr)

    def get_feature_importances(self) -> pd.DataFrame:
        """Returns Gini feature importance scores identifying the primary drivers of vulnerability."""
        if not self.is_fitted or self.tree_model is None:
            raise RuntimeError("[CARTAnalyser] Cannot extract importances before fit().")
            
        imp = self.tree_model.feature_importances_
        return pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': imp
        }).sort_values(by='Importance', ascending=False).reset_index(drop=True)