import pandas as pd
import numpy as np
from typing import List, Dict
from sklearn.feature_selection import SelectKBest, f_classif
import logging

logger = logging.getLogger(__name__)

class FeatureSelector:
    def __init__(self, k: int = 20):
        self.k = k
        self.selected_features: List[str] = []
        self.feature_scores: Dict[str, float] = {}
        self.selector = SelectKBest(score_func=f_classif, k=k)
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'FeatureSelector':
        """
        Fit the feature selector
        """
        # Remove constant and quasi-constant features
        self._remove_low_variance(X)
        
        # Select K best features
        self.selector.fit(X[self.selected_features], y)
        
        # Get feature scores
        scores = self.selector.scores_
        for feature, score in zip(self.selected_features, scores):
            self.feature_scores[feature] = score
            
        # Sort features by importance
        self.selected_features = [
            feature for feature, _ in sorted(
                self.feature_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )[:self.k]
        ]
        
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data using selected features
        """
        return X[self.selected_features]
        
    def _remove_low_variance(self, X: pd.DataFrame, threshold: float = 0.01) -> None:
        """Remove constant and quasi-constant features"""
        # Calculate variance for each feature
        variances = X.var()
        
        # Keep features with variance above threshold
        self.selected_features = list(variances[variances > threshold].index)
        
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        return {
            feature: score 
            for feature, score in sorted(
                self.feature_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
        } 