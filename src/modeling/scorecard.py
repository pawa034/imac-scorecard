import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from sklearn.preprocessing import KBinsDiscretizer
import pickle
import logging

logger = logging.getLogger(__name__)

class ScoreCardModel:
    def __init__(self, n_bins: int = 10, min_bin_size: float = 0.05):
        self.n_bins = n_bins
        self.min_bin_size = min_bin_size
        self.woe_dict: Dict[str, Dict[str, float]] = {}
        self.iv_dict: Dict[str, float] = {}
        self.binners: Dict[str, KBinsDiscretizer] = {}
        self.score_points: Dict[str, Dict[str, float]] = {}
        self.base_score = 600
        self.pdo = 20  # Points to Double Odds
        self.factor = self.pdo / np.log(2)
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'ScoreCardModel':
        """
        Fit the scorecard model
        """
        for column in X.columns:
            # Create bins
            self.binners[column] = self._create_bins(X[column])
            binned_column = self.binners[column].fit_transform(X[column].values.reshape(-1, 1))
            
            # Calculate WOE and IV
            self.woe_dict[column], self.iv_dict[column] = self._calculate_woe_iv(
                binned_column.flatten(), y
            )
            
            # Calculate score points
            self.score_points[column] = self._calculate_score_points(column)
            
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features to scorecard points
        """
        scores = pd.DataFrame(index=X.index)
        
        for column in X.columns:
            if column in self.binners:
                binned_values = self.binners[column].transform(
                    X[column].values.reshape(-1, 1)
                ).flatten()
                
                # Convert binned values to scores using numpy vectorization
                scores[column] = np.vectorize(lambda x: self.score_points[column].get(str(int(x)), 0))(binned_values)
                
        return scores
        
    def predict_score(self, X: pd.DataFrame) -> pd.Series:
        """
        Calculate final credit score
        """
        scores = self.transform(X)
        return self.base_score + scores.sum(axis=1)
        
    def _create_bins(self, series: pd.Series) -> KBinsDiscretizer:
        """Create bins for continuous variables"""
        binner = KBinsDiscretizer(
            n_bins=self.n_bins,
            encode='ordinal',
            strategy='quantile'
        )
        return binner
        
    def _calculate_woe_iv(
        self,
        binned_feature: np.ndarray,
        target: pd.Series
    ) -> Tuple[Dict[str, float], float]:
        """Calculate Weight of Evidence and Information Value"""
        woe_dict = {}
        iv = 0
        
        for bin_val in range(self.n_bins):
            bin_mask = (binned_feature == bin_val)
            
            # Calculate good and bad counts
            good = np.sum((target == 0) & bin_mask) + 0.5  # Add 0.5 for smoothing
            bad = np.sum((target == 1) & bin_mask) + 0.5
            
            # Calculate proportions
            good_prop = good / np.sum(target == 0)
            bad_prop = bad / np.sum(target == 1)
            
            # Calculate WOE
            woe = np.log(good_prop / bad_prop)
            woe_dict[str(bin_val)] = woe
            
            # Calculate IV
            iv += (good_prop - bad_prop) * woe
            
        return woe_dict, iv
        
    def _calculate_score_points(self, column: str) -> Dict[str, float]:
        """Calculate score points for each bin"""
        score_points = {}
        woe_values = self.woe_dict[column]
        
        for bin_val, woe in woe_values.items():
            score = -woe * self.factor
            score_points[bin_val] = score
            
        return score_points
        
    def save_model(self, path: str):
        """Save the model to disk"""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
            
    @classmethod
    def load_model(cls, path: str) -> 'ScoreCardModel':
        """Load the model from disk"""
        with open(path, 'rb') as f:
            return pickle.load(f) 