import pandas as pd
import numpy as np
from typing import Tuple, List
import yaml
import logging

logger = logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
    """Load configuration from yaml file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def split_train_test(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data into train and test sets"""
    from sklearn.model_selection import train_test_split
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

def evaluate_model(y_true: pd.Series, scores: pd.Series) -> dict:
    """Evaluate scorecard model performance"""
    from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
    
    # Convert scores to binary predictions using median as threshold
    threshold = scores.median()
    y_pred = (scores <= threshold).astype(int)
    
    return {
        'roc_auc': roc_auc_score(y_true, -scores),  # Negative scores because lower score = higher risk
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred)
    }

def plot_score_distribution(scores: pd.Series, y_true: pd.Series = None):
    """Plot the distribution of credit scores"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.figure(figsize=(10, 6))
    
    if y_true is not None:
        # Plot distribution by class
        sns.kdeplot(data=pd.DataFrame({'score': scores, 'target': y_true}),
                   x='score', hue='target', common_norm=False)
        plt.title('Score Distribution by Class')
    else:
        # Plot overall distribution
        sns.histplot(scores, bins=50)
        plt.title('Overall Score Distribution')
    
    plt.xlabel('Credit Score')
    plt.ylabel('Density')
    plt.show() 