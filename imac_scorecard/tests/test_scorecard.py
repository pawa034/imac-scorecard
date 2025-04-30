import pytest
import pandas as pd
import numpy as np
from imac_scorecard import ScoreCardRisk

@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    np.random.seed(42)
    n_samples = 1000
    
    return pd.DataFrame({
        'customer_id': range(n_samples),
        'age': np.random.randint(18, 70, n_samples),
        'income': np.random.normal(50000, 15000, n_samples),
        'credit_score': np.random.normal(650, 100, n_samples),
        'num_delinquencies': np.random.poisson(0.5, n_samples),
        'employment_duration': np.random.exponential(5, n_samples),
        'default_flag': np.random.binomial(1, 0.1, n_samples),
        'categorical_feature': np.random.choice(['A', 'B', 'C'], n_samples)
    })

@pytest.fixture
def scorecard():
    """Create a ScoreCardRisk instance"""
    return ScoreCardRisk(
        project_name="test_project",
        response_var="default_flag",
        unique_id="customer_id",
        iv_cutoff=0.01,
        corr_cutoff=0.65
    )

def test_initialization(scorecard):
    """Test ScoreCardRisk initialization"""
    assert scorecard.project_name == "test_project"
    assert scorecard.response_var == "default_flag"
    assert scorecard.unique_id == "customer_id"
    assert scorecard.iv_cutoff == 0.01
    assert scorecard.corr_cutoff == 0.65

def test_preprocess_data(scorecard, sample_data):
    """Test data preprocessing"""
    processed_data = scorecard.preprocess_data(sample_data)
    
    # Check if categorical variables are encoded
    assert pd.api.types.is_numeric_dtype(processed_data['categorical_feature'])
    
    # Check if no missing values remain
    assert processed_data.isnull().sum().sum() == 0
    
    # Check if original data is unchanged
    assert 'categorical_feature' in sample_data.columns
    assert pd.api.types.is_object_dtype(sample_data['categorical_feature'])

def test_calculate_iv(scorecard, sample_data):
    """Test Information Value calculation"""
    iv_scores = scorecard.calculate_iv_scores(sample_data)
    
    # Check if IV scores are calculated for all features
    expected_features = set(sample_data.columns) - {scorecard.response_var, scorecard.unique_id}
    assert set(iv_scores.keys()) == expected_features
    
    # Check if IV scores are non-negative
    assert all(iv >= 0 for iv in iv_scores.values())

def test_feature_importance(scorecard, sample_data):
    """Test feature importance calculation"""
    importance_scores = scorecard.calculate_feature_importance(sample_data)
    
    # Check if importance scores are calculated for all features
    expected_features = set(sample_data.columns) - {scorecard.response_var, scorecard.unique_id}
    assert set(importance_scores.keys()) == expected_features
    
    # Check if importance scores are non-negative
    assert all(importance >= 0 for importance in importance_scores.values())

def test_correlation_matrix(scorecard, sample_data):
    """Test correlation matrix calculation"""
    corr_matrix = scorecard.calculate_correlation_matrix(sample_data)
    
    # Check matrix dimensions
    expected_features = [col for col in sample_data.columns 
                        if col not in [scorecard.response_var, scorecard.unique_id]]
    assert corr_matrix.shape == (len(expected_features), len(expected_features))
    
    # Check if matrix is symmetric
    assert (corr_matrix == corr_matrix.T).all().all()
    
    # Check if diagonal is all ones
    assert (np.diag(corr_matrix) == 1).all()

def test_model_evaluation(scorecard, sample_data):
    """Test model evaluation metrics"""
    X = sample_data.drop([scorecard.response_var, scorecard.unique_id], axis=1)
    y = sample_data[scorecard.response_var]
    
    metrics = scorecard.evaluate_models(X, y)
    
    # Check if all metrics are present
    expected_metrics = {'roc_auc', 'precision', 'recall', 'f1'}
    assert set(metrics.keys()) == expected_metrics
    
    # Check if metrics are in valid range [0, 1]
    assert all(0 <= metric <= 1 for metric in metrics.values())

def test_build_scorecard(scorecard, sample_data):
    """Test full scorecard building process"""
    result, model_path, cols_to_use, mlvarlist, processed_data = scorecard.build_scorecard(sample_data)
    
    # Check if model is saved
    assert model_path == 'models/scorecard_model.pkl'
    
    # Check if features are selected
    assert len(cols_to_use) > 0
    assert all(col in sample_data.columns for col in cols_to_use)
    
    # Check if processed data has correct shape
    assert processed_data.shape[0] == sample_data.shape[0]
    assert all(col in processed_data.columns for col in cols_to_use)

def test_woe_bins(scorecard, sample_data):
    """Test Weight of Evidence binning"""
    feature = 'age'
    woe_bins = scorecard.build_woe_bins(sample_data, feature)
    
    # Check if bins are created
    assert len(woe_bins) > 0
    
    # Check if required statistics are calculated
    expected_stats = {'total', 'bad', 'good', 'bad_rate', 'woe'}
    assert all(stat in woe_bins for stat in expected_stats)

def test_ks_table(scorecard, sample_data):
    """Test KS table calculation"""
    y_true = sample_data[scorecard.response_var]
    y_pred = np.random.random(len(y_true))  # Random predictions for testing
    
    ks_table = scorecard.calculate_ks_table(y_true, y_pred)
    
    # Check if table has correct columns
    expected_columns = {'count', 'events', 'event_rate', 'cum_events', 'cum_non_events', 'ks'}
    assert set(ks_table.columns) == expected_columns
    
    # Check if KS values are in valid range [0, 100]
    assert all(0 <= ks <= 100 for ks in ks_table['ks']) 