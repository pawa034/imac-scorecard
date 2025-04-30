import pandas as pd
import numpy as np
from imac_scorecard import ScoreCardRisk

# Create sample data
np.random.seed(42)
n_samples = 1000

data = pd.DataFrame({
    'customer_id': range(n_samples),
    'age': np.random.randint(18, 70, n_samples),
    'income': np.random.normal(50000, 15000, n_samples),
    'credit_score': np.random.normal(650, 100, n_samples),
    'num_delinquencies': np.random.poisson(0.5, n_samples),
    'employment_duration': np.random.exponential(5, n_samples),
    'default_flag': np.random.binomial(1, 0.1, n_samples)
})

# Initialize scorecard
scorecard = ScoreCardRisk(
    project_name="test_project",
    response_var="default_flag",
    unique_id="customer_id",
    iv_cutoff=0.01,
    corr_cutoff=0.65
)

# Test IV calculation
print("Testing IV calculation...")
iv_scores = scorecard.calculate_iv_scores(data)
print("Top 5 features by IV:")
for feature, iv in sorted(iv_scores.items(), key=lambda x: x[1], reverse=True)[:5]:
    print(f"{feature}: {iv:.4f}")

# Test feature importance
print("\nTesting feature importance...")
importance_scores = scorecard.calculate_feature_importance(data)
print("Top 5 features by importance:")
for feature, importance in sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)[:5]:
    print(f"{feature}: {importance:.4f}")

# Test correlation matrix
print("\nTesting correlation matrix...")
corr_matrix = scorecard.calculate_correlation_matrix(data)
print("Correlation matrix:")
print(corr_matrix)

# Test model evaluation
print("\nTesting model evaluation...")
X = data.drop(['customer_id', 'default_flag'], axis=1)
y = data['default_flag']
metrics = scorecard.evaluate_models(X, y)
print("Model metrics:")
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")

# Test full scorecard building
print("\nTesting full scorecard building...")
result, model_path, cols_to_use, mlvarlist, processed_data = scorecard.build_scorecard(data)
print(f"Model saved to: {model_path}")
print(f"Selected features: {cols_to_use}") 