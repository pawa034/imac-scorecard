# IMAC Scorecard Usage Guide

## Installation

```bash
pip install imac_scorecard
```

## Quick Start

```python
from imac_scorecard import ScoreCardRisk
import pandas as pd

# Load your data
data = pd.read_csv('your_data.csv')

# Initialize the scorecard
scorecard = ScoreCardRisk(
    project_name="my_project",
    response_var="default_flag",  # Your target variable
    unique_id="customer_id",      # Unique identifier column
    iv_cutoff=0.01,              # Minimum Information Value to keep a feature
    corr_cutoff=0.65             # Maximum correlation between features
)

# Build the scorecard
result, model_path, selected_features, _, processed_data = scorecard.build_scorecard(data)
```

## Feature Selection

The package uses multiple methods for feature selection:

1. Information Value (IV):
```python
iv_scores = scorecard.calculate_iv_scores(data)
print("Top features by IV:")
for feature, iv in sorted(iv_scores.items(), key=lambda x: x[1], reverse=True)[:5]:
    print(f"{feature}: {iv:.4f}")
```

2. Feature Importance:
```python
importance_scores = scorecard.calculate_feature_importance(data)
print("Top features by importance:")
for feature, importance in sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)[:5]:
    print(f"{feature}: {importance:.4f}")
```

3. Correlation Analysis:
```python
corr_matrix = scorecard.calculate_correlation_matrix(data)
print("Correlation matrix:")
print(corr_matrix)
```

## Model Building

The package supports multiple models:
- Logistic Regression
- Random Forest
- Gradient Boosting
- K-Nearest Neighbors
- Naive Bayes
- Multi-Layer Perceptron

```python
# Train and compare models
model_results = scorecard.train_models(X, y)
print("Model comparison:")
print(model_results)
```

## Model Evaluation

1. Basic Metrics:
```python
metrics = scorecard.evaluate_models(X, y)
print("Model performance:")
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")
```

2. KS Table:
```python
ks_table = scorecard.calculate_ks_table(y_true, y_pred)
print("KS Table:")
print(ks_table)
```

## Weight of Evidence (WOE) Binning

```python
# Build WOE bins for a feature
woe_bins = scorecard.build_woe_bins(data, 'age', n_bins=10)
print("WOE bins for age:")
print(woe_bins)
```

## Best Practices

1. Data Preparation:
   - Handle missing values before passing data to the scorecard
   - Ensure numeric features are properly scaled
   - Convert categorical variables to appropriate format

2. Feature Selection:
   - Start with a higher IV cutoff and adjust based on results
   - Review correlation matrix to understand feature relationships
   - Use domain knowledge to validate selected features

3. Model Evaluation:
   - Always check both training and test performance
   - Pay attention to the KS statistic
   - Review feature importance to ensure model interpretability

## Example with Real Data

```python
import pandas as pd
from imac_scorecard import ScoreCardRisk

# Load data
data = pd.read_csv('credit_data.csv')

# Initialize scorecard
scorecard = ScoreCardRisk(
    project_name="credit_risk",
    response_var="default_flag",
    unique_id="customer_id",
    iv_cutoff=0.02,
    corr_cutoff=0.7
)

# Calculate feature importance
importance_scores = scorecard.calculate_feature_importance(data)
print("\nTop 10 features by importance:")
for feature, importance in sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"{feature}: {importance:.4f}")

# Build scorecard
result, model_path, selected_features, _, processed_data = scorecard.build_scorecard(data)

# Print model performance
print("\nSelected features:", selected_features)
print("\nModel saved to:", model_path)

# Calculate KS table
y_pred = result.predict_proba(processed_data[selected_features])[:, 1]
ks_table = scorecard.calculate_ks_table(processed_data[scorecard.response_var], y_pred)
print("\nKS Table:")
print(ks_table)
```

## Troubleshooting

Common issues and solutions:

1. No features selected:
   - Lower the IV cutoff
   - Check feature distributions
   - Verify target variable encoding

2. Poor model performance:
   - Review feature engineering
   - Adjust model parameters
   - Check for data quality issues

3. Memory issues:
   - Reduce number of features
   - Sample data for initial analysis
   - Use efficient data types

## Contributing

We welcome contributions! Please see our contribution guidelines for more information. 