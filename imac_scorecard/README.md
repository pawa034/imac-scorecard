# IMAC Scorecard

A Python package for building and evaluating credit risk scorecards using the IMAC methodology.

## Installation

```bash
pip install imac_scorecard
```

## Usage

```python
from imac_scorecard import ScoreCardRisk

# Initialize the scorecard
scorecard = ScoreCardRisk(
    project_name="my_project",
    response_var="default_flag",
    unique_id="customer_id",
    iv_cutoff=0.01,
    corr_cutoff=0.65
)

# Build the scorecard
result, model_path, cols_to_use, mlvarlist, processed_data = scorecard.build_scorecard(data)

# Get feature importance
importance_scores = scorecard.calculate_feature_importance(data)

# Calculate IV scores
iv_scores = scorecard.calculate_iv_scores(data)

# Evaluate models
metrics = scorecard.evaluate_models(processed_data[cols_to_use], processed_data[response_var])
```

## Features

- Information Value (IV) calculation
- Feature importance using permutation importance
- Correlation analysis
- Multiple model evaluation (Logistic Regression, Random Forest, GBM)
- KS statistics calculation
- Model performance metrics (ROC AUC, Precision, Recall, F1)
- HTML report generation

## License

MIT 