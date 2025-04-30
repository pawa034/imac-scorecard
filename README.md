# IMAC Scorecard

A Python package for building and evaluating credit risk scorecards.

## Features

- Information Value (IV) calculation
- Feature importance using permutation importance
- Correlation analysis
- Multiple model evaluation (Logistic Regression, Random Forest, GBM)
- KS statistics calculation
- WOE binning
- Model performance metrics

## Installation

```bash
pip install imac-scorecard
```

## Usage

```python
from imac_scorecard import ScoreCardRisk

# Initialize the scorecard
scorecard = ScoreCardRisk(
    target='default',
    categorical_features=['feature1', 'feature2'],
    numerical_features=['feature3', 'feature4']
)

# Fit the scorecard
scorecard.fit(X_train, y_train)

# Evaluate the scorecard
scorecard.evaluate(X_test, y_test)
```

For more detailed usage examples, please refer to the [documentation](docs/usage.md).

## Development

1. Clone the repository:
```bash
git clone https://github.com/yourusername/imac-scorecard.git
cd imac-scorecard
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install development dependencies:
```bash
pip install -e ".[dev]"
```

4. Run tests:
```bash
pytest
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 