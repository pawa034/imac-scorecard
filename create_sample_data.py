import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data
n_samples = 10000
n_features = 20

# Create feature names similar to your actual data
feature_names = [
    'customer_id',
    'age',
    'income',
    'credit_score',
    'loan_amount',
    'loan_term',
    'employment_duration',
    'debt_ratio',
    'num_credit_lines',
    'num_late_payments',
    'utilization_rate',
    'monthly_payment',
    'interest_rate',
    'num_inquiries',
    'num_delinquencies',
    'years_credit_history',
    'num_recent_loans',
    'savings_balance',
    'checking_balance',
    'employment_type'
]

# Generate synthetic classification data
X, y = make_classification(
    n_samples=n_samples,
    n_features=n_features-1,  # -1 because we'll add customer_id separately
    n_informative=10,
    n_redundant=5,
    n_classes=2,
    random_state=42
)

# Create DataFrame
df = pd.DataFrame(X, columns=feature_names[1:])  # Skip customer_id

# Add customer_id
df['customer_id'] = np.arange(1000000, 1000000 + n_samples)

# Add some random categorical data for employment_type
employment_types = ['SALARIED', 'SELF_EMPLOYED', 'BUSINESS', 'OTHER']
df['employment_type'] = np.random.choice(employment_types, size=n_samples)

# Add the target column
df['default_flag'] = y

# Add some missing values
for col in df.columns:
    if col not in ['customer_id', 'default_flag']:
        mask = np.random.random(n_samples) < 0.05  # 5% missing values
        df.loc[mask, col] = np.nan

# Save to CSV
df.to_csv('sample_demo.csv', index=False)

print("Sample data created with shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nSample of first few rows:")
print(df.head())
print("\nMissing values summary:")
print(df.isnull().sum()) 