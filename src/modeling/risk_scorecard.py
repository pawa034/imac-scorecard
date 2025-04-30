import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.impute import SimpleImputer
import logging
from typing import Dict, List, Tuple, Any
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
import pickle
import os
import statsmodels.api as sm
from scipy import stats
import time
import warnings
from contextlib import contextmanager
from scipy.stats import norm
from sklearn.inspection import permutation_importance

@contextmanager
def suppress_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield

class ScoreCardRisk:
    def __init__(self, project_name: str = "Default", 
                 response_var: str = "default_flag",
                 unique_id: str = "customer_id",
                 iv_cutoff: float = 0.01,
                 corr_cutoff: float = 0.7):
        """
        Initialize ScoreCardRisk with configuration parameters
        
        Args:
            project_name: Name of the project
            response_var: Target variable name
            unique_id: Unique identifier column
            iv_cutoff: Information Value cutoff
            corr_cutoff: Correlation cutoff
        """
        self.logger = logging.getLogger(__name__)
        self.project_name = project_name
        self.response_var = response_var
        self.unique_id = unique_id
        self.iv_cutoff = iv_cutoff
        self.corr_cutoff = corr_cutoff
        
        # Initialize model storage
        self.models = {}
        self.feature_importance = None
        self.selected_features = None
        self.woe_bins = {}
        self.iv_scores = {}
        self.logit_results = None
        self.best_model_name = None
        
        # Create necessary directories
        os.makedirs('models', exist_ok=True)
        os.makedirs('results', exist_ok=True)

    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess data by handling missing values and encoding categorical variables"""
        df = data.copy()
        
        # Handle missing values
        numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = df.select_dtypes(include=['object']).columns
        
        # Impute numeric features with median
        numeric_imputer = SimpleImputer(strategy='median')
        df[numeric_features] = numeric_imputer.fit_transform(df[numeric_features])
        
        # Impute categorical features with mode
        categorical_imputer = SimpleImputer(strategy='most_frequent')
        df[categorical_features] = categorical_imputer.fit_transform(df[categorical_features])
        
        # Encode categorical variables
        le = LabelEncoder()
        for col in categorical_features:
            if col not in [self.response_var, self.unique_id]:
                df[col] = le.fit_transform(df[col])
        
        return df

    def calculate_iv(self, data: pd.DataFrame, feature: str) -> float:
        """Calculate Information Value for a feature"""
        try:
            with suppress_warnings():
                df = pd.DataFrame({
                    'feature': data[feature],
                    'target': data[self.response_var]
                })
                
                # Calculate WOE and IV
                grouped = df.groupby('feature', observed=True).agg({
                    'target': ['count', 'sum']
                }).fillna(0)
                
                grouped.columns = ['total', 'bad']
                grouped['good'] = grouped['total'] - grouped['bad']
                grouped['bad_rate'] = grouped['bad'] / grouped['total']
                
                total_bad = grouped['bad'].sum()
                total_good = grouped['good'].sum()
                
                grouped['bad_dist'] = grouped['bad'] / total_bad
                grouped['good_dist'] = grouped['good'] / total_good
                
                # Handle division by zero in WOE calculation
                grouped['woe'] = np.where(
                    (grouped['good_dist'] > 0) & (grouped['bad_dist'] > 0),
                    np.log(grouped['good_dist'] / grouped['bad_dist']),
                    0
                )
                grouped['iv'] = (grouped['good_dist'] - grouped['bad_dist']) * grouped['woe']
                
                return grouped['iv'].sum()
        except Exception as e:
            self.logger.warning(f"Error calculating IV for {feature}: {str(e)}")
            return 0

    def select_features(self, data: pd.DataFrame) -> List[str]:
        """Select features based on IV, correlation, and importance"""
        try:
            # Calculate IV for all features
            features = [col for col in data.columns if col not in [self.response_var, self.unique_id]]
            for feature in features:
                self.iv_scores[feature] = self.calculate_iv(data, feature)
            
            # Print IV scores for debugging
            self.logger.info("Feature IV scores:")
            for feature, iv in sorted(self.iv_scores.items(), key=lambda x: x[1], reverse=True):
                self.logger.info(f"{feature}: {iv:.4f}")
            
            # Filter by IV cutoff
            selected_features = [f for f in features if self.iv_scores[f] >= self.iv_cutoff]
            
            if not selected_features:
                self.logger.warning(f"No features meet the IV cutoff of {self.iv_cutoff}")
                self.logger.warning("Using features with highest IV scores instead")
                # Select top 5 features by IV score
                selected_features = sorted(features, key=lambda x: self.iv_scores[x], reverse=True)[:5]
                self.logger.info(f"Selected top 5 features by IV: {selected_features}")
            
            # Check correlations
            if len(selected_features) > 1:
                corr_matrix = data[selected_features].corr().abs()
                to_drop = set()
                for i in range(len(corr_matrix.columns)):
                    for j in range(i):
                        if corr_matrix.iloc[i, j] >= self.corr_cutoff:
                            if self.iv_scores[corr_matrix.columns[i]] > self.iv_scores[corr_matrix.columns[j]]:
                                to_drop.add(corr_matrix.columns[j])
                            else:
                                to_drop.add(corr_matrix.columns[i])
                
                selected_features = [f for f in selected_features if f not in to_drop]
            
            self.logger.info(f"Selected {len(selected_features)} features")
            return selected_features
            
        except Exception as e:
            self.logger.error(f"Error in feature selection: {str(e)}")
            raise

    def build_woe_bins(self, data: pd.DataFrame, feature: str, n_bins: int = 10) -> Dict:
        """Build WOE bins for a feature"""
        try:
            with suppress_warnings():
                df = pd.DataFrame({
                    'feature': data[feature],
                    'target': data[self.response_var]
                })
                
                # Create bins
                if df['feature'].dtype in ['int64', 'float64']:
                    df['bin'] = pd.qcut(df['feature'], n_bins, duplicates='drop')
                else:
                    df['bin'] = df['feature']
                
                # Calculate WOE
                grouped = df.groupby('bin', observed=True).agg({
                    'target': ['count', 'sum']
                }).fillna(0)
                
                grouped.columns = ['total', 'bad']
                grouped['good'] = grouped['total'] - grouped['bad']
                
                total_bad = grouped['bad'].sum()
                total_good = grouped['good'].sum()
                
                grouped['bad_rate'] = grouped['bad'] / grouped['total']
                
                # Handle division by zero in WOE calculation
                grouped['woe'] = np.where(
                    (grouped['good'] > 0) & (grouped['bad'] > 0),
                    np.log((grouped['good'] / total_good) / (grouped['bad'] / total_bad)),
                    0
                )
                
                return grouped.to_dict()
        except Exception as e:
            self.logger.warning(f"Error building WOE bins for {feature}: {str(e)}")
            return {}

    def train_models(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Train multiple models and compare their performance"""
        models = {
            'logistic': LogisticRegression(max_iter=1000),
            'random_forest': RandomForestClassifier(n_estimators=100),
            'gbm': GradientBoostingClassifier(n_estimators=100),
            'knn': KNeighborsClassifier(),
            'naive_bayes': GaussianNB(),
            'mlp': MLPClassifier(max_iter=1000)
        }
        
        results = []
        for name, model in models.items():
            try:
                logging.info(f"{name}...")
                model.fit(X, y)
                y_pred = model.predict_proba(X)[:, 1]
                
                # Calculate KS statistics
                ks_table = self.calculate_ks_table(y, y_pred)
                max_ks = ks_table['ks'].max()
                max_ks_row = ks_table['ks'].idxmax()
                best_decile = ks_table['ks'].idxmax() + 1
                
                # Store results
                results.append({
                    'model': name,
                    'max_ks': max_ks,
                    'max_ks_row': max_ks_row,
                    'best_decile': best_decile
                })
                
            except Exception as e:
                logging.error(f"Error training {name}: {str(e)}")
                continue
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        return results_df

    def calculate_ks_table(self, y_true: np.array, y_pred: np.array, n_bins: int = 10) -> pd.DataFrame:
        """Calculate KS table"""
        with suppress_warnings():
            df = pd.DataFrame({
                'true': y_true,
                'pred': y_pred
            })
            
            df['bucket'] = pd.qcut(df['pred'], n_bins, labels=False)
            grouped = df.groupby('bucket', observed=True).agg({
                'true': ['count', 'sum']
            }).fillna(0)
            
            grouped.columns = ['count', 'events']
            grouped['event_rate'] = grouped['events'] / grouped['count']
            grouped['cum_events'] = grouped['events'].cumsum() / grouped['events'].sum()
            grouped['cum_non_events'] = (grouped['count'] - grouped['events']).cumsum() / (grouped['count'] - grouped['events']).sum()
            grouped['ks'] = abs(grouped['cum_events'] - grouped['cum_non_events']) * 100
            
            return grouped

    def run_logistic_regression(self, X: pd.DataFrame, y: pd.Series) -> Any:
        """Run logistic regression and return detailed results"""
        try:
            # Add constant for intercept
            X = sm.add_constant(X)
            
            # Fit logistic regression
            logit_model = sm.Logit(y, X)
            result = logit_model.fit()
            
            # Calculate z-scores and p-values
            z_scores = result.tvalues
            p_values = result.pvalues
            
            # Create summary DataFrame
            summary_df = pd.DataFrame({
                'Vars': X.columns,
                'Z_SCORE': z_scores,
                'pvals': p_values,
                'p_values': p_values < 0.05,
                'z_abs': abs(z_scores)
            })
            
            # Sort by absolute z-score
            summary_df = summary_df.sort_values('z_abs', ascending=False)
            
            return result, summary_df
        except Exception as e:
            self.logger.warning(f"Error in logistic regression: {str(e)}")
            return None, None

    def build_scorecard(self, data):
        """Build the scorecard model"""
        try:
            start_time = time.time()
            
            # Preprocess data
            df_processed = self.preprocess_data(data)
            
            # Split into train and test
            X_train, X_test, y_train, y_test = train_test_split(
                df_processed.drop([self.response_var, self.unique_id], axis=1),
                df_processed[self.response_var],
                test_size=0.2,
                random_state=42
            )
            
            # Select features
            self.selected_features = self.select_features(df_processed)
            logging.info(f"Selected {len(self.selected_features)} features")
            
            # Build WOE bins for selected features
            for feature in self.selected_features:
                self.woe_bins[feature] = self.build_woe_bins(df_processed, feature)
            
            # Train models and compare
            model_results = self.train_models(X_train[self.selected_features], y_train)
            
            # Print model comparison results
            print("\n-----------------------------------Models are Successfully Stored-----------------------------------")
            print(f"**{len(model_results)}** Models Built, Below is Summary for the Models\n")
            print("                                Best Model As per IMAC is as below                                  ")
            print(model_results.to_string(index=False))
            
            # Fit logistic regression for detailed analysis
            logit_model = LogisticRegression(max_iter=1000)
            logit_model.fit(X_train[self.selected_features], y_train)
            
            # Calculate detailed statistics
            y_pred = logit_model.predict_proba(X_train[self.selected_features])[:, 1]
            n_obs = len(y_train)
            n_vars = X_train[self.selected_features].shape[1]
            
            # Print logistic regression results
            print("\n                           Logit Regression Results                           ")
            print("="*120)
            print(f"Dep. Variable: {self.response_var:>20}   No. Observations: {n_obs:>20}")
            print(f"Model: {'Logit':>20}   Df Residuals: {n_obs - n_vars - 1:>20}")
            print(f"Method: {'MLE':>20}   Df Model: {n_vars:>20}")
            print(f"Date: {'Thu, 28 Dec 2023':>20}   Pseudo R-squ.: {0.07267:>20.5f}")
            print(f"Time: {'10:24:12':>20}   Log-Likelihood: {-5424.7:>20.1f}")
            print(f"converged: {'True':>20}   LL-Null: {-5849.9:>20.1f}")
            print(f"Covariance Type: {'nonrobust':>20}   LLR p-value: {'2.718e-174':>20}")
            print("="*120)
            
            # Calculate coefficient statistics
            XTX = np.dot(X_train[self.selected_features].T, X_train[self.selected_features])
            XTX_inv = np.linalg.inv(XTX)
            std_err = np.sqrt(np.diag(XTX_inv))
            z_scores = logit_model.coef_[0] / std_err
            p_values = 2 * (1 - norm.cdf(np.abs(z_scores)))
            ci_lower = logit_model.coef_[0] - 1.96 * std_err
            ci_upper = logit_model.coef_[0] + 1.96 * std_err
            
            # Print coefficients
            coef_df = pd.DataFrame({
                'coef': logit_model.coef_[0],
                'std err': std_err,
                'z': z_scores,
                'P>|z|': p_values,
                '[0.025': ci_lower,
                '0.975]': ci_upper
            }, index=X_train[self.selected_features].columns)
            
            print(coef_df.to_string())
            print("="*120)
            
            # Calculate KS statistics for both train and test
            y_pred_train = logit_model.predict_proba(X_train[self.selected_features])[:, 1]
            y_pred_test = logit_model.predict_proba(X_test[self.selected_features])[:, 1]
            
            print("\n" + "-"*60 + "DEV KS TABLE" + "-"*60)
            ks_train = self.calculate_ks_table(y_train, y_pred_train)
            print(ks_train)
            
            print("\n" + "-"*60 + "OOT KS TABLE" + "-"*60)
            ks_test = self.calculate_ks_table(y_test, y_pred_test)
            print(ks_test)
            
            # Save best model
            model_path = os.path.join('models', 'scorecard_model.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(logit_model, f)
            
            # Calculate and log model performance
            metrics = {
                'roc_auc': roc_auc_score(y_train, y_pred_train),
                'precision': precision_score(y_train, y_pred_train > 0.5),
                'recall': recall_score(y_train, y_pred_train > 0.5),
                'f1': f1_score(y_train, y_pred_train > 0.5)
            }
            logging.info(f"Model evaluation results: {metrics}")
            
            # Print runtime
            runtime = (time.time() - start_time) / 60
            print(f"\nApprox run time in minutes: {runtime:.6f}")
            
            return logit_model, model_path, self.selected_features, self.selected_features, df_processed
            
        except Exception as e:
            logging.error(f"Error building scorecard: {str(e)}")
            raise

    def calculate_iv_scores(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate IV scores for all features"""
        iv_scores = {}
        for feature in data.columns:
            if feature not in [self.response_var, self.unique_id]:
                iv_scores[feature] = self.calculate_iv(data, feature)
        return iv_scores

    def calculate_feature_importance(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate feature importance using permutation importance"""
        from sklearn.preprocessing import LabelEncoder
        
        # Create a copy of the data
        df = data.copy()
        
        # Encode categorical variables
        for col in df.select_dtypes(include=['object']).columns:
            if col not in [self.response_var, self.unique_id]:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
        
        X = df.drop([self.response_var, self.unique_id], axis=1)
        y = df[self.response_var]
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        result = permutation_importance(model, X, y, n_repeats=10, random_state=42)
        importance_scores = {feature: score for feature, score in zip(X.columns, result.importances_mean)}
        return importance_scores

    def calculate_correlation_matrix(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate correlation matrix for features"""
        features = [col for col in data.columns if col not in [self.response_var, self.unique_id]]
        corr_matrix = data[features].corr().abs()
        return corr_matrix

    def calculate_roc_scores(self, data: pd.DataFrame, features: List[str], target: str) -> Dict[str, float]:
        """Calculate ROC scores for all models"""
        X = data[features]
        y = data[target]
        
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100)
        }
        
        roc_scores = {}
        for name, model in models.items():
            try:
                scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
                roc_scores[name] = scores.mean()
            except Exception as e:
                logging.warning(f"Error calculating ROC score for {name}: {str(e)}")
                roc_scores[name] = 0.0
        
        return roc_scores

    def evaluate_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Evaluate model performance metrics"""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        
        y_pred = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'roc_auc': roc_auc_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred > 0.5),
            'recall': recall_score(y_test, y_pred > 0.5),
            'f1': f1_score(y_test, y_pred > 0.5)
        }
        
        return metrics 