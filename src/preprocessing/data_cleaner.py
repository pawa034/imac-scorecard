import pandas as pd
import numpy as np
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class DataCleaner:
    def __init__(self):
        self.numeric_columns: List[str] = []
        self.categorical_columns: List[str] = []
        
    def fit(self, df: pd.DataFrame) -> 'DataCleaner':
        """
        Fit the cleaner by identifying column types
        """
        self.numeric_columns = list(df.select_dtypes(include=[np.number]).columns)
        self.categorical_columns = list(df.select_dtypes(include=['object']).columns)
        return self
        
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess the data
        """
        df = df.copy()
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Convert categorical variables
        df = self._encode_categorical_variables(df)
        
        # Remove outliers
        df = self._handle_outliers(df)
        
        return df
        
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in numeric and categorical columns"""
        # For numeric columns, fill with mean
        for col in self.numeric_columns:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].mean())
                
        # For categorical columns, fill with mode
        for col in self.categorical_columns:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'MISSING')
                
        return df
        
    def _encode_categorical_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert categorical variables to numerical"""
        for col in self.categorical_columns:
            if col in df.columns:
                df[col] = pd.factorize(df[col])[0]
        return df
        
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers using IQR method"""
        for col in self.numeric_columns:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df[col] = df[col].clip(lower_bound, upper_bound)
        return df 