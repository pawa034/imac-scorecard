import pandas as pd
import numpy as np
from scorecard import ScoreCardRisk

def main():
    try:
        # Step 1: Read data from sample_demo.csv
        print("Step 1: Reading data from sample_demo.csv...")
        df = pd.read_csv("sample_demo.csv")
        
        # Step 2: Sample the data
        print("\nStep 2: Sampling the data...")
        df = df.sample(frac=0.4, random_state=42)
        
        # Step 3: Initialize ScoreCardRisk
        print("\nStep 3: Initializing ScoreCardRisk...")
        sc = ScoreCardRisk()
        
        # Step 4: Build scorecard
        print("\nStep 4: Building scorecard...")
        colToRemove = []
        result, path, cols_to_use, mlvarlist, data = sc.ScoreCardBuilder(df, colToRemove)
        
        # Step 5: Print results
        print("\nStep 5: Results:")
        print(f"Model path: {path}")
        print(f"Columns used: {cols_to_use}")
        print(f"ML variables: {mlvarlist}")
        print(f"Processed data shape: {data.shape}")
        print("\nFirst few rows of processed data:")
        print(data.head())
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    main() 