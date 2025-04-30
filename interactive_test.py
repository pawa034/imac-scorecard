import pandas as pd
import numpy as np
from src.modeling.risk_scorecard import ScoreCardRisk
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def interactive_test():
    # Read data
    logger.info("Reading data from sample_demo.csv...")
    data = pd.read_csv('sample_demo.csv')
    print("\nData shape:", data.shape)
    print("\nFirst few rows of data:")
    print(data.head())
    
    # Sample data
    logger.info("\nSampling data...")
    sample_data = data.sample(n=4000, random_state=42)
    print("\nSampled data shape:", sample_data.shape)
    
    # Initialize ScoreCardRisk
    logger.info("\nInitializing ScoreCardRisk...")
    scorecard = ScoreCardRisk()
    
    # Build scorecard
    logger.info("\nBuilding scorecard...")
    result, path, cols_to_use, mlvarlist, processed_data = scorecard.ScoreCardBuilder(sample_data, [])
    
    # Show results
    print("\nColumns used in model:", cols_to_use)
    print("\nML variables:", mlvarlist)
    print("\nProcessed data shape:", processed_data.shape)
    print("\nModel saved at:", path)
    
    return scorecard

if __name__ == "__main__":
    print("This script is meant to be run in IPython. Please run:")
    print("ipython")
    print("Then in IPython:")
    print("from interactive_test import interactive_test")
    print("scorecard = interactive_test()") 