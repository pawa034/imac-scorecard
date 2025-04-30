import pandas as pd
import numpy as np
from src.modeling.risk_scorecard import ScoreCardRisk
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    try:
        # Read data
        logger.info("Reading data from sample_demo.csv...")
        df = pd.read_csv("sample_demo.csv")
        
        # Sample the data
        logger.info("Sampling data...")
        df = df.sample(frac=0.4, random_state=42)
        
        # Initialize ScoreCardRisk
        logger.info("Initializing ScoreCardRisk...")
        sc = ScoreCardRisk()
        
        # Define columns to remove
        colToRemove = []
        
        # Build scorecard
        logger.info("Building scorecard...")
        result, path, cols_to_use, mlvarlist, data = sc.ScoreCardBuilder(df, colToRemove)
        
        # Print results
        logger.info("\nResults:")
        logger.info(f"Model saved at: {path}")
        logger.info(f"\nNumber of features used: {len(cols_to_use)}")
        logger.info("\nTop 10 selected features:")
        for i, col in enumerate(cols_to_use[:10], 1):
            logger.info(f"{i}. {col}")
            
        logger.info(f"\nProcessed data shape: {data.shape}")
        
        # Get feature importance
        if hasattr(result, 'iv_dict'):
            logger.info("\nTop 10 Information Values (IV):")
            sorted_iv = sorted(result.iv_dict.items(), key=lambda x: x[1], reverse=True)[:10]
            for feature, iv in sorted_iv:
                logger.info(f"{feature}: {iv:.4f}")
        
        # Print WOE details for a sample feature
        if hasattr(result, 'woe_dict') and cols_to_use:
            sample_feature = cols_to_use[0]
            logger.info(f"\nWOE bins for feature '{sample_feature}':")
            woe_values = result.woe_dict.get(sample_feature, {})
            for bin_val, woe in woe_values.items():
                logger.info(f"Bin {bin_val}: WOE = {woe:.4f}")
        
        logger.info("\nModel evaluation results will be in the logs above")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main() 