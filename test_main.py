import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from src.modeling.risk_scorecard import ScoreCardRisk
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def convert_decimal_to_double(df):
    """Convert decimal columns to double type"""
    for col in df.columns:
        if df.schema[col].dataType.typeName() == 'decimal':
            df = df.withColumn(col, df[col].cast('double'))
    return df

def main():
    try:
        # Initialize Spark session
        logger.info("Initializing Spark session...")
        spark = SparkSession.builder \
            .appName("ScoreCardTest") \
            .config("spark.driver.memory", "4g") \
            .getOrCreate()

        # Read data using your original query
        logger.info("Reading data...")
        try:
            # Try the original BFL query first
            df = spark.sql('select * from BFL_STD_LAKE.RISK_ANALYTICS.IMAC_DATA_B2B_NTB')
        except Exception as e:
            logger.warning(f"Could not read from BFL database: {str(e)}")
            logger.info("Using sample_demo.csv instead...")
            # Fallback to sample_demo.csv
            df = spark.read.csv("sample_demo.csv", header=True, inferSchema=True)

        # Sample the data
        logger.info("Sampling data...")
        df = df.sample(fraction=0.4)

        # Convert decimal to double
        logger.info("Converting decimal columns to double...")
        df = convert_decimal_to_double(df)

        # Convert to pandas
        logger.info("Converting to pandas DataFrame...")
        pdf = df.toPandas()

        # Initialize ScoreCardRisk
        logger.info("Initializing ScoreCardRisk...")
        sc = ScoreCardRisk()

        # Define columns to remove
        colToRemove = []

        # Build scorecard
        logger.info("Building scorecard...")
        result, path, cols_to_use, mlvarlist, data = sc.ScoreCardBuilder(pdf, colToRemove)

        # Print results
        logger.info("\nResults:")
        logger.info(f"Model saved at: {path}")
        logger.info(f"\nNumber of features used: {len(cols_to_use)}")
        logger.info("\nTop 10 selected features:")
        for i, col in enumerate(cols_to_use[:10], 1):
            logger.info(f"{i}. {col}")

        logger.info(f"\nProcessed data shape: {data.shape}")

        # Get feature importance
        feature_importance = result.feature_scores if hasattr(result, 'feature_scores') else {}
        if feature_importance:
            logger.info("\nTop 10 feature importance scores:")
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
            for feature, score in sorted_features:
                logger.info(f"{feature}: {score:.4f}")

        # Stop Spark session
        logger.info("\nStopping Spark session...")
        spark.stop()

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main() 