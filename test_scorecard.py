import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from scorecard import ScoreCardRisk

def convert_decimal_to_double(df):
    """Convert decimal columns to double type"""
    for col in df.columns:
        if df.schema[col].dataType.typeName() == 'decimal':
            df = df.withColumn(col, df[col].cast('double'))
    return df

def main():
    # Step 1: Initialize Spark session
    print("Step 1: Initializing Spark session...")
    spark = SparkSession.builder \
        .appName("ScoreCardTest") \
        .getOrCreate()
    
    try:
        # Step 2: Read data from sample_demo.csv
        print("\nStep 2: Reading data from sample_demo.csv...")
        df = spark.read.csv("sample_demo.csv", header=True, inferSchema=True)
        
        # Step 3: Sample the data
        print("\nStep 3: Sampling the data...")
        df = df.sample(fraction=0.4)
        
        # Step 4: Convert decimal to double
        print("\nStep 4: Converting decimal columns to double...")
        df = convert_decimal_to_double(df)
        
        # Step 5: Convert to pandas
        print("\nStep 5: Converting to pandas DataFrame...")
        pdf = df.toPandas()
        
        # Step 6: Initialize ScoreCardRisk
        print("\nStep 6: Initializing ScoreCardRisk...")
        sc = ScoreCardRisk()
        
        # Step 7: Build scorecard
        print("\nStep 7: Building scorecard...")
        colToRemove = []
        result, path, cols_to_use, mlvarlist, data = sc.ScoreCardBuilder(pdf, colToRemove)
        
        # Step 8: Print results
        print("\nStep 8: Results:")
        print(f"Model path: {path}")
        print(f"Columns used: {cols_to_use}")
        print(f"ML variables: {mlvarlist}")
        print(f"Processed data shape: {data.shape}")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
    finally:
        # Step 9: Stop Spark session
        print("\nStep 9: Stopping Spark session...")
        spark.stop()

if __name__ == "__main__":
    main() 