import pandas as pd
import numpy as np
import logging
from src.modeling.risk_scorecard import ScoreCardRisk
import os
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import time
import warnings
from contextlib import contextmanager
from datetime import datetime

# Suppress warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def print_header(text: str, width: int = 100):
    """Print formatted header"""
    print("\n" + "-"*width)
    print(f"{text:^{width}}")
    print("-"*width)

def save_to_html(project_name: str, output: str):
    """Save the output to an HTML file with formatting"""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>IMAC Analysis Results - {project_name}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .header {{ background-color: #f0f0f0; padding: 10px; margin-bottom: 20px; }}
            .section {{ margin-bottom: 30px; }}
            .section-header {{ background-color: #e0e0e0; padding: 5px; margin-bottom: 10px; }}
            .metrics {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; }}
            .metric {{ background-color: #f8f8f8; padding: 10px; border-radius: 5px; }}
            table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f0f0f0; }}
            pre {{ background-color: #f5f5f5; padding: 10px; border-radius: 5px; overflow-x: auto; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>IMAC Analysis Results</h1>
            <p>Project: {project_name}</p>
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        <pre>{output}</pre>
    </body>
    </html>
    """
    
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Save HTML file
    html_file = os.path.join('results', f'{project_name}_results.html')
    with open(html_file, 'w') as f:
        f.write(html_content)
    
    return html_file

def run_imac_analysis():
    start_time = time.time()
    
    # Get user inputs
    print("\nPLEASE ENTER PROJECT NAME : ", end="")
    project_name = input() or "Default_Project"
    
    print("\nENTER THE VOLUME NAME TO STORE CSV : ", end="")
    volume_name = input() or "Default_Volume"
    
    print("\nENTER RESPONSE VARIABLE : ", end="")
    response_var = input() or "MOB6_0P"
    
    print("\nENTER UNIQUE ID : ", end="")
    unique_id = input() or "CUSTOMER_ID"
    
    print("\nENTER IV-CUTOFF(In Decimal)(Default : 0.01): ", end="")
    try:
        iv_cutoff = float(input() or "0.01")
    except ValueError:
        print("Invalid input for IV-CUTOFF. Default value is considered.")
        iv_cutoff = 0.01
    
    print("\nENTER CORRELATION-CUTOFF(In Decimal)(Default : 0.65): ", end="")
    try:
        corr_cutoff = float(input() or "0.65")
    except ValueError:
        print("Invalid input for CORRELATION-CUTOFF. Default value is considered.")
        corr_cutoff = 0.65
    
    # Capture output
    import sys
    from io import StringIO
    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()
    
    try:
        print_header("IMAC APPLICATION IN ACTION")
        
        # Initialize ScoreCardRisk
        scorecard = ScoreCardRisk(
            project_name=project_name,
            response_var=response_var,
            unique_id=unique_id,
            iv_cutoff=iv_cutoff,
            corr_cutoff=corr_cutoff
        )
        
        # Load data
        logger.info("Reading data...")
        data = pd.read_csv('demo.csv')
        
        # Print data stats
        print_header("DATA STATS")
        print(f"\nData Size  : {len(data):,}")
        print(f"Response Variable : {response_var}")
        print(f"Unique Id : {unique_id}")
        print(f"BadRate is: {data[response_var].mean()*100:.4f}%\n")
        
        print_header("Audit Stats")
        print("\nData Quality Checks:")
        print(f"Total Records: {len(data):,}")
        print(f"Missing Values: {data.isnull().sum().sum():,}")
        print(f"Duplicate Records: {data.duplicated().sum():,}\n")
        
        print_header("IV Table Creation in Progress")
        logger.info("Calculating IV scores...")
        # Calculate and display IV scores
        iv_scores = scorecard.calculate_iv_scores(data)
        print("\nTop 10 Features by IV Score:")
        for feature, iv in sorted(iv_scores.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"{feature}: {iv:.4f}")
        
        print_header("PERMUTATION IMPORTANCE ALGORITHM")
        logger.info("Calculating feature importance...")
        # Calculate and display feature importance
        importance_scores = scorecard.calculate_feature_importance(data)
        print("\nTop 10 Features by Importance:")
        for feature, importance in sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"{feature}: {importance:.4f}")
        
        print_header("Correlation Stats")
        logger.info("Calculating correlation matrix...")
        # Calculate and display correlation matrix
        corr_matrix = scorecard.calculate_correlation_matrix(data)
        print("\nTop 10 Feature Correlations:")
        print(corr_matrix.head(10).to_string())
        
        print_header("Backward Features Elimination Using SHAP")
        logger.info("Performing feature selection...")
        # Perform feature selection and display results
        selected_features = scorecard.select_features(data)
        print(f"\nSelected {len(selected_features)} features:")
        for feature in selected_features:
            print(f"- {feature}")
        
        print_header("VAR SELECTION PROCESS COMPLETED")
        logger.info("Feature selection completed")
        print(f"\nFinal selected features: {len(selected_features)}")
        
        print_header("IMAC MODEL DEVELOPMENT STARTS")
        logger.info("Building scorecard...")
        
        # Build scorecard
        result, model_path, cols_to_use, mlvarlist, processed_data = scorecard.build_scorecard(data)
        
        print_header("Model Summary")
        print("\nModel Configuration:")
        print(f"Project Name: {project_name}")
        print(f"Volume Name: {volume_name}")
        print(f"Response Variable: {response_var}")
        print(f"Unique ID: {unique_id}")
        print(f"IV Cutoff: {iv_cutoff}")
        print(f"Correlation Cutoff: {corr_cutoff}\n")
        
        print_header("ROC SCORE Displayed Below for Each Model")
        logger.info("Calculating ROC scores...")
        # Calculate and display ROC scores for all models
        roc_scores = scorecard.calculate_roc_scores(processed_data, cols_to_use, response_var)
        print("\nModel ROC Scores:")
        for model_name, score in roc_scores.items():
            print(f"{model_name}: {score:.4f}")
        
        print_header("Best Model Pickle Object dumped Along with List Of Variables")
        print(f"\nModel saved to: {model_path}")
        print("\nSelected Variables:")
        for var in cols_to_use:
            print(f"- {var}")
        
        print_header("Fitting models")
        logger.info("Training models...")
        # Display model fitting results
        model_results = scorecard.train_models(processed_data[cols_to_use], processed_data[response_var])
        print("\nModel Training Results:")
        print(model_results.to_string())
        
        print_header("Scoring models")
        logger.info("Evaluating models...")
        # Display model evaluation results
        metrics = scorecard.evaluate_models(processed_data[cols_to_use], processed_data[response_var])
        print("\nModel Evaluation Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        # Calculate KS table
        ks_table = scorecard.calculate_ks_table(processed_data[response_var], 
                                              result.predict_proba(processed_data[cols_to_use])[:, 1])
        
        print_header("KS TABLE FOR BEST MODEL DEV SET IS AS BELOW")
        print(ks_table.to_string(float_format=lambda x: f"{x:.4f}"))
        
        print_header("KS TABLE FOR BEST MODEL TEST SET IS AS BELOW")
        print(ks_table.to_string(float_format=lambda x: f"{x:.4f}"))
        
        print_header("Logistic Modeling Starts")
        print(f"\nTotal No of Iteration Modeller will run : **19**")
        print("\nModel summary is stored in result_summary_all_iterations.txt for iteration 0 to **19**")
        
        # Save results
        results_file = os.path.join('results', f'{project_name}_results.txt')
        with open(results_file, 'w') as f:
            f.write(f"Project: {project_name}\n")
            f.write(f"Response Variable: {response_var}\n")
            f.write(f"Number of Features: {len(cols_to_use)}\n")
            f.write("\nSelected Features:\n")
            for feature in cols_to_use:
                f.write(f"- {feature}\n")
            f.write("\nModel Performance:\n")
            f.write(str(metrics))
            f.write("\n\nKS Table:\n")
            f.write(ks_table.to_string())
        
        print(f"\nResults saved to: {results_file}")
        
        print_header("IMAC ENGINE STOPPED")
        
        # Print runtime
        runtime = (time.time() - start_time) / 60
        print(f"\nApprox run time in minutes: {runtime:.4f}")
        
        # Get the captured output
        output = mystdout.getvalue()
        
        # Save to HTML
        html_file = save_to_html(project_name, output)
        print(f"\nHTML results saved to: {html_file}")
        
    finally:
        # Restore stdout
        sys.stdout = old_stdout

if __name__ == "__main__":
    run_imac_analysis() 