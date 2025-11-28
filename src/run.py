#!/usr/bin/env python3
"""
ATS Resume Matcher - Complete Pipeline Runner
Executes all steps in order: preprocessing -> dataset generation -> model training -> evaluation
"""

import sys
import subprocess
import os

def run_command(cmd, description):
    """Run a command and handle errors"""
    print("\n" + "="*70)
    print(f"🚀 {description}")
    print("="*70)
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            text=True,
            capture_output=False
        )
        print(f"{description} - COMPLETED")
        return True
    except subprocess.CalledProcessError as e:
        print(f"{description} - FAILED")
        print(f"Error: {e}")
        return False

def check_data_exists():
    """Check if required data file exists"""
    data_file = 'data/naukri_data_science_jobs_india.csv'
    if not os.path.exists(data_file):
        print("\n" + "="*70)
        print("ERROR: Data file not found!")
        print("="*70)
        print(f"Required file: {data_file}")
        print("\nPlease ensure the Naukri dataset is in the data/ directory.")
        print("="*70)
        return False
    return True

def main():
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║          ATS Resume Matcher - Complete Pipeline              ║
    ║                                                               ║
    ║  This script will run the entire ML pipeline:                ║
    ║    1. Data Preprocessing                                     ║
    ║    2. Synthetic Dataset Generation                           ║
    ║    3. TF-IDF Baseline Training                               ║
    ║    4. LSTM Model Training                                    ║
    ║    5. Model Evaluation & Comparison                          ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    # Check prerequisites
    if not check_data_exists():
        sys.exit(1)
    
    # Determine if we're in root or src directory
    if os.path.exists('preprocessing.py'):
        # We're in src/ directory
        prefix = 'python'
    else:
        # We're in root directory, need to use src/ prefix
        prefix = 'python src/'
    
    # Pipeline steps
    steps = [
        (f"{prefix}preprocessing.py", "Step 1/5: Preprocessing Raw Data"),
        (f"{prefix}build_dataset.py", "Step 2/5: Generating Training Dataset"),
        (f"{prefix}baseline_tfidf.py", "Step 3/5: Training TF-IDF Baseline"),
        (f"{prefix}train_lstm.py", "Step 4/5: Training LSTM Model"),
        (f"{prefix}evaluate.py", "Step 5/5: Evaluating & Comparing Models"),
    ]
    
    # Execute pipeline
    for cmd, description in steps:
        success = run_command(cmd, description)
        if not success:
            print("\nPipeline failed. Please fix the error and retry.")
            sys.exit(1)
    
    # Success message
    print("\n" + "="*70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nResults available in:")
    print("   - results/baseline_predictions.csv")
    print("   - results/lstm_predictions.csv")
    print("   - results/model_metrics.csv")
    print("   - results/model_comparison_plot.png")
    print("   - results/confusion_matrices.png")
    print("\nTo launch the interactive dashboard, run:")
    print("   streamlit run src/dashboard.py")
    print("="*70)

if __name__ == "__main__":
    main()
