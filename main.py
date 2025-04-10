"""
main.py: Main script to run the complete ETH price prediction pipeline

This script orchestrates the execution of the entire pipeline:
1. Data fetching from Binance
2. Data preprocessing
3. Model training
4. Model evaluation and prediction
"""

import os
import argparse
import subprocess

def run_pipeline(skip_fetch=False, skip_preprocess=False, skip_train=False, skip_predict=False):
    """
    Run the complete ETH price prediction pipeline
    
    Args:
        skip_fetch (bool): Skip data fetching step
        skip_preprocess (bool): Skip data preprocessing step
        skip_train (bool): Skip model training step
        skip_predict (bool): Skip prediction step
    """
    # Check for .env file
    if not skip_fetch and not os.path.exists('.env'):
        print("Warning: .env file not found. Please create one with your Binance API keys.")
        print("Example:")
        print("BINANCE_API_KEY=your_api_key_here")
        print("BINANCE_API_SECRET=your_api_secret_here")
        return

    # 1. Fetch data
    if not skip_fetch:
        print("\n" + "="*50)
        print("Step 1: Fetching ETH/USDT data from Binance")
        print("="*50)
        subprocess.run(['python', 'fetch_data.py'])
    else:
        print("\nSkipping data fetching step.")

    # 2. Preprocess data
    if not skip_preprocess:
        print("\n" + "="*50)
        print("Step 2: Preprocessing ETH/USDT data")
        print("="*50)
        subprocess.run(['python', 'preprocess.py'])
    else:
        print("\nSkipping data preprocessing step.")

    # 3. Train model
    if not skip_train:
        print("\n" + "="*50)
        print("Step 3: Training LSTM model")
        print("="*50)
        subprocess.run(['python', 'train_model.py'])
    else:
        print("\nSkipping model training step.")

    # 4. Evaluate and predict
    if not skip_predict:
        print("\n" + "="*50)
        print("Step 4: Evaluating model and making predictions")
        print("="*50)
        subprocess.run(['python', 'predict.py'])
    else:
        print("\nSkipping prediction step.")

    print("\n" + "="*50)
    print("Pipeline execution complete!")
    print("="*50)

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='ETH Price Prediction Pipeline')
    parser.add_argument('--skip-fetch', action='store_true', help='Skip data fetching step')
    parser.add_argument('--skip-preprocess', action='store_true', help='Skip data preprocessing step')
    parser.add_argument('--skip-train', action='store_true', help='Skip model training step')
    parser.add_argument('--skip-predict', action='store_true', help='Skip prediction step')
    
    args = parser.parse_args()
    
    # Run pipeline
    run_pipeline(
        skip_fetch=args.skip_fetch,
        skip_preprocess=args.skip_preprocess,
        skip_train=args.skip_train,
        skip_predict=args.skip_predict
    )