"""
predict.py: Evaluate LSTM model and make predictions for ETH prices

This script loads the trained LSTM model and makes predictions on the test set.
It also plots the predicted vs actual prices and evaluates the model performance.
"""

import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def load_test_data():
    """Load test data and related metadata."""
    try:
        X_test = np.load('data/processed/X_test.npy')
        y_test = np.load('data/processed/y_test.npy')
        dates = np.load('data/processed/dates.npy', allow_pickle=True)
        original_prices = np.load('data/processed/original_prices.npy')
        
        # Determine test dates (last 20% of the data)
        test_size = len(X_test)
        test_dates = dates[-test_size:]
        
        return X_test, y_test, test_dates, original_prices
    
    except FileNotFoundError:
        raise FileNotFoundError(
            "Test data not found. Run preprocess.py first."
        )

def create_scaler_from_data():
    """Recreate the scaler from the original data."""
    original_prices = np.load('data/processed/original_prices.npy')
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(original_prices)
    return scaler

def evaluate_model(model, X_test, y_test, scaler):
    """
    Evaluate model performance on test data
    
    Args:
        model: Trained Keras model
        X_test: Test input data
        y_test: Test target data
        scaler: Fitted MinMaxScaler
        
    Returns:
        dict: Dictionary containing predictions and evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Reshape data for inverse transform
    y_test_reshaped = y_test.reshape(-1, 1)
    y_pred_reshaped = y_pred.reshape(-1, 1)
    
    # Inverse transform to get actual prices
    y_test_inv = scaler.inverse_transform(y_test_reshaped)
    y_pred_inv = scaler.inverse_transform(y_pred_reshaped)
    
    # Calculate metrics
    mse = mean_squared_error(y_test_inv, y_pred_inv)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    r2 = r2_score(y_test_inv, y_pred_inv)
    
    # Calculate MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_test_inv - y_pred_inv) / y_test_inv)) * 100
    
    # Return results
    return {
        'y_test': y_test_inv,
        'y_pred': y_pred_inv,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape
    }

def plot_predictions(dates, actual, predicted):
    """
    Plot actual vs predicted prices
    
    Args:
        dates: Array of datetime objects
        actual: Array of actual prices
        predicted: Array of predicted prices
    """
    plt.figure(figsize=(12, 6))
    plt.plot(dates, actual, label='Actual ETH Price', color='blue')
    plt.plot(dates, predicted, label='Predicted ETH Price', color='red')
    plt.title('ETH/USDT Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price (USDT)')
    plt.legend()
    plt.grid(True)
    
    # Format dates on x-axis
    plt.gcf().autofmt_xdate()
    
    # Create figures directory if it doesn't exist
    os.makedirs('figures', exist_ok=True)
    plt.savefig('figures/eth_price_prediction.png')
    plt.close()

def save_predictions(dates, actual, predicted):
    """
    Save predictions to CSV file
    
    Args:
        dates: Array of datetime objects
        actual: Array of actual prices
        predicted: Array of predicted prices
    """
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Create DataFrame with results
    results_df = pd.DataFrame({
        'Date': dates,
        'Actual_Price': actual.flatten(),
        'Predicted_Price': predicted.flatten()
    })
    
    # Save to CSV
    results_df.to_csv('results/eth_predictions.csv', index=False)
    print("Predictions saved to results/eth_predictions.csv")

if __name__ == "__main__":
    # Load model
    try:
        print("Loading trained model...")
        model = load_model('models/eth_lstm_model.h5')
    except:
        raise FileNotFoundError(
            "Trained model not found. Run train_model.py first."
        )
    
    # Load test data
    print("Loading test data...")
    X_test, y_test, test_dates, original_prices = load_test_data()
    
    # Create scaler from original data
    scaler = create_scaler_from_data()
    
    # Evaluate model
    print("Evaluating model on test data...")
    results = evaluate_model(model, X_test, y_test, scaler)
    
    # Print metrics
    print("\nModel Evaluation Metrics:")
    print(f"Mean Squared Error (MSE): {results['mse']:.2f}")
    print(f"Root Mean Squared Error (RMSE): {results['rmse']:.2f}")
    print(f"Mean Absolute Error (MAE): {results['mae']:.2f}")
    print(f"R² Score: {results['r2']:.4f}")
    print(f"Mean Absolute Percentage Error (MAPE): {results['mape']:.2f}%")
    
    # Plot and save predictions
    print("\nPlotting predictions...")
    plot_predictions(test_dates, results['y_test'], results['y_pred'])
    print("Prediction plot saved to figures/eth_price_prediction.png")
    
    # Save predictions to CSV
    save_predictions(test_dates, results['y_test'], results['y_pred'])