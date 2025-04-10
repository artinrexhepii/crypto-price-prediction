"""
analyze_accuracy.py: Analyze the model's prediction accuracy on historical data

This script evaluates how well the model predicted past prices by analyzing
the test set results and visualizing prediction errors.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

def analyze_prediction_accuracy():
    """Analyze the model's prediction accuracy on historical test data."""
    print("Loading trained model...")
    try:
        model = load_model('models/eth_lstm_model.h5')
    except:
        raise FileNotFoundError(
            "Trained model not found. Run train_model.py first."
        )
    
    print("Loading test data...")
    X_test, y_test, test_dates, original_prices = load_test_data()
    
    print(f"Loaded {len(X_test)} test samples")
    
    # Create scaler from original data
    scaler = create_scaler_from_data()
    
    # Make predictions on test data
    print("Making predictions on test data...")
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
    
    # Print overall metrics
    print("\n==== Overall Model Performance on Test Data ====")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"R² Score: {r2:.4f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    
    # Create a DataFrame with actual vs predicted prices
    results_df = pd.DataFrame({
        'Date': test_dates,
        'Actual_Price': y_test_inv.flatten(),
        'Predicted_Price': y_pred_inv.flatten()
    })
    
    # Calculate daily percentage error
    results_df['Percentage_Error'] = np.abs((results_df['Actual_Price'] - results_df['Predicted_Price']) / results_df['Actual_Price'] * 100)
    
    # Save detailed results to CSV
    os.makedirs('results', exist_ok=True)
    results_df.to_csv('results/accuracy_analysis.csv', index=False)
    print(f"Detailed results saved to results/accuracy_analysis.csv")
    
    # Summary statistics for errors
    print("\n==== Error Distribution ====")
    error_stats = results_df['Percentage_Error'].describe()
    print(f"Min Error: {error_stats['min']:.2f}%")
    print(f"Max Error: {error_stats['max']:.2f}%")
    print(f"Mean Error: {error_stats['mean']:.2f}%")
    print(f"Median Error: {error_stats['50%']:.2f}%")
    print(f"Standard Deviation: {error_stats['std']:.2f}%")
    
    # Analyze accuracy over time (monthly)
    results_df['Month'] = pd.to_datetime(results_df['Date']).dt.to_period('M')
    monthly_errors = results_df.groupby('Month')['Percentage_Error'].mean()
    
    print("\n==== Monthly Average Error ====")
    for month, error in monthly_errors.items():
        print(f"{month}: {error:.2f}%")
    
    # Plot actual vs predicted prices
    plt.figure(figsize=(14, 7))
    plt.plot(results_df['Date'], results_df['Actual_Price'], label='Actual Price', color='blue')
    plt.plot(results_df['Date'], results_df['Predicted_Price'], label='Predicted Price', color='red', linestyle='--')
    plt.title('ETH/USDT Actual vs Predicted Prices (Test Data)')
    plt.xlabel('Date')
    plt.ylabel('Price (USDT)')
    plt.legend()
    plt.grid(True)
    plt.gcf().autofmt_xdate()
    plt.savefig('figures/historical_accuracy.png')
    print("Historical accuracy plot saved to figures/historical_accuracy.png")
    
    # Plot error distribution
    plt.figure(figsize=(14, 7))
    plt.hist(results_df['Percentage_Error'], bins=30, alpha=0.7, color='blue')
    plt.axvline(results_df['Percentage_Error'].mean(), color='red', linestyle='--', label=f'Mean Error: {results_df["Percentage_Error"].mean():.2f}%')
    plt.title('Distribution of Prediction Errors')
    plt.xlabel('Percentage Error (%)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.legend()
    plt.savefig('figures/error_distribution.png')
    print("Error distribution plot saved to figures/error_distribution.png")
    
    # Plot monthly errors
    plt.figure(figsize=(14, 7))
    monthly_errors.plot(kind='bar', color='skyblue')
    plt.title('Average Monthly Prediction Error')
    plt.xlabel('Month')
    plt.ylabel('Percentage Error (%)')
    plt.axhline(y=monthly_errors.mean(), color='red', linestyle='--', label=f'Overall Mean: {monthly_errors.mean():.2f}%')
    plt.legend()
    plt.grid(True, axis='y')
    plt.savefig('figures/monthly_errors.png')
    print("Monthly errors plot saved to figures/monthly_errors.png")
    
    # Plot prediction errors over time
    plt.figure(figsize=(14, 7))
    plt.plot(results_df['Date'], results_df['Percentage_Error'], color='purple')
    plt.title('Prediction Errors Over Time')
    plt.xlabel('Date')
    plt.ylabel('Percentage Error (%)')
    plt.grid(True)
    plt.gcf().autofmt_xdate()
    plt.savefig('figures/errors_over_time.png')
    print("Errors over time plot saved to figures/errors_over_time.png")
    
    return results_df, mape

if __name__ == "__main__":
    print("=" * 50)
    print("Analyzing Model Prediction Accuracy on Historical Data")
    print("=" * 50)
    results_df, overall_mape = analyze_prediction_accuracy()