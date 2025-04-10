"""
check_overfitting.py: Check model performance and overfitting

This script analyzes the training history to check for overfitting
and also sets up the model for future predictions.
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

def analyze_training_history():
    """Analyze the training history to check for overfitting"""
    # Load training history
    try:
        with open('models/training_history.pkl', 'rb') as f:
            history = pickle.load(f)
    except FileNotFoundError:
        print("Training history not found. Run train_model.py first.")
        return None
    
    # Print final losses
    print("\nTraining History Analysis:")
    print(f"Final training loss: {history['loss'][-1]:.6f}")
    print(f"Final validation loss: {history['val_loss'][-1]:.6f}")
    
    # Calculate loss ratio (val/train)
    loss_ratio = history['val_loss'][-1] / history['loss'][-1]
    print(f"Loss ratio (val/train): {loss_ratio:.2f}")
    
    # Interpret results
    if loss_ratio > 1.5:
        print("⚠️ Warning: The model might be overfitting (validation loss significantly higher than training loss).")
    elif loss_ratio < 0.8:
        print("⚠️ Warning: Unusual pattern - validation loss is much lower than training loss.")
    else:
        print("✅ The model shows a good balance between training and validation loss.")
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    
    # Save figure
    os.makedirs('figures', exist_ok=True)
    plt.savefig('figures/overfitting_analysis.png')
    print("Training history plot saved to figures/overfitting_analysis.png")
    
    return history

def predict_future_prices(days=30):
    """
    Generate predictions for future ETH prices
    
    Args:
        days (int): Number of days to predict into the future
    """
    # Load the model
    try:
        model = load_model('models/eth_lstm_model.h5')
        print("\nLoaded trained model successfully.")
    except:
        print("Failed to load model. Run train_model.py first.")
        return
    
    # Load the last sequence from test data
    try:
        X_test = np.load('data/processed/X_test.npy')
        original_prices = np.load('data/processed/original_prices.npy')
        dates = np.load('data/processed/dates.npy', allow_pickle=True)
        print(f"Loaded test data with {len(X_test)} samples.")
    except:
        print("Failed to load test data. Run preprocess.py first.")
        return
    
    # Get the last sequence for prediction
    last_sequence = X_test[-1:] # Shape: (1, 60, 1)
    print(f"Using the last sequence from test data with shape: {last_sequence.shape}")
    
    # Create scaler from original data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(original_prices)
    
    # Generate future dates
    last_date = pd.to_datetime(dates[-1])
    future_dates = [last_date + pd.Timedelta(days=i+1) for i in range(days)]
    
    # Make recursive predictions
    predicted_scaled = []
    curr_sequence = last_sequence[0].copy()
    
    print("\nPredicting future prices:")
    for i in range(days):
        # Reshape for prediction
        curr_sequence_reshaped = curr_sequence.reshape(1, 60, 1)
        
        # Predict next value
        next_pred = model.predict(curr_sequence_reshaped, verbose=0)[0][0]
        predicted_scaled.append(next_pred)
        
        # Update the sequence by shifting and adding the new prediction
        curr_sequence = np.append(curr_sequence[1:], next_pred)
    
    # Convert scaled predictions back to original scale
    predicted_prices = scaler.inverse_transform(
        np.array(predicted_scaled).reshape(-1, 1)
    )
    
    # Create a DataFrame for the results
    future_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted_Price': predicted_prices.flatten()
    })
    
    # Save predictions to CSV
    os.makedirs('results', exist_ok=True)
    future_df.to_csv('results/future_eth_predictions.csv', index=False)
    print(f"Future predictions saved to results/future_eth_predictions.csv")
    
    # Get the last actual price
    last_actual_price = scaler.inverse_transform(
        np.array(last_sequence[-1]).reshape(-1, 1)
    )[0][0]
    
    # Display future predictions
    print("\nETH Price Predictions:")
    print(f"Last known price (on {last_date.strftime('%Y-%m-%d')}): ${last_actual_price:.2f}")
    print("\nFuture price predictions:")
    for i in range(min(10, days)):  # Show first 10 days
        pred_price = predicted_prices[i][0]
        date_str = future_dates[i].strftime('%Y-%m-%d')
        change = ((pred_price / last_actual_price) - 1) * 100
        print(f"{date_str}: ${pred_price:.2f} ({change:+.2f}%)")
    
    if days > 10:
        print("...")
        for i in range(days-5, days):  # Show last 5 days
            pred_price = predicted_prices[i][0]
            date_str = future_dates[i].strftime('%Y-%m-%d')
            change = ((pred_price / last_actual_price) - 1) * 100
            print(f"{date_str}: ${pred_price:.2f} ({change:+.2f}%)")
    
    # Plot predictions
    plt.figure(figsize=(12, 6))
    
    # Get the last 30 actual prices for context
    last_30_dates = dates[-30:]
    last_30_actual = original_prices[-30:].flatten()
    
    # Plot last 30 days of actual prices
    plt.plot(last_30_dates, last_30_actual, label='Historical Prices', color='blue')
    
    # Plot future predictions
    plt.plot(future_dates, predicted_prices, label='Predicted Prices', color='red', linestyle='--')
    
    # Add labels and title
    plt.title('ETH/USDT Future Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price (USDT)')
    plt.legend()
    plt.grid(True)
    
    # Format dates on x-axis
    plt.gcf().autofmt_xdate()
    
    # Save the plot
    plt.savefig('figures/future_predictions.png')
    print("Future prediction plot saved to figures/future_predictions.png")

if __name__ == "__main__":
    print("=" * 50)
    print("Checking for Model Overfitting")
    print("=" * 50)
    history = analyze_training_history()
    
    print("\n" + "=" * 50)
    print("Generating Future ETH Price Predictions")
    print("=" * 50)
    predict_future_prices(days=30)  # Predict next 30 days