"""
preprocess.py: Preprocess ETH/USDT data for LSTM model

This script loads the historical ETH/USDT data, applies MinMax scaling to the close prices,
and creates sliding sequences for the LSTM model.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def load_data(filepath='data/eth_data.csv'):
    """Load ETH/USDT data from CSV file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}. Run fetch_data.py first.")
    
    data = pd.read_csv(filepath)
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    return data

def create_sequences(data, seq_length=60):
    """
    Create sliding window sequences from data
    
    Args:
        data (np.array): Array of scaled close prices
        seq_length (int): Length of input sequence
        
    Returns:
        X (np.array): Input sequences of shape (n_samples, seq_length, 1)
        y (np.array): Target values of shape (n_samples, 1)
    """
    X, y = [], []
    
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    
    return np.array(X), np.array(y)

def preprocess_data(seq_length=60, test_size=0.2, random_state=42):
    """
    Preprocess data for LSTM model
    
    Args:
        seq_length (int): Length of input sequence
        test_size (float): Proportion of data to use for testing
        random_state (int): Random state for reproducibility
        
    Returns:
        dict: Dictionary containing preprocessed data and metadata
    """
    # Load data
    print("Loading data...")
    data = load_data()
    print(f"Loaded {len(data)} records")
    
    # Extract close prices
    close_prices = data['close'].values.reshape(-1, 1)
    
    # Scale data
    print("Scaling data...")
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(close_prices)
    
    # Create sequences
    print(f"Creating sequences with window size {seq_length}...")
    X, y = create_sequences(scaled_prices, seq_length)
    
    # Reshape for LSTM [samples, time steps, features]
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    # Split into train and test sets
    print(f"Splitting data with test_size={test_size}...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )
    
    # Create output directory if it doesn't exist
    os.makedirs('data/processed', exist_ok=True)
    
    # Save metadata (dates, original prices, and scaler) for later
    dates = data['timestamp'].values[seq_length:]  # Align with sequences
    
    # Save the processed data and metadata
    np.save('data/processed/X_train.npy', X_train)
    np.save('data/processed/X_test.npy', X_test)
    np.save('data/processed/y_train.npy', y_train)
    np.save('data/processed/y_test.npy', y_test)
    np.save('data/processed/dates.npy', dates)
    np.save('data/processed/original_prices.npy', close_prices)
    
    # Create metadata for model building and evaluation
    metadata = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'dates': dates,
        'original_prices': close_prices,
        'scaler': scaler,
        'seq_length': seq_length,
        'train_size': len(X_train),
        'test_size': len(X_test)
    }
    
    print(f"Preprocessing complete. Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    return metadata

if __name__ == "__main__":
    metadata = preprocess_data()
    
    print("\nPreprocessing Statistics:")
    print(f"Input shape (X_train): {metadata['X_train'].shape}")
    print(f"Output shape (y_train): {metadata['y_train'].shape}")
    print(f"Date range: {metadata['dates'][0]} to {metadata['dates'][-1]}")
    print(f"Files saved to data/processed/")