"""
train_model.py: Build and train an LSTM model for ETH price prediction

This script builds a deep learning model with LSTM layers to predict ETH prices
based on the preprocessed data. It trains the model and saves it for later use.
"""

import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler

def load_preprocessed_data():
    """Load preprocessed data for model training."""
    try:
        X_train = np.load('data/processed/X_train.npy')
        X_test = np.load('data/processed/X_test.npy')
        y_train = np.load('data/processed/y_train.npy')
        y_test = np.load('data/processed/y_test.npy')
        
        return X_train, X_test, y_train, y_test
    
    except FileNotFoundError:
        raise FileNotFoundError(
            "Preprocessed data not found. Run preprocess.py first."
        )

def build_model(input_shape):
    """
    Build LSTM model for time series prediction
    
    Args:
        input_shape (tuple): Shape of input data (timesteps, features)
        
    Returns:
        model: Compiled Keras model
    """
    model = Sequential()
    
    # First LSTM layer with return sequences for stacking
    model.add(LSTM(
        units=50,
        return_sequences=True,
        input_shape=input_shape
    ))
    model.add(Dropout(0.2))
    
    # Second LSTM layer
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    
    # Output layer
    model.add(Dense(units=1))
    
    # Compile model
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model

def train_model(model, X_train, y_train, X_test, y_test, epochs=50, batch_size=32):
    """
    Train the LSTM model
    
    Args:
        model: Compiled Keras model
        X_train, y_train: Training data
        X_test, y_test: Test data for validation
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        
    Returns:
        history: Training history
    """
    # Create model directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Callbacks for training
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        ModelCheckpoint(
            filepath='models/eth_lstm_model.h5',
            monitor='val_loss',
            save_best_only=True
        )
    ]
    
    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    
    return history

def plot_loss(history):
    """Plot training and validation loss."""
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Create figures directory if it doesn't exist
    os.makedirs('figures', exist_ok=True)
    plt.savefig('figures/model_loss.png')
    plt.close()

def save_training_metadata(history):
    """Save training history and metadata."""
    with open('models/training_history.pkl', 'wb') as f:
        pickle.dump(history.history, f)

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Check for GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"Training with GPU: {gpus}")
    else:
        print("No GPU found. Training with CPU.")
    
    # Load preprocessed data
    print("Loading preprocessed data...")
    X_train, X_test, y_train, y_test = load_preprocessed_data()
    print(f"Data loaded: X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    
    # Build model
    print("Building LSTM model...")
    input_shape = (X_train.shape[1], 1)  # (timesteps, features)
    model = build_model(input_shape)
    model.summary()
    
    # Train model
    print("Training model...")
    history = train_model(model, X_train, y_train, X_test, y_test)
    
    # Plot and save training results
    plot_loss(history)
    save_training_metadata(history)
    
    print("Model training complete. Model saved to models/eth_lstm_model.h5")