# Ethereum Price Prediction

A machine learning pipeline for predicting Ethereum (ETH/USDT) prices using historical data from Binance and LSTM neural networks.

## Overview

This project implements a complete pipeline for predicting cryptocurrency prices with the following components:

1. **Data Fetching**: Securely access the Binance API to fetch historical ETH/USDT candlestick data
2. **Data Preprocessing**: Scale the data and create time sequences for the LSTM model
3. **Model Training**: Build and train a deep learning model with LSTM layers
4. **Prediction**: Evaluate the model and make price predictions

## Requirements

- Python 3.8+
- macOS (optimized for Apple Silicon M3 Pro)
- Binance API credentials

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/crypto-price-prediction.git
cd crypto-price-prediction
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root with your Binance API keys:
```
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here
```

## Usage

### Run the Full Pipeline

To run the complete pipeline from data fetching to prediction:

```bash
python main.py
```

### Run Individual Components

You can also run each component separately:

1. Fetch historical data:
```bash
python fetch_data.py
```

2. Preprocess the data:
```bash
python preprocess.py
```

3. Train the LSTM model:
```bash
python train_model.py
```

4. Make predictions using the trained model:
```bash
python predict.py
```

### Skip Steps in the Pipeline

You can skip specific steps in the pipeline using command-line arguments:

```bash
# Skip data fetching (use existing data)
python main.py --skip-fetch

# Skip data preprocessing (use existing processed data) 
python main.py --skip-fetch --skip-preprocess

# Only run the prediction step
python main.py --skip-fetch --skip-preprocess --skip-train
```

## Project Structure

- `fetch_data.py`: Downloads historical ETH/USDT data from Binance
- `preprocess.py`: Scales data and creates sequences for the LSTM model
- `train_model.py`: Builds and trains the LSTM model
- `predict.py`: Evaluates the model and generates predictions
- `main.py`: Orchestrates the entire pipeline
- `data/`: Directory for storing raw and processed data
- `models/`: Directory for storing trained models
- `figures/`: Directory for storing visualizations
- `results/`: Directory for storing prediction results

## Model Architecture

The LSTM model architecture consists of:
- Input layer for 60-day price sequences
- Two LSTM layers with dropout
- Dense output layer for next-day price prediction
- Optimized with Adam optimizer and MSE loss function

## License

MIT License

