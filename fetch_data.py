"""
fetch_data.py: Fetches historical ETH/USDT data from Binance API

This script securely loads Binance API keys from a .env file and
fetches historical daily candlestick data for ETH/USDT from January 1, 2020
to the current date, then saves it as a CSV file.
"""

import os
import pandas as pd
from datetime import datetime
from binance.client import Client
from dotenv import load_dotenv

def load_api_keys():
    """Load Binance API keys from .env file."""
    load_dotenv()
    api_key = os.getenv('BINANCE_API_KEY')
    api_secret = os.getenv('BINANCE_API_SECRET')
    
    if not api_key or not api_secret:
        raise ValueError("Binance API keys not found in .env file")
    
    return api_key, api_secret

def fetch_eth_historical_data(start_date="1 Jan, 2020"):
    """
    Fetch ETH/USDT historical data from Binance.
    
    Args:
        start_date (str): Start date for historical data in format "1 Jan, 2020"
        
    Returns:
        pd.DataFrame: DataFrame with historical price data
    """
    print("Loading API keys...")
    api_key, api_secret = load_api_keys()
    
    print("Connecting to Binance API...")
    client = Client(api_key, api_secret)
    
    # Get ETH/USDT historical data
    print(f"Fetching ETH/USDT data from {start_date} to today...")
    klines = client.get_historical_klines(
        "ETHUSDT", 
        Client.KLINE_INTERVAL_1DAY,
        start_date
    )
    
    # Convert to DataFrame
    data = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])
    
    # Process data
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    
    # Convert price columns to float
    for col in ['open', 'high', 'low', 'close', 'volume']:
        data[col] = data[col].astype(float)
    
    # Keep only relevant columns
    data = data[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    
    return data

def save_data(data, filepath='data/eth_data.csv'):
    """Save data to CSV file."""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Save to CSV
    data.to_csv(filepath, index=False)
    print(f"Data saved to {filepath}")

if __name__ == "__main__":
    # Fetch data
    eth_data = fetch_eth_historical_data()
    
    # Print data info
    print(f"Downloaded {len(eth_data)} days of ETH/USDT data")
    print(f"Date range: {eth_data['timestamp'].min()} to {eth_data['timestamp'].max()}")
    
    # Save data
    save_data(eth_data)