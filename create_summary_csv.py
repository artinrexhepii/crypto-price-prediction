"""
create_summary_csv.py: Generate summary of ETH/USDT data from Binance

This script reads the raw ETH/USDT data and generates a monthly and yearly summary,
saving the results to CSV files for easy analysis.
"""

import os
import pandas as pd
import numpy as np

def create_monthly_summary(data_path='data/eth_data.csv'):
    """
    Create a monthly summary of ETH price data
    
    Args:
        data_path (str): Path to the raw ETH data CSV
    
    Returns:
        pd.DataFrame: Monthly summary dataframe
    """
    print(f"Reading data from {data_path}...")
    
    # Read the data
    df = pd.read_csv(data_path)
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Extract year and month
    df['year'] = df['timestamp'].dt.year
    df['month'] = df['timestamp'].dt.month
    df['year_month'] = df['timestamp'].dt.strftime('%Y-%m')
    
    # Calculate daily returns
    df['daily_return'] = df['close'].pct_change() * 100
    
    # Group by year and month
    monthly_summary = df.groupby('year_month').agg(
        start_date=('timestamp', 'min'),
        end_date=('timestamp', 'max'),
        trading_days=('timestamp', 'count'),
        open_price=('open', 'first'),
        close_price=('close', 'last'),
        avg_price=('close', 'mean'),
        min_price=('low', 'min'),
        max_price=('high', 'max'),
        total_volume=('volume', 'sum'),
        avg_volume=('volume', 'mean'),
        volatility=('daily_return', lambda x: x.std()),
        month_return=('close', lambda x: (x.iloc[-1] / x.iloc[0] - 1) * 100)
    ).reset_index()
    
    # Calculate price range percentage
    monthly_summary['price_range_pct'] = ((monthly_summary['max_price'] - monthly_summary['min_price']) 
                                         / monthly_summary['min_price'] * 100)
    
    # Reorder columns
    monthly_summary = monthly_summary[[
        'year_month', 'start_date', 'end_date', 'trading_days',
        'open_price', 'close_price', 'avg_price', 'min_price', 'max_price',
        'price_range_pct', 'month_return', 'volatility',
        'total_volume', 'avg_volume'
    ]]
    
    return monthly_summary

def create_yearly_summary(monthly_data):
    """
    Create a yearly summary from monthly data
    
    Args:
        monthly_data (pd.DataFrame): Monthly summary dataframe
    
    Returns:
        pd.DataFrame: Yearly summary dataframe
    """
    # Extract year from year_month
    monthly_data['year'] = monthly_data['year_month'].str[:4]
    
    # Group by year
    yearly_summary = monthly_data.groupby('year').agg(
        start_date=('start_date', 'min'),
        end_date=('end_date', 'max'),
        trading_days=('trading_days', 'sum'),
        open_price=('open_price', lambda x: x.iloc[0]),  # First month's open price
        close_price=('close_price', lambda x: x.iloc[-1]),  # Last month's close price
        avg_price=('avg_price', 'mean'),
        min_price=('min_price', 'min'),
        max_price=('max_price', 'max'),
        avg_monthly_volume=('total_volume', 'mean'),
        total_volume=('total_volume', 'sum'),
        avg_volatility=('volatility', 'mean'),
        year_return=('open_price', lambda x: (x.iloc[-1] / x.iloc[0] - 1) * 100)
    ).reset_index()
    
    # Calculate price range percentage
    yearly_summary['price_range_pct'] = ((yearly_summary['max_price'] - yearly_summary['min_price']) 
                                        / yearly_summary['min_price'] * 100)
    
    # Reorder columns
    yearly_summary = yearly_summary[[
        'year', 'start_date', 'end_date', 'trading_days',
        'open_price', 'close_price', 'avg_price', 'min_price', 'max_price',
        'price_range_pct', 'year_return', 'avg_volatility',
        'total_volume', 'avg_monthly_volume'
    ]]
    
    return yearly_summary

def create_full_dataset_summary(data_path='data/eth_data.csv'):
    """
    Create a single-row summary of the entire dataset
    
    Args:
        data_path (str): Path to the raw ETH data CSV
    
    Returns:
        pd.DataFrame: Summary dataframe with one row
    """
    # Read the data
    df = pd.read_csv(data_path)
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Calculate daily returns
    df['daily_return'] = df['close'].pct_change() * 100
    
    # Create summary dictionary
    summary = {
        'start_date': df['timestamp'].min(),
        'end_date': df['timestamp'].max(),
        'total_days': len(df),
        'starting_price': df.iloc[0]['open'],
        'ending_price': df.iloc[-1]['close'],
        'all_time_low': df['low'].min(),
        'all_time_high': df['high'].max(),
        'avg_price': df['close'].mean(),
        'median_price': df['close'].median(),
        'price_range_pct': (df['high'].max() - df['low'].min()) / df['low'].min() * 100,
        'total_return_pct': (df.iloc[-1]['close'] / df.iloc[0]['open'] - 1) * 100,
        'annualized_return_pct': ((df.iloc[-1]['close'] / df.iloc[0]['open']) ** 
                                (365 / (df['timestamp'].max() - df['timestamp'].min()).days) - 1) * 100,
        'avg_daily_volume': df['volume'].mean(),
        'total_volume': df['volume'].sum(),
        'volatility': df['daily_return'].std(),
        'max_daily_gain_pct': df['daily_return'].max(),
        'max_daily_loss_pct': df['daily_return'].min(),
        'positive_days_pct': (df['daily_return'] > 0).mean() * 100
    }
    
    # Convert to DataFrame
    summary_df = pd.DataFrame([summary])
    
    return summary_df

def save_summaries():
    """Save all summary data to CSV files"""
    # Make sure results directory exists
    os.makedirs('results', exist_ok=True)
    
    # Create monthly summary
    monthly_summary = create_monthly_summary()
    monthly_summary.to_csv('results/eth_monthly_summary.csv', index=False)
    print(f"Monthly summary saved to results/eth_monthly_summary.csv with {len(monthly_summary)} rows")
    
    # Create yearly summary
    yearly_summary = create_yearly_summary(monthly_summary)
    yearly_summary.to_csv('results/eth_yearly_summary.csv', index=False)
    print(f"Yearly summary saved to results/eth_yearly_summary.csv with {len(yearly_summary)} rows")
    
    # Create full dataset summary
    full_summary = create_full_dataset_summary()
    full_summary.to_csv('results/eth_full_dataset_summary.csv', index=False)
    print(f"Full dataset summary saved to results/eth_full_dataset_summary.csv")
    
    # Print some interesting statistics
    print("\nInteresting Statistics:")
    print(f"Total trading days: {full_summary['total_days'].iloc[0]}")
    print(f"Price range: ${full_summary['all_time_low'].iloc[0]:.2f} - ${full_summary['all_time_high'].iloc[0]:.2f}")
    print(f"Total return: {full_summary['total_return_pct'].iloc[0]:.2f}%")
    print(f"Annualized return: {full_summary['annualized_return_pct'].iloc[0]:.2f}%")
    print(f"Average daily volume: {full_summary['avg_daily_volume'].iloc[0]:.2f} ETH")
    print(f"Percentage of positive days: {full_summary['positive_days_pct'].iloc[0]:.2f}%")
    
    return monthly_summary, yearly_summary, full_summary

if __name__ == "__main__":
    print("=" * 50)
    print("Creating ETH/USDT Data Summaries")
    print("=" * 50)
    monthly_summary, yearly_summary, full_summary = save_summaries()