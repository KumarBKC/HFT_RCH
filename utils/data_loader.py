"""
Data loading utilities for HFT strategies
"""
import pandas as pd
import numpy as np
from pathlib import Path


def load_sentiment_data(filepath: Path) -> pd.DataFrame:
    """
    Load Twitter sentiment data from CSV.
    
    Returns DataFrame with columns:
    - date, symbol, twitterPosts, twitterComments, twitterLikes, 
      twitterImpressions, twitterSentiment
    """
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index(['date', 'symbol'])
    
    # Calculate engagement ratio
    df['engagement_ratio'] = (
        df['twitterComments'] + df['twitterLikes']
    ) / df['twitterImpressions']
    
    # Handle infinity and NaN
    df['engagement_ratio'] = df['engagement_ratio'].replace([np.inf, -np.inf], np.nan)
    
    return df


def load_daily_data(filepath: Path) -> pd.DataFrame:
    """
    Load simulated daily OHLCV data.
    
    Returns DataFrame with columns:
    - Date, Open, High, Low, Close, Adj Close, Volume
    """
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    
    # Clean column names
    df.columns = [c.strip() for c in df.columns]
    
    # Remove any trailing empty columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    return df


def load_intraday_data(filepath: Path) -> pd.DataFrame:
    """
    Load 5-minute intraday OHLCV data.
    
    Returns DataFrame with columns:
    - datetime, open, low, high, close, volume
    """
    df = pd.read_csv(filepath)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime')
    
    # Clean column names
    df.columns = [c.strip().lower() for c in df.columns]
    
    # Remove any trailing empty columns
    df = df.loc[:, ~df.columns.str.contains('^unnamed')]
    
    # Add date column for merging with daily data
    df['date'] = df.index.date
    
    return df


def filter_stocks_by_engagement(df: pd.DataFrame, min_posts: int = 20) -> pd.DataFrame:
    """
    Filter out stocks with minimal Twitter activity.
    
    Args:
        df: DataFrame with Twitter sentiment data
        min_posts: Minimum number of posts to include stock
    
    Returns:
        Filtered DataFrame
    """
    # Calculate average posts per stock
    avg_posts = df.groupby(level='symbol')['twitterPosts'].mean()
    valid_symbols = avg_posts[avg_posts >= min_posts].index
    
    return df[df.index.get_level_values('symbol').isin(valid_symbols)]
