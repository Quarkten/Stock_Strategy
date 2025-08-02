import pandas as pd
import numpy as np

def add_multi_timeframe_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds multi-timeframe features to the DataFrame.

    Args:
        df: The input DataFrame with OHLCV data.

    Returns:
        The DataFrame with the added multi-timeframe features.
    """
    # Daily features
    daily_df = df.resample('D').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })
    daily_df['daily_ema_20'] = daily_df['close'].ewm(span=20, adjust=False).mean()
    daily_df['daily_rsi_14'] = _rsi(daily_df['close'], 14)

    # Weekly features
    weekly_df = df.resample('W').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })
    weekly_df['weekly_ema_20'] = weekly_df['close'].ewm(span=20, adjust=False).mean()

    # Add support/resistance levels
    daily_df['support'] = daily_df['low'].rolling(window=14).min()
    daily_df['resistance'] = daily_df['high'].rolling(window=14).max()

    # Merge features back to the original DataFrame
    df = df.merge(daily_df[['daily_ema_20', 'daily_rsi_14', 'support', 'resistance']], how='left', left_on=df.index.date, right_on=daily_df.index.date)
    df = df.merge(weekly_df[['weekly_ema_20']], how='left', left_on=df.index.to_period('W'), right_on=weekly_df.index.to_period('W'))

    # Forward fill the new features
    df[['daily_ema_20', 'daily_rsi_14', 'support', 'resistance', 'weekly_ema_20']] = df[['daily_ema_20', 'daily_rsi_14', 'support', 'resistance', 'weekly_ema_20']].fillna(method='ffill')

    return df

def _rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))
