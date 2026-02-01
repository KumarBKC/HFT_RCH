"""
Technical indicator calculations
"""
import pandas as pd
import numpy as np

try:
    import pandas_ta
    HAS_PANDAS_TA = True
except ImportError:
    HAS_PANDAS_TA = False


def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index.
    """
    if HAS_PANDAS_TA:
        return pandas_ta.rsi(close=close, length=period)
    
    # Manual calculation
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_garman_klass_volatility(
    open_price: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series
) -> pd.Series:
    """
    Calculate Garman-Klass volatility estimator.
    
    GK = (ln(H) - ln(L))^2 / 2 - (2*ln(2) - 1) * (ln(C) - ln(O))^2
    """
    log_hl = np.log(high) - np.log(low)
    log_co = np.log(close) - np.log(open_price)
    
    return (log_hl ** 2) / 2 - (2 * np.log(2) - 1) * (log_co ** 2)


def calculate_bollinger_bands(
    close: pd.Series,
    period: int = 20,
    std_dev: float = 2.0
) -> tuple:
    """
    Calculate Bollinger Bands.
    
    Returns:
        Tuple of (lower_band, middle_band, upper_band)
    """
    if HAS_PANDAS_TA:
        bb = pandas_ta.bbands(close=close, length=period, std=std_dev)
        return bb.iloc[:, 0], bb.iloc[:, 1], bb.iloc[:, 2]
    
    # Manual calculation
    middle = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    lower = middle - (std_dev * std)
    upper = middle + (std_dev * std)
    
    return lower, middle, upper


def calculate_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14
) -> pd.Series:
    """
    Calculate Average True Range.
    """
    if HAS_PANDAS_TA:
        return pandas_ta.atr(high=high, low=low, close=close, length=period)
    
    # Manual calculation
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    return true_range.rolling(window=period).mean()


def calculate_macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9
) -> tuple:
    """
    Calculate MACD indicator.
    
    Returns:
        Tuple of (macd_line, signal_line, histogram)
    """
    if HAS_PANDAS_TA:
        macd_df = pandas_ta.macd(close=close, fast=fast, slow=slow, signal=signal)
        return macd_df.iloc[:, 0], macd_df.iloc[:, 1], macd_df.iloc[:, 2]
    
    # Manual calculation
    ema_fast = close.ewm(span=fast).mean()
    ema_slow = close.ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def calculate_dollar_volume(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    Calculate dollar volume (close price * volume).
    """
    return close * volume


def calculate_lbos(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 5
) -> pd.Series:
    """
    Calculate Larry Williams' %R (Lower Bound Oscillator).
    
    Range: -100 to 0
    - Values near 0 indicate overbought
    - Values near -100 indicate oversold
    """
    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()
    
    return -100 * (highest_high - close) / (highest_high - lowest_low)
