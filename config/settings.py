"""
HFT Project Configuration
"""
import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"

# Data file paths
SENTIMENT_DATA_FILE = PROJECT_ROOT / "sentiment_data.csv"
DAILY_DATA_FILE = PROJECT_ROOT / "simulated_daily_data.csv"
INTRADAY_DATA_FILE = PROJECT_ROOT / "simulated_5min_data.csv"

# Output directory
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Strategy Parameters
class SentimentStrategyConfig:
    """Configuration for Twitter Sentiment Strategy"""
    TOP_N_STOCKS = 5  # Number of top stocks to select each month
    START_DATE = "2021-11-01"
    END_DATE = "2024-01-01"
    BENCHMARK = "QQQ"


class GARCHStrategyConfig:
    """Configuration for Intraday GARCH Strategy"""
    ROLLING_WINDOW = 180  # 6 months rolling window
    RSI_PERIOD = 14
    LBOS_PERIOD = 5  # Larry Williams' %R period
    START_YEAR = "2020"


class UnsupervisedStrategyConfig:
    """Configuration for Unsupervised Learning Strategy"""
    N_CLUSTERS = 4
    TARGET_RSI_VALUES = [30, 45, 55, 70]
    TARGET_CLUSTER = 3  # Momentum cluster (RSI ~70)
    TOP_N_LIQUID = 150  # Top liquid stocks to consider
    ROLLING_BETA_WINDOW = 24  # Months
    START_DATE = "2015-01-01"
    BENCHMARK = "SPY"
