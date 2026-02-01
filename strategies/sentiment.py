"""
Twitter Sentiment Trading Strategy

This strategy:
1. Loads Twitter sentiment data
2. Calculates engagement ratio for each stock
3. Ranks stocks by engagement each month
4. Selects top N stocks to form equally-weighted portfolios
5. Rebalances monthly
"""
import pandas as pd
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from strategies.base import BaseStrategy
from utils.data_loader import load_sentiment_data, filter_stocks_by_engagement
from utils.visualization import plot_cumulative_returns, print_performance_metrics
from config import SENTIMENT_DATA_FILE, OUTPUT_DIR, SentimentStrategyConfig


class SentimentStrategy(BaseStrategy):
    """Twitter Sentiment Trading Strategy."""
    
    def __init__(self, config: SentimentStrategyConfig = None):
        super().__init__("Twitter Sentiment Strategy")
        self.config = config or SentimentStrategyConfig()
        self.sentiment_df = None
        self.portfolio_df = None
        
    def load_data(self):
        """Load and preprocess sentiment data."""
        print("Loading sentiment data...")
        self.sentiment_df = load_sentiment_data(SENTIMENT_DATA_FILE)
        
        # Filter stocks with minimal activity
        self.sentiment_df = filter_stocks_by_engagement(self.sentiment_df, min_posts=20)
        
        print(f"Loaded {len(self.sentiment_df)} records for "
              f"{self.sentiment_df.index.get_level_values('symbol').nunique()} stocks")
        
    def generate_signals(self) -> pd.DataFrame:
        """
        Generate trading signals based on engagement ranking.
        
        Returns:
            DataFrame with selected stocks for each month
        """
        if self.sentiment_df is None:
            self.load_data()
        
        print("Generating signals...")
        
        # Aggregate to monthly level
        aggregated_df = (
            self.sentiment_df
            .reset_index('symbol')
            .groupby([pd.Grouper(freq='ME'), 'symbol'])
            [['engagement_ratio']]
            .mean()
        )
        
        # Rank stocks by engagement each month
        aggregated_df['rank'] = (
            aggregated_df
            .groupby(level=0)['engagement_ratio']
            .transform(lambda x: x.rank(ascending=False))
        )
        
        # Select top N stocks
        filtered_df = aggregated_df[aggregated_df['rank'] <= self.config.TOP_N_STOCKS].copy()
        
        # Shift dates to start of next month (for forward-looking signals)
        filtered_df = filtered_df.reset_index(level=1)
        filtered_df.index = filtered_df.index + pd.DateOffset(1)
        filtered_df = filtered_df.reset_index().set_index(['date', 'symbol'])
        
        return filtered_df
    
    def run(self) -> pd.DataFrame:
        """
        Execute the strategy and calculate returns.
        
        Returns:
            DataFrame with strategy and benchmark returns
        """
        signals = self.generate_signals()
        
        # Extract dates and stocks
        dates = signals.index.get_level_values('date').unique().tolist()
        fixed_dates = {}
        for d in dates:
            fixed_dates[d.strftime('%Y-%m-%d')] = signals.xs(d, level=0).index.tolist()
        
        print(f"Trading periods: {len(fixed_dates)}")
        if fixed_dates:
            print(f"First period: {list(fixed_dates.keys())[0]}")
            print(f"Last period: {list(fixed_dates.keys())[-1]}")
        
        # Get unique stocks
        stocks_list = self.sentiment_df.index.get_level_values('symbol').unique().tolist()
        
        # Try to download prices from Yahoo Finance
        returns_df = pd.DataFrame()
        try:
            import yfinance as yf
            print("Downloading stock prices from Yahoo Finance...")
            prices_df = yf.download(
                tickers=stocks_list,
                start=self.config.START_DATE,
                end=self.config.END_DATE,
                progress=False
            )
            
            if not prices_df.empty:
                # Calculate daily returns
                returns_df = np.log(prices_df['Adj Close']).diff().dropna()
            
        except Exception as e:
            print(f"Warning: Could not download prices: {e}")

        # Fallback to simulated returns if download failed or empty
        if returns_df.empty:
            print("Using simulated returns for demonstration (Download failed or returned empty)...")
            date_range = pd.date_range(
                start=self.config.START_DATE,
                end=self.config.END_DATE,
                freq='B'
            )
            returns_df = pd.DataFrame(
                np.random.randn(len(date_range), len(stocks_list)) * 0.02,
                index=date_range,
                columns=stocks_list
            )
        
        # Calculate portfolio returns
        print("Calculating portfolio returns...")
        portfolio_returns = []
        
        for start_date in fixed_dates.keys():
            end_date = (pd.to_datetime(start_date) + pd.offsets.MonthEnd()).strftime('%Y-%m-%d')
            
            try:
                # Get stocks for this period
                stocks = fixed_dates[start_date]
                
                # Filter returns for this period
                period_returns = returns_df.loc[start_date:end_date, stocks]
                
                if not period_returns.empty:
                    # Equally weighted portfolio
                    portfolio_ret = period_returns.mean(axis=1)
                    portfolio_ret.name = 'strategy_return'
                    portfolio_returns.append(portfolio_ret)
                    
            except Exception as e:
                print(f"Skipping {start_date}: {e}")
                continue
        
        if not portfolio_returns:
            raise ValueError("No valid trading periods found")
        
        self.portfolio_df = pd.concat(portfolio_returns)
        self.portfolio_df = self.portfolio_df[~self.portfolio_df.index.duplicated(keep='first')]
        
        # Download benchmark (QQQ)
        try:
            import yfinance as yf
            print(f"Downloading {self.config.BENCHMARK} benchmark...")
            benchmark_df = yf.download(
                tickers=self.config.BENCHMARK,
                start=self.config.START_DATE,
                end=self.config.END_DATE,
                progress=False
            )
            benchmark_ret = np.log(benchmark_df['Adj Close']).diff().dropna()
            benchmark_ret.name = f'{self.config.BENCHMARK}_return'
            
        except Exception as e:
            print(f"Warning: Could not download benchmark: {e}")
            benchmark_ret = pd.Series(
                np.random.randn(len(self.portfolio_df)) * 0.015,
                index=self.portfolio_df.index,
                name=f'{self.config.BENCHMARK}_return'
            )
        
        # Merge strategy and benchmark returns
        result_df = pd.DataFrame(self.portfolio_df)
        result_df = result_df.merge(
            benchmark_ret.to_frame(),
            left_index=True,
            right_index=True,
            how='left'
        )
        
        self.returns = result_df['strategy_return']
        
        return result_df
    
    def plot_results(self, save: bool = True):
        """Plot strategy performance."""
        if self.portfolio_df is None:
            raise ValueError("Strategy has not been run yet. Call run() first.")
        
        # Get the full results DataFrame
        result_df = self.run()
        
        plot_cumulative_returns(
            result_df,
            title=f'{self.name} vs {self.config.BENCHMARK}',
            save_path=OUTPUT_DIR / 'sentiment_strategy_returns.png' if save else None
        )
        
        print_performance_metrics(self.returns, self.name)


def run_sentiment_strategy():
    """Run the Twitter Sentiment Strategy."""
    strategy = SentimentStrategy()
    results = strategy.run()
    strategy.plot_results()
    return results


if __name__ == "__main__":
    run_sentiment_strategy()
