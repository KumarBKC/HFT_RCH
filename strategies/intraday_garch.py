"""
Intraday GARCH Trading Strategy

This strategy:
1. Loads daily and 5-minute intraday data
2. Fits GARCH(1,1) model in rolling window to predict volatility
3. Calculates prediction premium (predicted vs realized volatility)
4. Generates daily signals based on prediction premium
5. Uses intraday indicators (RSI, LBOS) to time entry
6. Holds position until end of day
"""
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from strategies.base import BaseStrategy
from utils.data_loader import load_daily_data, load_intraday_data
from utils.features import calculate_rsi, calculate_lbos
from utils.visualization import plot_cumulative_returns, print_performance_metrics
from config import DAILY_DATA_FILE, INTRADAY_DATA_FILE, OUTPUT_DIR, GARCHStrategyConfig


class GARCHStrategy(BaseStrategy):
    """Intraday GARCH Trading Strategy."""
    
    def __init__(self, config: GARCHStrategyConfig = None):
        super().__init__("Intraday GARCH Strategy")
        self.config = config or GARCHStrategyConfig()
        self.daily_df = None
        self.intraday_df = None
        self.final_df = None
        
    def load_data(self):
        """Load daily and intraday data."""
        print("Loading daily data...")
        self.daily_df = load_daily_data(DAILY_DATA_FILE)
        
        print("Loading intraday data...")
        self.intraday_df = load_intraday_data(INTRADAY_DATA_FILE)
        
        # Calculate daily log returns
        self.daily_df['log_ret'] = np.log(self.daily_df['Adj Close']).diff()
        
        # Calculate rolling variance
        self.daily_df['variance'] = (
            self.daily_df['log_ret']
            .rolling(self.config.ROLLING_WINDOW)
            .var()
        )
        
        # Filter to start year
        self.daily_df = self.daily_df[self.config.START_YEAR:]
        
        print(f"Daily data: {len(self.daily_df)} rows")
        print(f"Intraday data: {len(self.intraday_df)} rows")
        
    def predict_volatility(self, returns: pd.Series) -> float:
        """
        Fit GARCH(1,1) and predict next day volatility.
        """
        try:
            from arch import arch_model
            
            # Scale returns for numerical stability
            scaled_returns = returns * 100
            
            # Fit GARCH(1,1)
            model = arch_model(
                scaled_returns,
                vol='GARCH',
                p=1,
                q=1,
                rescale=False
            )
            
            result = model.fit(disp='off', show_warning=False)
            
            # Forecast 1-day ahead
            forecast = result.forecast(horizon=1)
            predicted_var = forecast.variance.values[-1, 0] / 10000  # Unscale
            
            return predicted_var
            
        except Exception as e:
            # Return NaN if model fails
            return np.nan
    
    def generate_signals(self) -> pd.DataFrame:
        """
        Generate trading signals based on GARCH predictions.
        """
        if self.daily_df is None:
            self.load_data()
        
        print("Generating GARCH predictions (this may take a while)...")
        
        # Apply rolling GARCH predictions
        predictions = []
        window = self.config.ROLLING_WINDOW
        
        for i in range(window, len(self.daily_df)):
            if i % 50 == 0:
                print(f"  Processing {i}/{len(self.daily_df)}...")
            
            returns_window = self.daily_df['log_ret'].iloc[i-window:i].dropna()
            
            if len(returns_window) >= window // 2:
                pred = self.predict_volatility(returns_window)
            else:
                pred = np.nan
            
            predictions.append(pred)
        
        # Add NaN for initial period
        self.daily_df['predictions'] = [np.nan] * window + predictions
        
        # Calculate prediction premium
        self.daily_df['prediction_premium'] = (
            (self.daily_df['predictions'] - self.daily_df['variance']) 
            / self.daily_df['variance']
        )
        
        # Rolling std of prediction premium
        self.daily_df['premium_std'] = (
            self.daily_df['prediction_premium']
            .rolling(self.config.ROLLING_WINDOW)
            .std()
        )
        
        # Generate daily signal
        # Signal = 1 if prediction_premium > premium_std (expect high vol, go short)
        # Signal = -1 if prediction_premium < -premium_std (expect low vol, go long)
        def get_signal(row):
            if pd.isna(row['prediction_premium']) or pd.isna(row['premium_std']):
                return np.nan
            if row['prediction_premium'] > row['premium_std']:
                return 1  # High vol predicted - trade accordingly
            elif row['prediction_premium'] < -row['premium_std']:
                return -1  # Low vol predicted
            else:
                return np.nan  # No signal
        
        self.daily_df['signal_daily'] = self.daily_df.apply(get_signal, axis=1)
        
        print(f"Generated {self.daily_df['signal_daily'].notna().sum()} daily signals")
        
        return self.daily_df
    
    def calculate_intraday_signals(self):
        """
        Merge with intraday data and calculate intraday signals.
        """
        print("Calculating intraday signals...")
        
        # Prepare daily signals
        daily_signals = self.daily_df[['signal_daily']].copy()
        daily_signals.index = daily_signals.index.date
        daily_signals = daily_signals.reset_index()
        daily_signals.columns = ['date', 'signal_daily']
        
        # Merge intraday with daily signals
        intraday = self.intraday_df.reset_index()
        intraday['date'] = pd.to_datetime(intraday['date'])
        daily_signals['date'] = pd.to_datetime(daily_signals['date'])
        
        self.final_df = intraday.merge(
            daily_signals,
            on='date',
            how='left'
        )
        self.final_df = self.final_df.set_index('datetime')
        
        # Calculate intraday indicators
        self.final_df['rsi'] = calculate_rsi(
            self.final_df['close'],
            period=self.config.RSI_PERIOD
        )
        
        self.final_df['lbos'] = calculate_lbos(
            self.final_df['high'],
            self.final_df['low'],
            self.final_df['close'],
            period=self.config.LBOS_PERIOD
        )
        
        # Generate intraday signal
        # Signal = 1 if RSI < 30 and LBOS < -80 (oversold)
        # Signal = -1 if RSI > 70 and LBOS > -20 (overbought)
        def intraday_signal(row):
            if row['rsi'] < 30 and row['lbos'] < -80:
                return 1  # Oversold - potential buy
            elif row['rsi'] > 70 and row['lbos'] > -20:
                return -1  # Overbought - potential sell
            else:
                return np.nan
        
        self.final_df['signal_intraday'] = self.final_df.apply(intraday_signal, axis=1)
        
        # Combine signals
        # Final position based on daily and intraday signals alignment
        def get_position(row):
            if pd.isna(row['signal_daily']) or pd.isna(row['signal_intraday']):
                return np.nan
            if row['signal_daily'] == 1 and row['signal_intraday'] == 1:
                return -1  # Both say high vol/oversold - go short
            elif row['signal_daily'] == -1 and row['signal_intraday'] == -1:
                return 1  # Both say low vol/overbought - go long
            else:
                return np.nan
        
        self.final_df['return_sign'] = self.final_df.apply(get_position, axis=1)
        
        # Forward fill position within each day (hold until end of day)
        self.final_df['return_sign'] = (
            self.final_df
            .groupby(pd.Grouper(freq='D'))['return_sign']
            .transform(lambda x: x.ffill())
        )
        
    def run(self) -> pd.DataFrame:
        """
        Execute the strategy and calculate returns.
        """
        # Generate signals
        self.generate_signals()
        self.calculate_intraday_signals()
        
        print("Calculating strategy returns...")
        
        # Calculate intraday returns
        self.final_df['return'] = (
            np.log(self.final_df['close']).diff()
        )
        
        # Calculate strategy returns (position * return)
        self.final_df['strategy_return'] = (
            self.final_df['return_sign'].shift(1) * self.final_df['return']
        )
        
        # Aggregate to daily returns
        daily_returns = (
            self.final_df
            .groupby(pd.Grouper(freq='D'))
            ['strategy_return']
            .sum()
            .dropna()
        )
        
        # Also get buy-and-hold returns for comparison
        buy_hold_returns = (
            self.final_df
            .groupby(pd.Grouper(freq='D'))
            ['return']
            .sum()
            .dropna()
        )
        
        result_df = pd.DataFrame({
            'strategy_return': daily_returns,
            'buy_hold_return': buy_hold_returns
        }).dropna()
        
        self.returns = result_df['strategy_return']
        
        return result_df
    
    def plot_results(self, save: bool = True):
        """Plot strategy performance."""
        if self.returns is None:
            raise ValueError("Strategy has not been run yet. Call run() first.")
        
        result_df = pd.DataFrame({
            'GARCH Strategy': self.returns,
            'Buy & Hold': self.final_df.groupby(pd.Grouper(freq='D'))['return'].sum()
        }).dropna()
        
        plot_cumulative_returns(
            result_df,
            title=self.name,
            save_path=OUTPUT_DIR / 'garch_strategy_returns.png' if save else None
        )
        
        print_performance_metrics(self.returns, self.name)


def run_garch_strategy():
    """Run the Intraday GARCH Strategy."""
    strategy = GARCHStrategy()
    results = strategy.run()
    strategy.plot_results()
    return results


if __name__ == "__main__":
    run_garch_strategy()
