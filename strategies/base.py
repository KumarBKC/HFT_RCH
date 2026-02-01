"""
Base strategy class for all trading strategies
"""
from abc import ABC, abstractmethod
import pandas as pd
from pathlib import Path


class BaseStrategy(ABC):
    """Abstract base class for trading strategies."""
    
    def __init__(self, name: str):
        self.name = name
        self.returns = None
        self.positions = None
        
    @abstractmethod
    def run(self) -> pd.DataFrame:
        """
        Execute the strategy and return daily returns.
        
        Returns:
            DataFrame with daily returns
        """
        pass
    
    @abstractmethod
    def generate_signals(self) -> pd.DataFrame:
        """
        Generate trading signals.
        
        Returns:
            DataFrame with signals
        """
        pass
    
    def get_performance_summary(self) -> dict:
        """
        Calculate performance summary statistics.
        """
        if self.returns is None:
            raise ValueError("Strategy has not been run yet. Call run() first.")
        
        import numpy as np
        
        returns = self.returns
        
        # Total return
        total_return = (1 + returns).prod() - 1
        
        # Annualized metrics
        n_years = len(returns) / 252
        ann_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
        ann_vol = returns.std() * np.sqrt(252)
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0
        
        # Max drawdown
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_dd = drawdown.min()
        
        return {
            'name': self.name,
            'total_return': total_return,
            'annualized_return': ann_return,
            'annualized_volatility': ann_vol,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'win_rate': (returns > 0).sum() / len(returns),
            'trading_days': len(returns)
        }
