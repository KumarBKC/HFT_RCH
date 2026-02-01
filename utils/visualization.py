"""
Visualization utilities for HFT strategies
"""
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import numpy as np
from pathlib import Path


def setup_plot_style():
    """Set up consistent plot styling."""
    plt.style.use('ggplot')
    plt.rcParams['figure.figsize'] = (14, 6)
    plt.rcParams['font.size'] = 10


def plot_cumulative_returns(
    returns_df: pd.DataFrame,
    title: str = "Strategy Cumulative Returns",
    save_path: Path = None
):
    """
    Plot cumulative returns for one or more strategies.
    
    Args:
        returns_df: DataFrame with daily returns (columns are different strategies)
        title: Plot title
        save_path: Optional path to save the figure
    """
    setup_plot_style()
    
    # Calculate cumulative returns
    cumulative = np.exp(np.log1p(returns_df).cumsum()) - 1
    
    fig, ax = plt.subplots(figsize=(14, 6))
    cumulative.plot(ax=ax)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Return')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.show()
    return fig


def plot_strategy_comparison(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    strategy_name: str,
    benchmark_name: str,
    save_path: Path = None
):
    """
    Compare strategy returns vs benchmark.
    """
    setup_plot_style()
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Cumulative returns
    strategy_cum = np.exp(np.log1p(strategy_returns).cumsum()) - 1
    benchmark_cum = np.exp(np.log1p(benchmark_returns).cumsum()) - 1
    
    axes[0].plot(strategy_cum.index, strategy_cum.values, label=strategy_name, linewidth=2)
    axes[0].plot(benchmark_cum.index, benchmark_cum.values, label=benchmark_name, linewidth=2, alpha=0.7)
    axes[0].set_title('Cumulative Returns Comparison', fontsize=12, fontweight='bold')
    axes[0].yaxis.set_major_formatter(mtick.PercentFormatter(1))
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Rolling Sharpe ratio (252-day)
    rolling_sharpe = (
        strategy_returns.rolling(252).mean() / strategy_returns.rolling(252).std()
    ) * np.sqrt(252)
    
    axes[1].plot(rolling_sharpe.index, rolling_sharpe.values, label='Rolling Sharpe (252d)')
    axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[1].set_title('Rolling Sharpe Ratio', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
    return fig


def print_performance_metrics(returns: pd.Series, name: str = "Strategy"):
    """
    Print key performance metrics for a returns series.
    """
    # Annualized return
    total_return = (1 + returns).prod() - 1
    n_years = len(returns) / 252
    ann_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
    
    # Annualized volatility
    ann_vol = returns.std() * np.sqrt(252)
    
    # Sharpe ratio (assuming 0% risk-free rate)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0
    
    # Max drawdown
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_dd = drawdown.min()
    
    # Win rate
    win_rate = (returns > 0).sum() / len(returns)
    
    print(f"\n{'='*50}")
    print(f"  {name} Performance Metrics")
    print(f"{'='*50}")
    print(f"  Total Return:        {total_return:>10.2%}")
    print(f"  Annualized Return:   {ann_return:>10.2%}")
    print(f"  Annualized Vol:      {ann_vol:>10.2%}")
    print(f"  Sharpe Ratio:        {sharpe:>10.2f}")
    print(f"  Max Drawdown:        {max_dd:>10.2%}")
    print(f"  Win Rate:            {win_rate:>10.2%}")
    print(f"  Trading Days:        {len(returns):>10d}")
    print(f"{'='*50}\n")
