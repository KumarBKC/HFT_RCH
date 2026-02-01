#!/usr/bin/env python
"""
HFT Trading Strategies - Main Entry Point

Run algorithmic trading strategies:
  - sentiment: Twitter Sentiment Strategy
  - garch: Intraday GARCH Strategy

Usage:
    python main.py --strategy sentiment
    python main.py --strategy garch
    python main.py --all
"""
import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def run_sentiment():
    """Run Twitter Sentiment Strategy."""
    from strategies.sentiment import run_sentiment_strategy
    print("\n" + "="*60)
    print("  Twitter Sentiment Trading Strategy")
    print("="*60 + "\n")
    return run_sentiment_strategy()


def run_garch():
    """Run Intraday GARCH Strategy."""
    from strategies.intraday_garch import run_garch_strategy
    print("\n" + "="*60)
    print("  Intraday GARCH Trading Strategy")
    print("="*60 + "\n")
    return run_garch_strategy()


def main():
    parser = argparse.ArgumentParser(
        description="HFT Trading Strategies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py --strategy sentiment   # Run Twitter Sentiment Strategy
    python main.py --strategy garch       # Run Intraday GARCH Strategy
    python main.py --all                  # Run all strategies
    python main.py --list                 # List available strategies
        """
    )
    
    parser.add_argument(
        '--strategy', '-s',
        type=str,
        choices=['sentiment', 'garch'],
        help='Strategy to run'
    )
    
    parser.add_argument(
        '--all', '-a',
        action='store_true',
        help='Run all strategies'
    )
    
    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='List available strategies'
    )
    
    args = parser.parse_args()
    
    if args.list:
        print("\nAvailable Strategies:")
        print("-" * 40)
        print("  sentiment  - Twitter Sentiment Strategy")
        print("               Uses engagement ratio to rank stocks")
        print("               Data: sentiment_data.csv")
        print()
        print("  garch      - Intraday GARCH Strategy")
        print("               Uses volatility prediction for signals")
        print("               Data: simulated_daily_data.csv,")
        print("                     simulated_5min_data.csv")
        print("-" * 40)
        return
    
    if args.all:
        print("\n" + "="*60)
        print("  Running All Strategies")
        print("="*60)
        
        results = {}
        
        try:
            results['sentiment'] = run_sentiment()
        except Exception as e:
            print(f"Error running sentiment strategy: {e}")
        
        try:
            results['garch'] = run_garch()
        except Exception as e:
            print(f"Error running GARCH strategy: {e}")
        
        print("\n" + "="*60)
        print("  All Strategies Complete!")
        print("="*60 + "\n")
        return results
    
    if args.strategy == 'sentiment':
        return run_sentiment()
    
    elif args.strategy == 'garch':
        return run_garch()
    
    else:
        parser.print_help()
        print("\n\nNo strategy specified. Use --strategy or --all to run strategies.")


if __name__ == "__main__":
    main()
