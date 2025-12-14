"""
Strategy Optimizer
Test different parameter combinations to find optimal settings.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from itertools import product
import yaml
from typing import Dict, List
import sys
sys.path.append(str(Path(__file__).parent.parent))

from backtesting import Backtester


def optimize_strategy(
    market_data: pd.DataFrame,
    predictions: pd.DataFrame,
    param_grid: Dict[str, List]
) -> pd.DataFrame:
    """
    Test multiple parameter combinations.
    
    Args:
        market_data: Historical market data
        predictions: Model predictions
        param_grid: Dictionary of parameters to test
        
    Returns:
        DataFrame with results for each combination
    """
    print("\n🔧 Starting Strategy Optimization...")
    print("=" * 60)
    
    results = []
    
    # Generate all combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(product(*param_values))
    
    print(f"📊 Testing {len(combinations)} parameter combinations...")
    
    for i, combo in enumerate(combinations, 1):
        params = dict(zip(param_names, combo))
        
        print(f"\n[{i}/{len(combinations)}] Testing: {params}")
        
        # Run backtest with these parameters
        backtester = Backtester(
            initial_capital=100000,
            commission=params.get('commission', 0.001)
        )
        
        # Temporarily modify strategy parameters
        backtester.stop_loss_pct = params.get('stop_loss', 0.05)
        backtester.take_profit_pct = params.get('take_profit', 0.10)
        backtester.trailing_stop_pct = params.get('trailing_stop', 0.03)
        backtester.max_position_pct = params.get('max_position', 0.25)
        
        try:
            result = backtester.run_backtest(market_data, predictions)
            
            if 'error' not in result:
                results.append({
                    **params,
                    'total_return': result['total_return_pct'],
                    'sharpe_ratio': result['sharpe_ratio'],
                    'max_drawdown': result['max_drawdown_pct'],
                    'win_rate': result['win_rate_pct'],
                    'profit_factor': result['profit_factor'],
                    'total_trades': result['total_trades']
                })
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            continue
    
    results_df = pd.DataFrame(results)
    
    if len(results_df) > 0:
        # Rank by Sharpe ratio
        results_df = results_df.sort_values('sharpe_ratio', ascending=False)
        
        print("\n" + "=" * 60)
        print("🏆 TOP 5 PARAMETER COMBINATIONS")
        print("=" * 60)
        print(results_df.head().to_string(index=False))
        
        # Save results
        output_file = 'data/results/optimization_results.csv'
        results_df.to_csv(output_file, index=False)
        print(f"\n💾 Saved full results to: {output_file}")
    
    return results_df


def get_optimized_parameters() -> Dict:
    """Return best parameters from optimization results."""
    results_file = Path('data/results/optimization_results.csv')
    
    if results_file.exists():
        df = pd.read_csv(results_file)
        best = df.iloc[0].to_dict()
        return {
            'stop_loss': best.get('stop_loss', 0.05),
            'take_profit': best.get('take_profit', 0.10),
            'trailing_stop': best.get('trailing_stop', 0.03),
            'max_position': best.get('max_position', 0.25),
            'commission': best.get('commission', 0.001)
        }
    
    # Default parameters
    return {
        'stop_loss': 0.05,
        'take_profit': 0.10,
        'trailing_stop': 0.03,
        'max_position': 0.25,
        'commission': 0.001
    }


def main():
    """Run strategy optimization."""
    # Load data
    market_files = list(Path('data/raw').glob('market_data_*.csv'))
    pred_file = Path('data/processed/predictions.csv')
    
    if not market_files:
        print("❌ No market data found!")
        return
    
    latest_market = max(market_files, key=lambda x: x.stat().st_ctime)
    market_df = pd.read_csv(latest_market, parse_dates=True, index_col=0)
    
    predictions_df = pd.DataFrame()
    if pred_file.exists():
        predictions_df = pd.read_csv(pred_file)
    
    # Define parameter grid
    param_grid = {
        'stop_loss': [0.03, 0.05, 0.07],  # 3%, 5%, 7%
        'take_profit': [0.08, 0.10, 0.12, 0.15],  # 8%, 10%, 12%, 15%
        'trailing_stop': [0.02, 0.03, 0.04],  # 2%, 3%, 4%
        'max_position': [0.15, 0.20, 0.25],  # 15%, 20%, 25%
        'commission': [0.001]  # 0.1%
    }
    
    # Run optimization
    results = optimize_strategy(market_df, predictions_df, param_grid)
    
    if len(results) > 0:
        print("\n✅ Optimization complete!")
        print(f"📊 Tested {len(results)} combinations")
        print(f"🏆 Best Sharpe Ratio: {results.iloc[0]['sharpe_ratio']:.2f}")


if __name__ == "__main__":
    main()
