"""
Backtesting Module
Test trading strategies on historical data with realistic simulation.
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple


class Backtester:
    """
    Backtest trading strategies on historical market data.
    """
    
    def __init__(self, initial_capital: float = 100000, commission: float = 0.001):
        """
        Initialize backtester.
        
        Args:
            initial_capital: Starting capital in rupees
            commission: Commission rate per trade (0.001 = 0.1%)
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.trades = []
        self.portfolio_value = []
        
        # Strategy parameters (can be modified for optimization)
        self.stop_loss_pct = 0.05  # 5%
        self.take_profit_pct = 0.10  # 10%
        self.trailing_stop_pct = 0.03  # 3%
        self.max_position_pct = 0.25  # 25%
        
    def run_backtest(
        self,
        market_data: pd.DataFrame,
        predictions: pd.DataFrame,
        kelly_fractions: pd.DataFrame = None
    ) -> Dict:
        """
        Run backtest on historical data.
        
        Args:
            market_data: Historical market data with OHLCV + indicators
            predictions: Model predictions with signals
            kelly_fractions: Optional Kelly Criterion position sizes
            
        Returns:
            Dictionary with backtest results
        """
        print("\n🔬 Starting Backtest...")
        print("=" * 60)
        
        # Initialize portfolio
        capital = self.initial_capital
        position = None  # Current position: {ticker, shares, entry_price, entry_date}
        
        # Track metrics
        trades_log = []
        portfolio_history = []
        
        # Merge data
        data = market_data.copy()
        if not predictions.empty:
            data = data.merge(
                predictions[['ticker', 'signal', 'model_probability', 'kelly_fraction']],
                left_on='Ticker',
                right_on='ticker',
                how='left'
            )
        
        # Iterate through each day
        tickers = data['Ticker'].unique()
        
        for ticker in tickers:
            ticker_data = data[data['Ticker'] == ticker].sort_index()
            
            for idx, row in ticker_data.iterrows():
                # Check exit conditions for existing position
                if position and position['ticker'] == ticker:
                    exit_signal = False
                    exit_reason = None
                    
                    # Stop loss: configurable %
                    if row['Close'] <= position['entry_price'] * (1 - self.stop_loss_pct):
                        exit_signal = True
                        exit_reason = 'STOP_LOSS'
                    
                    # Take profit: configurable %
                    elif row['Close'] >= position['entry_price'] * (1 + self.take_profit_pct):
                        exit_signal = True
                        exit_reason = 'TAKE_PROFIT'
                    
                    # Trailing stop: configurable % from highest
                    elif 'highest' in position and row['Close'] <= position['highest'] * (1 - self.trailing_stop_pct):
                        exit_signal = True
                        exit_reason = 'TRAILING_STOP'
                    
                    # Signal reversal
                    elif 'signal' in row and row['signal'] == 0:
                        exit_signal = True
                        exit_reason = 'SIGNAL_EXIT'
                    
                    # Execute exit
                    if exit_signal:
                        exit_price = row['Close']
                        pnl = (exit_price - position['entry_price']) * position['shares']
                        pnl_pct = ((exit_price / position['entry_price']) - 1) * 100
                        
                        # Apply commission
                        commission_cost = exit_price * position['shares'] * self.commission
                        pnl -= commission_cost
                        
                        # Update capital
                        capital += (exit_price * position['shares']) - commission_cost
                        
                        # Log trade
                        holding_period = (idx - position['entry_date']) if isinstance(idx - position['entry_date'], int) else (idx - position['entry_date']).days
                        
                        trades_log.append({
                            'ticker': ticker,
                            'entry_date': position['entry_date'],
                            'exit_date': idx,
                            'entry_price': position['entry_price'],
                            'exit_price': exit_price,
                            'shares': position['shares'],
                            'pnl': pnl,
                            'pnl_pct': pnl_pct,
                            'exit_reason': exit_reason,
                            'holding_days': holding_period
                        })
                        
                        position = None
                    
                    # Update highest price for trailing stop
                    elif 'highest' not in position or row['High'] > position['highest']:
                        position['highest'] = row['High']
                
                # Check entry conditions
                if not position and 'signal' in row and row['signal'] == 1:
                    # Calculate position size
                    if kelly_fractions is not None and 'kelly_fraction' in row:
                        position_size = capital * min(row['kelly_fraction'], self.max_position_pct)
                    else:
                        position_size = capital * 0.10  # Default 10% per position
                    
                    shares = int(position_size / row['Close'])
                    
                    if shares > 0:
                        entry_price = row['Close']
                        commission_cost = entry_price * shares * self.commission
                        
                        # Update capital
                        capital -= (entry_price * shares) + commission_cost
                        
                        # Create position
                        position = {
                            'ticker': ticker,
                            'shares': shares,
                            'entry_price': entry_price,
                            'entry_date': idx,
                            'highest': row['High']
                        }
                
                # Track portfolio value
                portfolio_val = capital
                if position:
                    portfolio_val += position['shares'] * row['Close']
                
                portfolio_history.append({
                    'date': idx,
                    'portfolio_value': portfolio_val,
                    'capital': capital,
                    'ticker': ticker
                })
        
        # Calculate metrics
        trades_df = pd.DataFrame(trades_log)
        portfolio_df = pd.DataFrame(portfolio_history)
        
        if len(trades_df) > 0:
            total_return = ((portfolio_df['portfolio_value'].iloc[-1] / self.initial_capital) - 1) * 100
            win_rate = (trades_df['pnl'] > 0).sum() / len(trades_df) * 100
            avg_win = trades_df[trades_df['pnl'] > 0]['pnl_pct'].mean()
            avg_loss = trades_df[trades_df['pnl'] < 0]['pnl_pct'].mean()
            profit_factor = abs(trades_df[trades_df['pnl'] > 0]['pnl'].sum() / trades_df[trades_df['pnl'] < 0]['pnl'].sum()) if (trades_df['pnl'] < 0).any() else np.inf
            max_drawdown = self._calculate_max_drawdown(portfolio_df['portfolio_value'])
            sharpe_ratio = self._calculate_sharpe_ratio(portfolio_df['portfolio_value'])
            
            results = {
                'total_return_pct': total_return,
                'final_capital': portfolio_df['portfolio_value'].iloc[-1],
                'total_trades': len(trades_df),
                'winning_trades': (trades_df['pnl'] > 0).sum(),
                'losing_trades': (trades_df['pnl'] < 0).sum(),
                'win_rate_pct': win_rate,
                'avg_win_pct': avg_win,
                'avg_loss_pct': avg_loss,
                'profit_factor': profit_factor,
                'max_drawdown_pct': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'avg_holding_days': trades_df['holding_days'].mean(),
                'trades_df': trades_df,
                'portfolio_df': portfolio_df
            }
        else:
            results = {
                'total_return_pct': 0,
                'final_capital': self.initial_capital,
                'total_trades': 0,
                'error': 'No trades executed'
            }
        
        self._print_results(results)
        return results
    
    def _calculate_max_drawdown(self, portfolio_values: pd.Series) -> float:
        """Calculate maximum drawdown percentage."""
        cummax = portfolio_values.cummax()
        drawdown = (portfolio_values - cummax) / cummax * 100
        return drawdown.min()
    
    def _calculate_sharpe_ratio(self, portfolio_values: pd.Series, risk_free_rate: float = 0.05) -> float:
        """Calculate annualized Sharpe ratio."""
        returns = portfolio_values.pct_change().dropna()
        if len(returns) == 0:
            return 0
        
        excess_returns = returns - (risk_free_rate / 252)  # Daily risk-free rate
        if excess_returns.std() == 0:
            return 0
        
        sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)
        return sharpe
    
    def _print_results(self, results: Dict):
        """Print backtest results."""
        print("\n" + "=" * 60)
        print("📊 BACKTEST RESULTS")
        print("=" * 60)
        
        if 'error' in results:
            print(f"⚠️  {results['error']}")
            return
        
        print(f"💰 Initial Capital:      ₹{self.initial_capital:,.2f}")
        print(f"💰 Final Capital:        ₹{results['final_capital']:,.2f}")
        print(f"📈 Total Return:         {results['total_return_pct']:+.2f}%")
        print(f"\n📊 Trade Statistics:")
        print(f"   Total Trades:         {results['total_trades']}")
        print(f"   Winning Trades:       {results['winning_trades']} ({results['win_rate_pct']:.1f}%)")
        print(f"   Losing Trades:        {results['losing_trades']}")
        print(f"   Avg Win:              {results['avg_win_pct']:+.2f}%")
        print(f"   Avg Loss:             {results['avg_loss_pct']:+.2f}%")
        print(f"   Profit Factor:        {results['profit_factor']:.2f}")
        print(f"   Avg Holding Period:   {results['avg_holding_days']:.1f} days")
        print(f"\n📉 Risk Metrics:")
        print(f"   Max Drawdown:         {results['max_drawdown_pct']:.2f}%")
        print(f"   Sharpe Ratio:         {results['sharpe_ratio']:.2f}")
        print("=" * 60)
    
    def plot_results(self, results: Dict, save_path: str = None):
        """Plot backtest results."""
        if 'error' in results:
            print(f"⚠️  Cannot plot: {results['error']}")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Backtest Results', fontsize=16, fontweight='bold')
        
        # Portfolio value over time
        portfolio_df = results['portfolio_df'].groupby('date')['portfolio_value'].first()
        axes[0, 0].plot(portfolio_df.index, portfolio_df.values, linewidth=2, color='#D71921')
        axes[0, 0].axhline(y=self.initial_capital, color='gray', linestyle='--', label='Initial Capital')
        axes[0, 0].set_title('Portfolio Value Over Time')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Value (₹)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Trade PnL distribution
        trades_df = results['trades_df']
        axes[0, 1].hist(trades_df['pnl_pct'], bins=20, color='#D71921', alpha=0.7, edgecolor='black')
        axes[0, 1].axvline(x=0, color='gray', linestyle='--')
        axes[0, 1].set_title('Trade P&L Distribution')
        axes[0, 1].set_xlabel('P&L (%)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Cumulative returns
        returns = portfolio_df.pct_change().fillna(0)
        cumulative_returns = (1 + returns).cumprod()
        axes[1, 0].plot(cumulative_returns.index, (cumulative_returns - 1) * 100, linewidth=2, color='#D71921')
        axes[1, 0].set_title('Cumulative Returns')
        axes[1, 0].set_xlabel('Date')
        axes[1, 0].set_ylabel('Return (%)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Drawdown
        cummax = portfolio_df.cummax()
        drawdown = (portfolio_df - cummax) / cummax * 100
        axes[1, 1].fill_between(drawdown.index, drawdown.values, 0, color='#D71921', alpha=0.5)
        axes[1, 1].set_title('Drawdown')
        axes[1, 1].set_xlabel('Date')
        axes[1, 1].set_ylabel('Drawdown (%)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Saved backtest plot to: {save_path}")
        
        plt.show()


def main():
    """Run backtest on latest data."""
    # Load config
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load latest data
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
    
    print(f"\n📂 Loading data...")
    print(f"   Market data: {latest_market.name}")
    print(f"   Predictions: {'predictions.csv' if not predictions_df.empty else 'None (using buy-and-hold)'}")
    
    # Run backtest
    backtester = Backtester(initial_capital=100000, commission=0.001)
    results = backtester.run_backtest(market_df, predictions_df)
    
    # Plot results
    if 'error' not in results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = f"data/results/backtest_{timestamp}.png"
        Path('data/results').mkdir(exist_ok=True)
        backtester.plot_results(results, save_path=plot_path)
        
        # Save detailed results
        results['trades_df'].to_csv(f"data/results/backtest_trades_{timestamp}.csv", index=False)
        print(f"\n💾 Saved trade log to: data/results/backtest_trades_{timestamp}.csv")


if __name__ == "__main__":
    main()
