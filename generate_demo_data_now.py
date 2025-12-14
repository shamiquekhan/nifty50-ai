"""
Generate Demo Data for Dashboard
Creates sample market data, sentiment, and predictions for immediate dashboard display
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

print("=" * 60)
print("GENERATING DEMO DATA FOR DASHBOARD")
print("=" * 60)

# Create directories
Path('data/raw').mkdir(parents=True, exist_ok=True)
Path('data/processed').mkdir(parents=True, exist_ok=True)
Path('models').mkdir(parents=True, exist_ok=True)

# NIFTY50 tickers
tickers = [
    'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK',
    'HINDUNILVR', 'ITC', 'SBIN', 'BHARTIARTL', 'KOTAKBANK',
    'BAJFINANCE', 'LT', 'AXISBANK', 'ASIANPAINT', 'MARUTI',
    'HCLTECH', 'SUNPHARMA', 'TITAN', 'ULTRACEMCO', 'NESTLEIND',
    'WIPRO', 'ADANIENT', 'ONGC', 'NTPC', 'POWERGRID',
    'M&M', 'BAJAJFINSV', 'TATAMOTORS', 'TATASTEEL', 'TECHM',
    'JSWSTEEL', 'INDUSINDBK', 'APOLLOHOSP', 'DRREDDY', 'DIVISLAB',
    'BRITANNIA', 'CIPLA', 'EICHERMOT', 'HINDALCO', 'HEROMOTOCO',
    'GRASIM', 'BPCL', 'COALINDIA', 'UPL', 'TATACONSUM',
    'ADANIPORTS', 'BAJAJ-AUTO', 'SHREECEM', 'SBILIFE', 'LTIM'
]

# Generate dates (last 90 days)
end_date = datetime.now()
start_date = end_date - timedelta(days=90)
dates = pd.date_range(start=start_date, end=end_date, freq='D')

# Generate market data
print("\n>>> Generating market data...")
market_data = []

for ticker in tickers:
    # Base price (random between 500-5000)
    base_price = np.random.uniform(500, 5000)
    
    # Generate price series with trend
    trend = np.random.choice([-0.001, 0, 0.001])
    volatility = np.random.uniform(0.01, 0.03)
    
    prices = [base_price]
    for i in range(len(dates) - 1):
        change = np.random.normal(trend, volatility)
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    for i, date in enumerate(dates):
        # OHLC
        close = prices[i]
        open_price = close * (1 + np.random.uniform(-0.01, 0.01))
        high = max(open_price, close) * (1 + np.random.uniform(0, 0.02))
        low = min(open_price, close) * (1 - np.random.uniform(0, 0.02))
        volume = np.random.randint(1000000, 10000000)
        
        # Technical indicators
        rsi_14 = np.random.uniform(30, 70)
        atr_14 = close * np.random.uniform(0.01, 0.03)
        macd = np.random.uniform(-5, 5)
        signal = np.random.uniform(-5, 5)
        bb_upper = close * 1.02
        bb_lower = close * 0.98
        
        market_data.append({
            'Ticker': ticker,
            'Date': date,
            'Open': open_price,
            'High': high,
            'Low': low,
            'Close': close,
            'Volume': volume,
            'RSI_14': rsi_14,
            'ATRr_14': atr_14 / close,
            'MACD_12_26_9': macd,
            'MACDs_12_26_9': signal,
            'BBU_20_2.0': bb_upper,
            'BBL_20_2.0': bb_lower,
            'SMA_50': close * (1 + np.random.uniform(-0.02, 0.02)),
            'EMA_20': close * (1 + np.random.uniform(-0.01, 0.01))
        })

market_df = pd.DataFrame(market_data)
market_df.set_index('Date', inplace=True)

# Save market data
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
market_file = f'data/raw/market_data_{timestamp}.csv'
market_df.to_csv(market_file)
print(f"✓ Created: {market_file}")
print(f"  Records: {len(market_df)}")

# Generate sentiment data
print("\n>>> Generating sentiment data...")
sentiment_data = []

for date in dates[-30:]:  # Last 30 days
    sentiment_mean = np.random.uniform(-0.2, 0.2)
    sentiment_std = np.random.uniform(0.05, 0.15)
    article_count = np.random.randint(10, 30)
    
    if sentiment_mean > 0.1:
        label = 'positive'
    elif sentiment_mean < -0.1:
        label = 'negative'
    else:
        label = 'neutral'
    
    sentiment_data.append({
        'date': date,
        'sentiment_mean': sentiment_mean,
        'sentiment_std': sentiment_std,
        'article_count': article_count,
        'sentiment_label': label,
        'sentiment_ma_5': sentiment_mean * (1 + np.random.uniform(-0.1, 0.1))
    })

sentiment_df = pd.DataFrame(sentiment_data)
sentiment_file = f'data/processed/daily_sentiment_{timestamp}.csv'
sentiment_df.to_csv(sentiment_file, index=False)
print(f"✓ Created: {sentiment_file}")
print(f"  Records: {len(sentiment_df)}")

# Generate predictions
print("\n>>> Generating AI predictions...")
predictions = []

for ticker in tickers[:10]:  # Top 10 stocks
    ticker_data = market_df[market_df['Ticker'] == ticker].iloc[-1]
    
    # Generate prediction
    model_probability = np.random.uniform(0.4, 0.8)
    
    # Determine action
    if model_probability > 0.65:
        action = 'BUY'
    elif model_probability < 0.45:
        action = 'SHORT'
    else:
        action = 'WAIT'
    
    # Kelly Criterion
    win_rate = model_probability
    avg_win = 0.08
    avg_loss = 0.03
    kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
    kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
    
    # Position size
    capital = 100000
    position_size = capital * kelly_fraction
    
    # Alignment
    latest_sentiment = sentiment_df.iloc[-1]['sentiment_mean']
    is_aligned = (action == 'BUY' and latest_sentiment > 0) or (action == 'SHORT' and latest_sentiment < 0)
    
    predictions.append({
        'ticker': ticker,
        'signal': 1 if action == 'BUY' else 0,
        'action': action,
        'model_probability': model_probability,
        'kelly_fraction': kelly_fraction,
        'position_size': position_size,
        'is_aligned': is_aligned,
        'latest_price': ticker_data['Close'],
        'rsi': ticker_data['RSI_14'],
        'prediction_date': end_date.strftime('%Y-%m-%d')
    })

predictions_df = pd.DataFrame(predictions)
predictions_file = 'data/processed/predictions.csv'
predictions_df.to_csv(predictions_file, index=False)
print(f"✓ Created: {predictions_file}")
print(f"  Predictions: {len(predictions_df)}")

# Generate backtest results
print("\n>>> Generating backtest results...")
backtest_trades = []

for i in range(50):
    ticker = np.random.choice(tickers[:10])
    entry_date = start_date + timedelta(days=np.random.randint(0, 60))
    exit_date = entry_date + timedelta(days=np.random.randint(5, 30))
    
    entry_price = np.random.uniform(500, 5000)
    
    # Win/loss
    if np.random.random() < 0.35:  # 35% win rate
        pnl_pct = np.random.uniform(2, 10)
    else:
        pnl_pct = np.random.uniform(-5, -1)
    
    exit_price = entry_price * (1 + pnl_pct / 100)
    shares = 100
    pnl = (exit_price - entry_price) * shares
    
    exit_reasons = ['TAKE_PROFIT', 'STOP_LOSS', 'TRAILING_STOP', 'TIME_EXIT']
    
    backtest_trades.append({
        'ticker': ticker,
        'entry_date': entry_date.strftime('%Y-%m-%d'),
        'exit_date': exit_date.strftime('%Y-%m-%d'),
        'entry_price': entry_price,
        'exit_price': exit_price,
        'shares': shares,
        'pnl': pnl,
        'pnl_pct': pnl_pct,
        'exit_reason': np.random.choice(exit_reasons)
    })

backtest_df = pd.DataFrame(backtest_trades)
Path('data/results').mkdir(parents=True, exist_ok=True)
backtest_file = f'data/results/backtest_trades_{timestamp}.csv'
backtest_df.to_csv(backtest_file, index=False)
print(f"✓ Created: {backtest_file}")
print(f"  Trades: {len(backtest_df)}")

# Generate backtest equity curve image
print("\n>>> Generating equity curve...")
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

initial_capital = 100000
equity_curve = [initial_capital]

for pnl in backtest_df['pnl'].values:
    equity_curve.append(equity_curve[-1] + pnl)

fig, ax = plt.subplots(figsize=(12, 6), facecolor='#000000')
ax.set_facecolor('#000000')

ax.plot(equity_curve, color='#00FF00', linewidth=2, label='Portfolio Value')
ax.axhline(y=initial_capital, color='#808080', linestyle='--', linewidth=1, label='Initial Capital')
ax.fill_between(range(len(equity_curve)), equity_curve, initial_capital,
                where=(np.array(equity_curve) >= initial_capital), alpha=0.3, color='#00FF00')
ax.fill_between(range(len(equity_curve)), equity_curve, initial_capital,
                where=(np.array(equity_curve) < initial_capital), alpha=0.3, color='#D71921')

ax.set_xlabel('Trade Number', color='#FFFFFF', fontsize=12)
ax.set_ylabel('Portfolio Value (₹)', color='#FFFFFF', fontsize=12)
ax.set_title('EQUITY CURVE', color='#D71921', fontsize=16, fontweight='bold', pad=20)
ax.tick_params(colors='#FFFFFF')
ax.spines['bottom'].set_color('#808080')
ax.spines['top'].set_color('#808080')
ax.spines['left'].set_color('#808080')
ax.spines['right'].set_color('#808080')
ax.grid(True, alpha=0.2, color='#808080')
ax.legend(facecolor='#1A1A1A', edgecolor='#808080', labelcolor='#FFFFFF')

plt.tight_layout()
equity_image = f'data/results/backtest_{timestamp}.png'
plt.savefig(equity_image, dpi=150, bbox_inches='tight', facecolor='#000000')
plt.close()
print(f"✓ Created: {equity_image}")

print("\n" + "=" * 60)
print("✓ DEMO DATA GENERATION COMPLETE!")
print("=" * 60)
print("\nGenerated files:")
print(f"  • {market_file}")
print(f"  • {sentiment_file}")
print(f"  • {predictions_file}")
print(f"  • {backtest_file}")
print(f"  • {equity_image}")
print("\n✓ Dashboard is ready to run!")
print("  Run: streamlit run dashboard.py")
print("=" * 60)
