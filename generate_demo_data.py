"""
Generate demo data for dashboard testing
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Create data directories
Path('data/raw').mkdir(parents=True, exist_ok=True)
Path('data/processed').mkdir(parents=True, exist_ok=True)

# Generate sample market data
def generate_demo_market_data():
    """Generate sample market data for 3 stocks over 100 days"""
    
    tickers = ['RELIANCE', 'TCS', 'HDFCBANK']
    dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
    
    all_data = []
    
    for ticker in tickers:
        # Base price
        base_price = np.random.uniform(1000, 3000)
        
        # Generate price series with random walk
        returns = np.random.normal(0.001, 0.02, len(dates))
        prices = base_price * np.exp(np.cumsum(returns))
        
        # Generate OHLCV
        df = pd.DataFrame({
            'Open': prices * np.random.uniform(0.98, 1.02, len(dates)),
            'High': prices * np.random.uniform(1.00, 1.05, len(dates)),
            'Low': prices * np.random.uniform(0.95, 1.00, len(dates)),
            'Close': prices,
            'Volume': np.random.randint(1000000, 10000000, len(dates)),
            'Ticker': ticker
        }, index=dates)
        
        # Add technical indicators
        df['RSI_14'] = 50 + np.random.randn(len(dates)) * 15
        df['RSI_14'] = df['RSI_14'].clip(0, 100)
        
        df['BBU_20_2.0'] = df['Close'] * 1.05
        df['BBL_20_2.0'] = df['Close'] * 0.95
        
        df['MACD_12_26_9'] = np.random.randn(len(dates)) * 10
        df['MACDs_12_26_9'] = df['MACD_12_26_9'].rolling(9).mean()
        
        df['ATRr_14'] = np.random.uniform(0.01, 0.05, len(dates))
        
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        
        all_data.append(df)
    
    combined = pd.concat(all_data)
    combined = combined.dropna()
    
    # Save
    filename = f"data/raw/market_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    combined.to_csv(filename)
    print(f"✅ Created {filename}")
    print(f"   {len(combined)} records for {len(tickers)} stocks")
    
    return combined

# Generate sample sentiment data
def generate_demo_sentiment_data():
    """Generate sample sentiment data"""
    
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    
    df = pd.DataFrame({
        'date': dates,
        'sentiment_mean': np.random.uniform(-0.5, 0.5, len(dates)),
        'sentiment_ma_5': np.random.uniform(-0.3, 0.3, len(dates)),
        'article_count': np.random.randint(5, 20, len(dates)),
        'dominant_sentiment': np.random.choice(['Positive', 'Negative', 'Neutral'], len(dates))
    })
    
    # Save
    filename = f"data/processed/daily_sentiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(filename, index=False)
    print(f"✅ Created {filename}")
    print(f"   {len(df)} days of sentiment data")
    
    return df

# Generate sample predictions
def generate_demo_predictions():
    """Generate sample prediction data"""
    
    tickers = ['RELIANCE', 'TCS', 'HDFCBANK']
    
    predictions = []
    for ticker in tickers:
        prob = np.random.uniform(0.5, 0.9)
        action = "BUY" if prob > 0.6 else "WAIT"
        
        predictions.append({
            'ticker': ticker,
            'model_probability': prob,
            'action': action,
            'kelly_fraction': prob * 0.5 if action == "BUY" else 0,
            'position_size': 100000 * prob * 0.5 if action == "BUY" else 0,
            'is_aligned': np.random.choice([True, False])
        })
    
    df = pd.DataFrame(predictions)
    
    # Save
    filename = "data/processed/predictions.csv"
    df.to_csv(filename, index=False)
    print(f"✅ Created {filename}")
    print(f"   {len(df)} predictions")
    
    return df

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  GENERATING DEMO DATA FOR DASHBOARD")
    print("="*60 + "\n")
    
    generate_demo_market_data()
    generate_demo_sentiment_data()
    generate_demo_predictions()
    
    print("\n" + "="*60)
    print("  ✅ DEMO DATA GENERATION COMPLETE!")
    print("="*60)
    print("\nYou can now run: streamlit run dashboard.py\n")
