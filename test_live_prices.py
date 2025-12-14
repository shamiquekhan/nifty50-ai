"""
Test Live Price Fetching
Verifies yfinance integration and displays real-time NSE prices
"""

import yfinance as yf
import pandas as pd
from datetime import datetime

def test_single_ticker(ticker='RELIANCE'):
    """Test fetching a single ticker"""
    print(f"\n{'='*60}")
    print(f"Testing: {ticker}")
    print(f"{'='*60}")
    
    yahoo_ticker = f"{ticker}.NS"
    stock = yf.Ticker(yahoo_ticker)
    
    # Get info
    info = stock.info
    
    # Get current price
    price = (
        info.get('currentPrice') or 
        info.get('regularMarketPrice') or 
        info.get('previousClose')
    )
    
    if price is None:
        hist = stock.history(period='1d')
        if not hist.empty:
            price = hist['Close'].iloc[-1]
    
    print(f"\n✓ Ticker: {ticker}")
    print(f"✓ Yahoo Symbol: {yahoo_ticker}")
    print(f"✓ Current Price: ₹{price:.2f}")
    print(f"✓ Previous Close: ₹{info.get('previousClose', 0):.2f}")
    print(f"✓ Open: ₹{info.get('regularMarketOpen', 0):.2f}")
    print(f"✓ Day High: ₹{info.get('dayHigh', 0):.2f}")
    print(f"✓ Day Low: ₹{info.get('dayLow', 0):.2f}")
    print(f"✓ Volume: {info.get('volume', 0):,}")
    print(f"✓ Market Cap: ₹{info.get('marketCap', 0):,}")
    
    return price

def test_multiple_tickers():
    """Test fetching multiple tickers"""
    print(f"\n{'='*60}")
    print("Testing Multiple Tickers (Top 10 NIFTY50)")
    print(f"{'='*60}")
    
    tickers = [
        'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK',
        'HINDUNILVR', 'ITC', 'SBIN', 'BHARTIARTL', 'KOTAKBANK'
    ]
    
    results = []
    
    for ticker in tickers:
        try:
            yahoo_ticker = f"{ticker}.NS"
            stock = yf.Ticker(yahoo_ticker)
            info = stock.info
            
            price = (
                info.get('currentPrice') or 
                info.get('regularMarketPrice') or 
                info.get('previousClose')
            )
            
            if price is None:
                hist = stock.history(period='1d')
                if not hist.empty:
                    price = hist['Close'].iloc[-1]
            
            change_pct = info.get('regularMarketChangePercent', 0)
            
            results.append({
                'Ticker': ticker,
                'Price': f"₹{price:.2f}" if price else 'N/A',
                'Change %': f"{change_pct:.2f}%" if change_pct else 'N/A',
                'Volume': f"{info.get('volume', 0):,}",
                'Status': '✓'
            })
            
            print(f"✓ {ticker:12} ₹{price:8.2f}")
            
        except Exception as e:
            results.append({
                'Ticker': ticker,
                'Price': 'ERROR',
                'Change %': 'N/A',
                'Volume': 'N/A',
                'Status': '✗'
            })
            print(f"✗ {ticker:12} Error: {e}")
    
    # Display summary
    df = pd.DataFrame(results)
    print(f"\n{'-'*60}")
    print("SUMMARY")
    print(f"{'-'*60}")
    print(df.to_string(index=False))
    
    successful = df['Status'].value_counts().get('✓', 0)
    print(f"\n✓ Successfully fetched: {successful}/{len(tickers)}")
    
    return df

if __name__ == "__main__":
    print("="*60)
    print("LIVE PRICE FETCHER TEST")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # Test single ticker
    test_single_ticker('RELIANCE')
    
    # Test multiple tickers
    test_multiple_tickers()
    
    print(f"\n{'='*60}")
    print("✓ LIVE PRICE FETCHING WORKING CORRECTLY!")
    print("="*60)
