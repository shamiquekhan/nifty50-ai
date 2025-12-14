"""
Live Price Fetcher for NIFTY50 Stocks
Fetches real-time prices from Yahoo Finance (NSE)
"""

import yfinance as yf
import pandas as pd
from datetime import datetime
from pathlib import Path

class LivePriceFetcher:
    """Fetch real-time stock prices from NSE via Yahoo Finance"""
    
    def __init__(self):
        self.nse_suffix = '.NS'  # NSE suffix for Yahoo Finance
        
    def get_live_price(self, ticker: str) -> dict:
        """
        Get live price for a single ticker
        
        Args:
            ticker: Stock ticker (e.g., 'RELIANCE')
            
        Returns:
            dict with price information
        """
        try:
            # Add NSE suffix
            yahoo_ticker = f"{ticker}{self.nse_suffix}"
            
            # Fetch data
            stock = yf.Ticker(yahoo_ticker)
            info = stock.info
            
            # Get current price (try multiple fields)
            current_price = (
                info.get('currentPrice') or 
                info.get('regularMarketPrice') or 
                info.get('previousClose')
            )
            
            if current_price is None:
                # Fallback: get latest close from history
                hist = stock.history(period='1d')
                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]
            
            return {
                'ticker': ticker,
                'price': float(current_price) if current_price else None,
                'open': float(info.get('regularMarketOpen', 0)),
                'high': float(info.get('dayHigh', 0)),
                'low': float(info.get('dayLow', 0)),
                'volume': int(info.get('volume', 0)),
                'previous_close': float(info.get('previousClose', 0)),
                'change': float(info.get('regularMarketChange', 0)),
                'change_percent': float(info.get('regularMarketChangePercent', 0)),
                'timestamp': datetime.now().isoformat(),
                'success': True
            }
            
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")
            return {
                'ticker': ticker,
                'price': None,
                'success': False,
                'error': str(e)
            }
    
    def get_multiple_prices(self, tickers: list) -> pd.DataFrame:
        """
        Get live prices for multiple tickers
        
        Args:
            tickers: List of stock tickers
            
        Returns:
            DataFrame with price information
        """
        results = []
        
        for ticker in tickers:
            print(f"Fetching {ticker}...", end=' ')
            result = self.get_live_price(ticker)
            results.append(result)
            print("✓" if result['success'] else "✗")
        
        return pd.DataFrame(results)
    
    def get_nifty50_prices(self) -> pd.DataFrame:
        """Get live prices for all NIFTY50 stocks"""
        
        nifty50_tickers = [
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
        
        return self.get_multiple_prices(nifty50_tickers)
    
    def save_prices(self, df: pd.DataFrame, filepath: str = None):
        """Save prices to CSV"""
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f'data/live_prices_{timestamp}.csv'
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filepath, index=False)
        print(f"\n✓ Saved to: {filepath}")
        return filepath


def fetch_and_save_live_prices():
    """Main function to fetch and save live prices"""
    print("=" * 60)
    print("FETCHING LIVE NIFTY50 PRICES")
    print("=" * 60)
    
    fetcher = LivePriceFetcher()
    prices_df = fetcher.get_nifty50_prices()
    
    # Save to data directory
    filepath = fetcher.save_prices(prices_df, 'data/live_prices.csv')
    
    # Display summary
    successful = prices_df['success'].sum()
    total = len(prices_df)
    print(f"\n{'='*60}")
    print(f"SUMMARY: {successful}/{total} prices fetched successfully")
    print(f"{'='*60}")
    
    # Show sample prices
    print("\nSample Prices:")
    print(prices_df[prices_df['success']][['ticker', 'price', 'change_percent']].head(10).to_string(index=False))
    
    return prices_df


if __name__ == "__main__":
    fetch_and_save_live_prices()
