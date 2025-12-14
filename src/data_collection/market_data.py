"""
Market Data Collection Module
Downloads NIFTY50 stock data using yfinance and adds technical indicators using pandas-ta.
"""

import yfinance as yf
import pandas as pd
import pandas_ta as ta
import yaml
import os
from datetime import datetime
from pathlib import Path


class MarketDataCollector:
    """Collects and processes market data with technical indicators."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.tickers = self.config['market']['tickers']
        self.period = self.config['market']['period']
        self.interval = self.config['market']['interval']
        self.raw_data_path = Path(self.config['paths']['raw_data'])
        
        # Ensure data directory exists
        self.raw_data_path.mkdir(parents=True, exist_ok=True)
    
    def download_stock_data(self, ticker: str) -> pd.DataFrame:
        """
        Download market data for a single ticker.
        
        Args:
            ticker: Stock ticker symbol (e.g., 'RELIANCE.NS')
            
        Returns:
            DataFrame with OHLCV data
        """
        print(f"📥 Downloading data for {ticker}...")
        
        try:
            df = yf.download(ticker, period=self.period, interval=self.interval, progress=False)
            
            if df.empty:
                print(f"⚠️  No data found for {ticker}")
                return None
            
            # Flatten multi-index columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            # Add ticker column
            df['Ticker'] = ticker.replace('.NS', '')
            
            print(f"✅ Downloaded {len(df)} records for {ticker}")
            return df
            
        except Exception as e:
            print(f"❌ Error downloading {ticker}: {str(e)}")
            return None
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators using pandas-ta.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added technical indicators
        """
        if df is None or df.empty:
            return df
        
        print("📊 Calculating technical indicators...")
        
        # Get indicator parameters from config
        ind_config = self.config['market']['indicators']
        
        # RSI (Relative Strength Index)
        df.ta.rsi(length=ind_config['rsi_length'], append=True)
        
        # Bollinger Bands
        df.ta.bbands(length=ind_config['bbands_length'], append=True)
        
        # MACD (Moving Average Convergence Divergence)
        df.ta.macd(fast=ind_config['macd_fast'], 
                   slow=ind_config['macd_slow'], 
                   append=True)
        
        # ATR (Average True Range) - Volatility measure
        df.ta.atr(length=ind_config['atr_length'], append=True)
        
        # SMA (Simple Moving Averages) - 20 and 50 day
        df.ta.sma(length=20, append=True)
        df.ta.sma(length=50, append=True)
        
        # Volume features
        df.ta.obv(append=True)  # On-Balance Volume
        
        print(f"✅ Added {len([col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close', 'Ticker']])} indicators")
        
        return df
    
    def create_target_variable(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create target variable: 1 if price goes up next day, 0 otherwise.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            DataFrame with 'Target' column
        """
        # Calculate next day's return
        df['Next_Close'] = df['Close'].shift(-1)
        df['Return'] = (df['Next_Close'] - df['Close']) / df['Close']
        
        # Binary target: 1 if price goes up, 0 if down
        df['Target'] = (df['Return'] > 0).astype(int)
        
        # Drop helper columns
        df = df.drop(['Next_Close', 'Return'], axis=1)
        
        return df
    
    def collect_all_stocks(self) -> pd.DataFrame:
        """
        Collect data for all configured tickers.
        
        Returns:
            Combined DataFrame with all stocks
        """
        all_data = []
        
        print(f"\n🚀 Starting data collection for {len(self.tickers)} stocks...")
        print("=" * 60)
        
        for ticker in self.tickers:
            # Download data
            df = self.download_stock_data(ticker)
            
            if df is not None:
                # Add technical indicators
                df = self.add_technical_indicators(df)
                
                # Create target variable
                df = self.create_target_variable(df)
                
                # Remove NaN rows (from indicators and target)
                df = df.dropna()
                
                all_data.append(df)
                
                print(f"✅ Processed {ticker}: {len(df)} clean records")
                print("-" * 60)
        
        # Combine all stocks
        if all_data:
            combined_df = pd.concat(all_data, axis=0)
            combined_df = combined_df.sort_index()
            
            # Save to CSV
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = self.raw_data_path / f"market_data_{timestamp}.csv"
            combined_df.to_csv(filename)
            
            print(f"\n💾 Saved {len(combined_df)} total records to {filename}")
            print(f"📅 Date range: {combined_df.index.min()} to {combined_df.index.max()}")
            print(f"📈 Stocks: {combined_df['Ticker'].nunique()}")
            print(f"🔢 Features: {len(combined_df.columns)}")
            
            return combined_df
        else:
            print("❌ No data collected!")
            return pd.DataFrame()
    
    def get_latest_data(self) -> pd.DataFrame:
        """
        Load the most recent market data file.
        
        Returns:
            DataFrame with market data
        """
        files = list(self.raw_data_path.glob("market_data_*.csv"))
        
        if not files:
            print("⚠️  No market data files found. Run collect_all_stocks() first.")
            return pd.DataFrame()
        
        # Get most recent file
        latest_file = max(files, key=os.path.getctime)
        print(f"📂 Loading {latest_file.name}...")
        
        df = pd.read_csv(latest_file, index_col=0, parse_dates=True)
        return df


def main():
    """Main execution function."""
    collector = MarketDataCollector()
    
    # Collect all stock data
    data = collector.collect_all_stocks()
    
    if not data.empty:
        print("\n" + "=" * 60)
        print("📋 Data Summary:")
        print("=" * 60)
        print(data.info())
        print("\n📊 Sample Data:")
        print(data.head())
        
        print("\n✅ Market data collection complete!")


if __name__ == "__main__":
    main()
