"""
Shoonya (Finvasia) API Integration - Unlimited Free Real-Time Data
Industry secret: Zero brokerage broker with unlimited API access
Install: pip install shoonya
"""

import logging
from typing import Dict, List, Optional
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ShoonyaDataFeed:
    """
    Shoonya API wrapper for unlimited free market data.
    
    Setup Instructions:
    1. Open free account at https://shoonya.com
    2. Get API credentials (User ID, Password, Vendor Code, API Key)
    3. Store in environment variables or config file
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Shoonya connection.
        
        Args:
            config: Dictionary with keys: user_id, password, vendor_code, api_key, imei
        """
        self.api = None
        self.config = config or self._load_config()
        self.connected = False
        
    def _load_config(self) -> Dict:
        """Load config from environment or config file."""
        import os
        
        # Try environment variables first
        config = {
            'user_id': os.getenv('SHOONYA_USER_ID', ''),
            'password': os.getenv('SHOONYA_PASSWORD', ''),
            'vendor_code': os.getenv('SHOONYA_VENDOR_CODE', ''),
            'api_key': os.getenv('SHOONYA_API_KEY', ''),
            'imei': os.getenv('SHOONYA_IMEI', 'abc1234')
        }
        
        # Check if config file exists
        config_file = Path('config/shoonya_config.yaml')
        if config_file.exists():
            import yaml
            with open(config_file, 'r') as f:
                file_config = yaml.safe_load(f)
                config.update(file_config)
        
        return config
    
    def connect(self) -> bool:
        """
        Connect to Shoonya API.
        
        Returns:
            True if connected successfully
        """
        try:
            from NorenRestApiPy.NorenApi import NorenApi
            
            self.api = NorenApi(
                host='https://api.shoonya.com/NorenWClientTP/',
                websocket='wss://api.shoonya.com/NorenWSTP/'
            )
            
            # Login
            ret = self.api.login(
                userid=self.config['user_id'],
                password=self.config['password'],
                twoFA='',
                vendor_code=self.config['vendor_code'],
                api_secret=self.config['api_key'],
                imei=self.config['imei']
            )
            
            if ret and ret.get('stat') == 'Ok':
                self.connected = True
                logger.info("✓ Connected to Shoonya API")
                return True
            else:
                logger.error(f"✗ Shoonya login failed: {ret}")
                return False
                
        except ImportError:
            logger.warning("⚠ Shoonya library not installed. Run: pip install shoonya")
            logger.warning("⚠ Using fallback data source (yfinance)")
            return False
        except Exception as e:
            logger.error(f"✗ Shoonya connection error: {e}")
            return False
    
    def get_historical_data(
        self,
        symbol: str,
        exchange: str = 'NSE',
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
        interval: str = '1d'
    ) -> pd.DataFrame:
        """
        Get historical OHLCV data (UNLIMITED - No rate limits).
        
        Args:
            symbol: Stock symbol (e.g., 'RELIANCE-EQ')
            exchange: Exchange code (NSE, BSE)
            from_date: Start date
            to_date: End date
            interval: '1m', '5m', '15m', '1d'
            
        Returns:
            DataFrame with OHLCV data
        """
        if not self.connected:
            logger.warning("Not connected. Using fallback...")
            return self._fallback_historical_data(symbol)
        
        try:
            # Format dates
            if from_date is None:
                from_date = datetime.now() - timedelta(days=365)
            if to_date is None:
                to_date = datetime.now()
            
            # Get data
            ret = self.api.get_time_price_series(
                exchange=exchange,
                token=self._get_token(symbol, exchange),
                starttime=from_date.timestamp(),
                endtime=to_date.timestamp(),
                interval=interval
            )
            
            if ret:
                df = pd.DataFrame(ret)
                df['time'] = pd.to_datetime(df['time'], format='%d-%m-%Y %H:%M:%S')
                df = df.rename(columns={
                    'time': 'Date',
                    'into': 'Open',
                    'inth': 'High',
                    'intl': 'Low',
                    'intc': 'Close',
                    'v': 'Volume'
                })
                df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
                df.set_index('Date', inplace=True)
                
                logger.info(f"✓ Fetched {len(df)} candles for {symbol}")
                return df
            
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
        
        return pd.DataFrame()
    
    def _get_token(self, symbol: str, exchange: str) -> str:
        """Get exchange token for symbol."""
        # This would require symbol lookup
        # For simplicity, using direct symbol
        return symbol
    
    def _fallback_historical_data(self, symbol: str) -> pd.DataFrame:
        """Fallback to yfinance if Shoonya not available."""
        try:
            import yfinance as yf
            ticker = f"{symbol}.NS"
            data = yf.download(ticker, period='1y', interval='1d', progress=False)
            logger.info(f"✓ Fetched {len(data)} days via yfinance (fallback)")
            return data
        except Exception as e:
            logger.error(f"Fallback failed: {e}")
            return pd.DataFrame()
    
    def get_quote(self, symbol: str, exchange: str = 'NSE') -> Dict:
        """
        Get live quote for symbol.
        
        Returns:
            Dict with ltp, open, high, low, close, volume
        """
        if not self.connected:
            return {}
        
        try:
            quote = self.api.get_quotes(exchange=exchange, token=symbol)
            return quote
        except Exception as e:
            logger.error(f"Error getting quote: {e}")
            return {}


# Convenience function for scripts
def get_nifty50_data(ticker: str, days: int = 365) -> pd.DataFrame:
    """
    Get NIFTY50 stock data using best available source.
    
    Args:
        ticker: Stock ticker (e.g., 'RELIANCE')
        days: Number of days of history
        
    Returns:
        DataFrame with OHLCV data
    """
    # Try Shoonya first
    shoonya = ShoonyaDataFeed()
    if shoonya.connect():
        from_date = datetime.now() - timedelta(days=days)
        return shoonya.get_historical_data(
            symbol=f"{ticker}-EQ",
            from_date=from_date
        )
    
    # Fallback to yfinance
    logger.info("Using yfinance as primary source")
    try:
        import yfinance as yf
        ticker_symbol = f"{ticker}.NS"
        data = yf.download(ticker_symbol, period=f"{days}d", interval='1d', progress=False)
        return data
    except Exception as e:
        logger.error(f"All data sources failed: {e}")
        return pd.DataFrame()


if __name__ == '__main__':
    # Test the data feed
    print("Testing Shoonya Data Feed...")
    
    # Test with RELIANCE
    df = get_nifty50_data('RELIANCE', days=30)
    
    if not df.empty:
        print(f"\n✓ Successfully fetched {len(df)} days of data")
        print(f"\nLatest data:\n{df.tail()}")
    else:
        print("\n✗ No data retrieved")
