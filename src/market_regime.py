"""
Market Regime Detection
Identify market conditions (bull, bear, sideways) to adjust trading strategy.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple


class MarketRegimeDetector:
    """
    Detect current market regime using multiple indicators.
    """
    
    def __init__(self, lookback_period: int = 50):
        """
        Initialize regime detector.
        
        Args:
            lookback_period: Number of periods to analyze
        """
        self.lookback_period = lookback_period
    
    def detect_trend(self, data: pd.DataFrame) -> str:
        """
        Detect market trend direction.
        
        Args:
            data: DataFrame with price data
            
        Returns:
            'BULL', 'BEAR', or 'SIDEWAYS'
        """
        if len(data) < self.lookback_period:
            return 'UNKNOWN'
        
        recent_data = data.tail(self.lookback_period)
        
        # Calculate trend metrics
        close_prices = recent_data['Close']
        
        # 1. Simple trend: Compare current to past
        price_change = (close_prices.iloc[-1] / close_prices.iloc[0] - 1) * 100
        
        # 2. Moving average slope
        if 'SMA_50' in recent_data.columns:
            sma_slope = (recent_data['SMA_50'].iloc[-1] / recent_data['SMA_50'].iloc[0] - 1) * 100
        else:
            sma_slope = price_change
        
        # 3. ADX for trend strength (if available)
        trend_strength = 'STRONG' if abs(price_change) > 10 else 'WEAK'
        
        # Determine regime
        if price_change > 5 and sma_slope > 3:
            return f'BULL_{trend_strength}'
        elif price_change < -5 and sma_slope < -3:
            return f'BEAR_{trend_strength}'
        else:
            return 'SIDEWAYS'
    
    def detect_volatility_regime(self, data: pd.DataFrame) -> str:
        """
        Detect volatility regime.
        
        Args:
            data: DataFrame with price data
            
        Returns:
            'HIGH_VOL', 'MEDIUM_VOL', or 'LOW_VOL'
        """
        if len(data) < self.lookback_period:
            return 'UNKNOWN'
        
        recent_data = data.tail(self.lookback_period)
        
        # Calculate historical volatility (ATR)
        if 'ATRr_14' in recent_data.columns:
            current_vol = recent_data['ATRr_14'].iloc[-1]
            avg_vol = recent_data['ATRr_14'].mean()
            
            if current_vol > avg_vol * 1.5:
                return 'HIGH_VOL'
            elif current_vol < avg_vol * 0.7:
                return 'LOW_VOL'
            else:
                return 'MEDIUM_VOL'
        
        # Fallback: Use price range
        returns = recent_data['Close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)  # Annualized
        
        if volatility > 0.30:  # 30%
            return 'HIGH_VOL'
        elif volatility < 0.15:  # 15%
            return 'LOW_VOL'
        else:
            return 'MEDIUM_VOL'
    
    def get_regime_adjustments(self, trend: str, volatility: str) -> Dict[str, float]:
        """
        Get strategy adjustments based on regime.
        
        Args:
            trend: Market trend
            volatility: Volatility regime
            
        Returns:
            Dictionary of adjustment factors
        """
        adjustments = {
            'position_size': 1.0,
            'stop_loss': 1.0,
            'take_profit': 1.0,
            'confidence_threshold': 0.60
        }
        
        # Trend-based adjustments
        if 'BULL' in trend:
            adjustments['position_size'] = 1.2  # Larger positions in bull market
            adjustments['take_profit'] = 1.3  # Wider profit targets
            adjustments['confidence_threshold'] = 0.55  # More aggressive
        elif 'BEAR' in trend:
            adjustments['position_size'] = 0.7  # Smaller positions in bear market
            adjustments['stop_loss'] = 0.8  # Tighter stops
            adjustments['confidence_threshold'] = 0.70  # More conservative
        elif trend == 'SIDEWAYS':
            adjustments['position_size'] = 0.8  # Reduced exposure
            adjustments['take_profit'] = 0.8  # Quicker profits
        
        # Volatility-based adjustments
        if volatility == 'HIGH_VOL':
            adjustments['position_size'] *= 0.7  # Reduce size in high vol
            adjustments['stop_loss'] *= 1.5  # Wider stops
            adjustments['take_profit'] *= 1.5  # Wider targets
        elif volatility == 'LOW_VOL':
            adjustments['position_size'] *= 1.2  # Increase size in low vol
            adjustments['stop_loss'] *= 0.8  # Tighter stops
            adjustments['take_profit'] *= 0.8  # Quicker profits
        
        # Cap adjustments
        adjustments['position_size'] = max(0.3, min(1.5, adjustments['position_size']))
        adjustments['stop_loss'] = max(0.5, min(2.0, adjustments['stop_loss']))
        adjustments['take_profit'] = max(0.5, min(2.0, adjustments['take_profit']))
        
        return adjustments
    
    def analyze_regime(self, data: pd.DataFrame) -> Dict:
        """
        Complete regime analysis.
        
        Args:
            data: DataFrame with market data
            
        Returns:
            Dictionary with regime information
        """
        trend = self.detect_trend(data)
        volatility = self.detect_volatility_regime(data)
        adjustments = self.get_regime_adjustments(trend, volatility)
        
        return {
            'trend': trend,
            'volatility': volatility,
            'adjustments': adjustments,
            'recommendation': self._get_recommendation(trend, volatility)
        }
    
    def _get_recommendation(self, trend: str, volatility: str) -> str:
        """Get trading recommendation based on regime."""
        if 'BULL_STRONG' in trend and volatility == 'MEDIUM_VOL':
            return 'AGGRESSIVE_LONG'
        elif 'BULL' in trend and volatility in ['LOW_VOL', 'MEDIUM_VOL']:
            return 'MODERATE_LONG'
        elif 'BEAR_STRONG' in trend:
            return 'AVOID_LONGS'
        elif 'BEAR' in trend and volatility == 'HIGH_VOL':
            return 'WAIT_FOR_STABILITY'
        elif trend == 'SIDEWAYS' and volatility == 'LOW_VOL':
            return 'RANGE_TRADING'
        elif volatility == 'HIGH_VOL':
            return 'REDUCE_EXPOSURE'
        else:
            return 'SELECTIVE_TRADING'


def main():
    """Test regime detection."""
    from pathlib import Path
    
    # Load market data
    market_files = list(Path('data/raw').glob('market_data_*.csv'))
    if not market_files:
        print("❌ No market data found!")
        return
    
    latest_market = max(market_files, key=lambda x: x.stat().st_ctime)
    market_df = pd.read_csv(latest_market, parse_dates=True, index_col=0)
    
    # Analyze each stock
    detector = MarketRegimeDetector(lookback_period=50)
    
    print("\n📊 MARKET REGIME ANALYSIS")
    print("=" * 80)
    
    for ticker in market_df['Ticker'].unique():
        ticker_data = market_df[market_df['Ticker'] == ticker].sort_index()
        regime = detector.analyze_regime(ticker_data)
        
        print(f"\n{ticker}:")
        print(f"  Trend: {regime['trend']}")
        print(f"  Volatility: {regime['volatility']}")
        print(f"  Recommendation: {regime['recommendation']}")
        print(f"  Position Size Adj: {regime['adjustments']['position_size']:.2f}x")
        print(f"  Stop Loss Adj: {regime['adjustments']['stop_loss']:.2f}x")
        print(f"  Take Profit Adj: {regime['adjustments']['take_profit']:.2f}x")
        print(f"  Confidence Threshold: {regime['adjustments']['confidence_threshold']:.0%}")


if __name__ == "__main__":
    main()
