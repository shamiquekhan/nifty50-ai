"""
Real-Time Alerts System
Monitor market conditions and generate trading alerts.
"""

import pandas as pd
import yaml
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict


class AlertEngine:
    """
    Generate trading alerts based on signals and market conditions.
    """
    
    def __init__(self):
        """Initialize alert engine."""
        with open('config/config.yaml', 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.alerts = []
    
    def check_price_alerts(self, market_data: pd.DataFrame) -> List[Dict]:
        """
        Check for price-based alerts.
        
        Args:
            market_data: Latest market data
            
        Returns:
            List of alert dictionaries
        """
        alerts = []
        
        for ticker in market_data['Ticker'].unique():
            ticker_data = market_data[market_data['Ticker'] == ticker].copy()
            latest = ticker_data.iloc[-1]
            
            # RSI Overbought/Oversold
            if 'RSI_14' in latest:
                rsi = latest['RSI_14']
                if rsi > 70:
                    alerts.append({
                        'ticker': ticker,
                        'type': 'RSI_OVERBOUGHT',
                        'severity': 'WARNING',
                        'message': f"{ticker}: RSI at {rsi:.1f} (Overbought)",
                        'timestamp': datetime.now()
                    })
                elif rsi < 30:
                    alerts.append({
                        'ticker': ticker,
                        'type': 'RSI_OVERSOLD',
                        'severity': 'OPPORTUNITY',
                        'message': f"{ticker}: RSI at {rsi:.1f} (Oversold)",
                        'timestamp': datetime.now()
                    })
            
            # Bollinger Band Breakout
            if all(col in latest for col in ['Close', 'BBU_20_2.0_2.0', 'BBL_20_2.0_2.0']):
                if latest['Close'] > latest['BBU_20_2.0_2.0']:
                    alerts.append({
                        'ticker': ticker,
                        'type': 'BB_BREAKOUT_UPPER',
                        'severity': 'WARNING',
                        'message': f"{ticker}: Price above upper Bollinger Band",
                        'timestamp': datetime.now()
                    })
                elif latest['Close'] < latest['BBL_20_2.0_2.0']:
                    alerts.append({
                        'ticker': ticker,
                        'type': 'BB_BREAKOUT_LOWER',
                        'severity': 'OPPORTUNITY',
                        'message': f"{ticker}: Price below lower Bollinger Band",
                        'timestamp': datetime.now()
                    })
            
            # MACD Crossover
            if len(ticker_data) >= 2 and all(col in latest for col in ['MACD_12_26_9', 'MACDs_12_26_9']):
                prev = ticker_data.iloc[-2]
                
                # Bullish crossover
                if prev['MACD_12_26_9'] <= prev['MACDs_12_26_9'] and latest['MACD_12_26_9'] > latest['MACDs_12_26_9']:
                    alerts.append({
                        'ticker': ticker,
                        'type': 'MACD_BULLISH_CROSS',
                        'severity': 'BUY_SIGNAL',
                        'message': f"{ticker}: MACD bullish crossover",
                        'timestamp': datetime.now()
                    })
                
                # Bearish crossover
                elif prev['MACD_12_26_9'] >= prev['MACDs_12_26_9'] and latest['MACD_12_26_9'] < latest['MACDs_12_26_9']:
                    alerts.append({
                        'ticker': ticker,
                        'type': 'MACD_BEARISH_CROSS',
                        'severity': 'SELL_SIGNAL',
                        'message': f"{ticker}: MACD bearish crossover",
                        'timestamp': datetime.now()
                    })
            
            # Volume Spike
            if 'Volume' in latest and len(ticker_data) >= 20:
                avg_volume = ticker_data['Volume'].tail(20).mean()
                if latest['Volume'] > avg_volume * 2:
                    alerts.append({
                        'ticker': ticker,
                        'type': 'VOLUME_SPIKE',
                        'severity': 'INFO',
                        'message': f"{ticker}: Volume spike ({latest['Volume']/avg_volume:.1f}x average)",
                        'timestamp': datetime.now()
                    })
        
        return alerts
    
    def check_prediction_alerts(self, predictions: pd.DataFrame) -> List[Dict]:
        """
        Check for prediction-based alerts.
        
        Args:
            predictions: Model predictions
            
        Returns:
            List of alert dictionaries
        """
        alerts = []
        
        for _, pred in predictions.iterrows():
            # Strong buy signals
            if pred['signal'] == 1 and pred['model_probability'] > 0.75:
                alerts.append({
                    'ticker': pred['ticker'],
                    'type': 'STRONG_BUY',
                    'severity': 'BUY_SIGNAL',
                    'message': f"{pred['ticker']}: Strong BUY signal ({pred['model_probability']:.1%} confidence)",
                    'timestamp': datetime.now()
                })
            
            # Aligned signals (tech + sentiment)
            if pred['signal'] == 1 and pred.get('is_aligned', False):
                alerts.append({
                    'ticker': pred['ticker'],
                    'type': 'ALIGNED_BUY',
                    'severity': 'BUY_SIGNAL',
                    'message': f"{pred['ticker']}: Technical & Sentiment ALIGNED",
                    'timestamp': datetime.now()
                })
        
        return alerts
    
    def get_all_alerts(self, market_data: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
        """
        Get all current alerts.
        
        Args:
            market_data: Latest market data
            predictions: Model predictions
            
        Returns:
            DataFrame of all alerts
        """
        all_alerts = []
        
        # Get price-based alerts
        all_alerts.extend(self.check_price_alerts(market_data))
        
        # Get prediction alerts
        if not predictions.empty:
            all_alerts.extend(self.check_prediction_alerts(predictions))
        
        if all_alerts:
            alerts_df = pd.DataFrame(all_alerts)
            alerts_df = alerts_df.sort_values('severity', ascending=False)
            return alerts_df
        
        return pd.DataFrame()
    
    def save_alerts(self, alerts_df: pd.DataFrame):
        """Save alerts to file."""
        if not alerts_df.empty:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"data/processed/alerts_{timestamp}.csv"
            alerts_df.to_csv(output_file, index=False)
            print(f"💾 Saved {len(alerts_df)} alerts to: {output_file}")


def main():
    """Generate and display current alerts."""
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
    
    # Generate alerts
    engine = AlertEngine()
    alerts_df = engine.get_all_alerts(market_df, predictions_df)
    
    if not alerts_df.empty:
        print("\n🚨 ACTIVE ALERTS")
        print("=" * 80)
        for _, alert in alerts_df.iterrows():
            severity_icon = {
                'BUY_SIGNAL': '🟢',
                'SELL_SIGNAL': '🔴',
                'WARNING': '🟡',
                'OPPORTUNITY': '🟢',
                'INFO': '⚪'
            }.get(alert['severity'], '⚪')
            
            print(f"{severity_icon} [{alert['severity']}] {alert['message']}")
        
        # Save alerts
        engine.save_alerts(alerts_df)
    else:
        print("✅ No alerts at this time")


if __name__ == "__main__":
    main()
