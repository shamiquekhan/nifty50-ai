"""
Inference Script
Make predictions on new data using trained model.
"""

import sys
import os
from pathlib import Path
import yaml
import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from utils.preprocessing import DataPreprocessor
from models.dual_lstm import DualLSTMModel
from agents.kelly_agent import KellyCriterionAgent


def predict_single_stock(
    model: DualLSTMModel,
    preprocessor: DataPreprocessor,
    stock_data: pd.DataFrame,
    sentiment_score: float,
    ticker: str = "STOCK"
) -> dict:
    """
    Make prediction for a single stock.
    
    Args:
        model: Trained model
        preprocessor: Data preprocessor with fitted scalers
        stock_data: Recent stock data (must have at least lookback_days rows)
        sentiment_score: Current sentiment score
        ticker: Stock ticker
        
    Returns:
        Prediction dictionary
    """
    # Load artifacts to get feature names
    artifacts = preprocessor.load_artifacts()
    price_features = artifacts['price_features']
    sentiment_features = artifacts['sentiment_features']
    lookback_days = artifacts['lookback_days']
    
    # Prepare price data
    if len(stock_data) < lookback_days:
        raise ValueError(f"Need at least {lookback_days} days of data")
    
    # Get last N days
    recent_data = stock_data.tail(lookback_days)[price_features].copy()
    
    # Normalize
    X_price = preprocessor.price_scaler.transform(recent_data)
    X_price = X_price.reshape(1, lookback_days, len(price_features))
    
    # Prepare sentiment
    sentiment_data = np.array([[sentiment_score, sentiment_score]])  # Use same for both features
    X_sentiment = preprocessor.sentiment_scaler.transform(sentiment_data)
    
    # Predict
    prediction_prob = model.predict(X_price, X_sentiment)[0][0]
    
    # Get Kelly recommendation
    agent = KellyCriterionAgent()
    recommendation = agent.get_recommendation(
        model_prob=prediction_prob,
        sentiment_score=sentiment_score,
        capital=100000,  # Example capital
        ticker=ticker,
        verbose=True
    )
    
    return recommendation


def batch_predict(
    model_path: str = 'data/models/best_model.keras',
    market_file: str | None = None,
    sentiment_file: str | None = None
):
    """
    Make predictions on batch of stocks.
    
    Args:
        model_path: Path to trained model
        market_file: Path to market data
        sentiment_file: Path to sentiment data
    """
    print("\n🔮 Running batch predictions...")
    print("=" * 60)
    
    # Load model
    model = DualLSTMModel()
    model.load_model(model_path)
    
    # Load preprocessor
    preprocessor = DataPreprocessor()
    artifacts = preprocessor.load_artifacts()
    
    # Find latest data if not provided
    if market_file is None:
        market_files = list(Path('data/raw').glob('market_data_*.csv'))
        market_file = str(max(market_files, key=os.path.getctime))
    
    if sentiment_file is None:
        sentiment_files = list(Path('data/processed').glob('daily_sentiment_*.csv'))
        sentiment_file = str(max(sentiment_files, key=os.path.getctime))
    
    # Load data
    market_df = pd.read_csv(market_file, index_col=0, parse_dates=True)
    sentiment_df = pd.read_csv(sentiment_file, parse_dates=['date'])
    
    # Get unique tickers
    tickers = market_df['Ticker'].unique()
    
    print(f"\n📊 Making predictions for {len(tickers)} stocks...")
    
    results = []
    
    for ticker in tickers[:5]:  # Limit to first 5 for demo
        try:
            # Get stock data
            stock_data = market_df[market_df['Ticker'] == ticker].copy()
            
            # Get latest sentiment
            latest_date = stock_data.index.max().date()
            sentiment_row = sentiment_df[pd.to_datetime(sentiment_df['date']).dt.date == latest_date]
            
            if len(sentiment_row) > 0:
                sentiment_score = sentiment_row['sentiment_mean'].values[0]
            else:
                sentiment_score = 0.0
            
            # Make prediction
            rec = predict_single_stock(
                model, preprocessor, stock_data, sentiment_score, ticker
            )
            
            results.append(rec)
            
        except Exception as e:
            print(f"⚠️  Error predicting {ticker}: {str(e)}")
    
    # Display results
    print("\n" + "=" * 60)
    print("📊 BATCH PREDICTION RESULTS")
    print("=" * 60)
    
    results_df = pd.DataFrame(results)
    print(results_df[['ticker', 'action', 'position_size', 'model_probability', 'sentiment_score']])
    
    # Save results
    output_file = Path('data/processed/predictions.csv')
    results_df.to_csv(output_file, index=False)
    print(f"\n💾 Saved predictions to: {output_file}")


def main():
    """Main inference pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Make predictions with trained model')
    parser.add_argument('--model', type=str, default='data/models/best_model.keras',
                      help='Path to trained model')
    parser.add_argument('--market', type=str, default=None,
                      help='Path to market data CSV')
    parser.add_argument('--sentiment', type=str, default=None,
                      help='Path to sentiment data CSV')
    
    args = parser.parse_args()
    
    batch_predict(args.model, args.market, args.sentiment)


if __name__ == "__main__":
    main()
