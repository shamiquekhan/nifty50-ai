"""
Incremental Model Fine-Tuning
Fine-tunes existing model with new data without full retraining
"""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import joblib

print("\n" + "=" * 60)
print("INCREMENTAL MODEL FINE-TUNING")
print("=" * 60)


def load_latest_data():
    """Load the most recent market and sentiment data."""
    print("\n>>> Loading latest data...")
    
    # Find latest files
    market_files = list(Path('data/raw').glob('market_data_*.csv'))
    sentiment_files = list(Path('data/processed').glob('daily_sentiment_*.csv'))
    
    if not market_files:
        raise FileNotFoundError("No market data found!")
    
    latest_market = max(market_files, key=lambda x: x.stat().st_ctime)
    market_df = pd.read_csv(latest_market, index_col=0, parse_dates=True)
    
    sentiment_df = pd.DataFrame()
    if sentiment_files:
        latest_sentiment = max(sentiment_files, key=lambda x: x.stat().st_ctime)
        sentiment_df = pd.read_csv(latest_sentiment, parse_dates=['date'])
    
    print(f"SUCCESS: Loaded {len(market_df)} market records")
    if not sentiment_df.empty:
        print(f"SUCCESS: Loaded {len(sentiment_df)} sentiment records")
    
    return market_df, sentiment_df


def prepare_incremental_data(market_df, sentiment_df, lookback=10):
    """Prepare new data for incremental training."""
    print("\n>>> Preparing incremental data...")
    
    # Select features
    feature_cols = ['Close', 'Volume', 'RSI_14', 'ATRr_14']
    
    # Get only new data (last 30 days)
    market_df = market_df.tail(30 * len(market_df['Ticker'].unique()))
    
    # Create sequences
    sequences = []
    labels = []
    
    for ticker in market_df['Ticker'].unique():
        ticker_data = market_df[market_df['Ticker'] == ticker].sort_index()
        
        if len(ticker_data) < lookback + 1:
            continue
        
        features = ticker_data[feature_cols].values
        
        for i in range(len(features) - lookback):
            sequences.append(features[i:i+lookback])
            
            # Label: 1 if price goes up, 0 if down
            future_price = ticker_data.iloc[i + lookback]['Close']
            current_price = ticker_data.iloc[i + lookback - 1]['Close']
            labels.append(1 if future_price > current_price else 0)
    
    X = np.array(sequences)
    y = np.array(labels)
    
    print(f"SUCCESS: Created {len(sequences)} new sequences")
    print(f"   Shape: {X.shape}")
    
    return X, y


def fine_tune_model(X_new, y_new, epochs=5):
    """Fine-tune existing model with new data."""
    print("\n>>> Loading existing model...")
    
    model_path = Path('models/lstm_model.keras')
    
    if not model_path.exists():
        print("WARNING: No existing model found. Run full training first.")
        return False
    
    # Load model
    model = keras.models.load_model(model_path)
    print("SUCCESS: Model loaded")
    
    print("\n>>> Fine-tuning model...")
    print(f"   Training on {len(X_new)} new samples")
    print(f"   Epochs: {epochs}")
    
    # Lower learning rate for fine-tuning
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),  # Lower LR
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Split data
    split = int(0.8 * len(X_new))
    X_train, X_val = X_new[:split], X_new[split:]
    y_train, y_val = y_new[:split], y_new[split:]
    
    # Fine-tune with early stopping
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=2,
        restore_best_weights=True
    )
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )
    
    # Evaluate
    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
    
    print(f"\nSUCCESS: Fine-tuning complete!")
    print(f"   Validation Accuracy: {val_accuracy:.4f}")
    print(f"   Validation Loss: {val_loss:.4f}")
    
    # Save updated model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"models/lstm_model_backup_{timestamp}.keras"
    
    # Backup old model
    import shutil
    shutil.copy(model_path, backup_path)
    print(f"\n>>> Backup saved: {backup_path}")
    
    # Save fine-tuned model
    model.save(model_path)
    print(f">>> Updated model saved: {model_path}")
    
    # Save training history
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(f'models/finetune_history_{timestamp}.csv', index=False)
    
    return True


def main():
    """Run incremental fine-tuning."""
    try:
        # Load latest data
        market_df, sentiment_df = load_latest_data()
        
        # Prepare new sequences
        X_new, y_new = prepare_incremental_data(market_df, sentiment_df)
        
        # Fine-tune model
        success = fine_tune_model(X_new, y_new, epochs=5)
        
        if success:
            print("\n" + "=" * 60)
            print("SUCCESS! Model fine-tuned with latest data")
            print("=" * 60)
        
    except Exception as e:
        print(f"\nERROR: Fine-tuning failed - {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
