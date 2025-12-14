"""
Quick Model Training
Simplified training script with error handling.
"""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['PYTHONIOENCODING'] = 'utf-8'

import sys
from pathlib import Path
import yaml
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

print("\n>>> Quick Model Training")
print("=" * 60)

try:
    # Load config
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Find data files
    print("\n>>> Loading data...")
    market_files = list(Path('data/raw').glob('market_data_*.csv'))
    sentiment_files = list(Path('data/processed').glob('daily_sentiment_*.csv'))
    
    if not market_files or not sentiment_files:
        print("ERROR: Missing data files!")
        sys.exit(1)
    
    latest_market = max(market_files, key=lambda x: x.stat().st_ctime)
    latest_sentiment = max(sentiment_files, key=lambda x: x.stat().st_ctime)
    
    print(f"SUCCESS: Market: {latest_market.name}")
    print(f"SUCCESS: Sentiment: {latest_sentiment.name}")
    
    # Load data
    market_df = pd.read_csv(latest_market, parse_dates=True)
    sentiment_df = pd.read_csv(latest_sentiment, parse_dates=['date'])
    
    print(f"\n>>> Data loaded:")
    print(f"   Market records: {len(market_df)}")
    print(f"   Sentiment records: {len(sentiment_df)}")
    
    # Simple feature preparation
    print("\n>>> Preparing features...")
    
    # Get numeric columns for price features
    price_features = ['Close', 'Volume', 'RSI_14', 'ATRr_14']
    available_features = [f for f in price_features if f in market_df.columns]
    
    if len(available_features) < 2:
        print("ERROR: Insufficient features in market data!")
        sys.exit(1)
    
    print(f"SUCCESS: Using {len(available_features)} price features")
    
    # Create sequences for training
    sequence_length = 10
    X_sequences = []
    y_labels = []
    
    for ticker in market_df['Ticker'].unique():
        ticker_data = market_df[market_df['Ticker'] == ticker].copy()
        ticker_data = ticker_data.sort_index()
        
        # Normalize features
        for feature in available_features:
            if feature in ticker_data.columns:
                mean = ticker_data[feature].mean()
                std = ticker_data[feature].std()
                if std > 0:
                    ticker_data[feature] = (ticker_data[feature] - mean) / std
        
        # Create sequences
        for i in range(len(ticker_data) - sequence_length):
            sequence = ticker_data[available_features].iloc[i:i+sequence_length].values
            label = ticker_data['Target'].iloc[i+sequence_length] if 'Target' in ticker_data.columns else 0
            
            X_sequences.append(sequence)
            y_labels.append(label)
    
    X = np.array(X_sequences)
    y = np.array(y_labels)
    
    print(f"SUCCESS: Created {len(X)} training sequences")
    print(f"   Shape: {X.shape}")
    
    # Split data
    train_size = int(0.7 * len(X))
    val_size = int(0.15 * len(X))
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size+val_size]
    y_val = y[train_size:train_size+val_size]
    X_test = X[train_size+val_size:]
    y_test = y[train_size+val_size:]
    
    print(f"\n>>> Data split:")
    print(f"   Train: {len(X_train)}")
    print(f"   Val: {len(X_val)}")
    print(f"   Test: {len(X_test)}")
    
    # Build simple model
    print("\n>>> Building model...")
    
    import tensorflow as tf
    from tensorflow import keras
    
    model = keras.Sequential([
        keras.layers.LSTM(64, input_shape=(sequence_length, len(available_features)), return_sequences=True),
        keras.layers.Dropout(0.2),
        keras.layers.LSTM(32),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    print("SUCCESS: Model built successfully")
    print(f"\n>>> Model Summary:")
    model.summary()
    
    # Train model
    print("\n>>> Training model...")
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=10,
        batch_size=32,
        verbose=1,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.5)
        ]
    )
    
    # Evaluate
    print("\n>>> Evaluating model...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"\nSUCCESS: Training Complete!")
    print(f"   Test Loss: {test_loss:.4f}")
    print(f"   Test Accuracy: {test_acc:.4f}")
    
    # Save model
    model_path = 'models/lstm_model.keras'
    Path('models').mkdir(exist_ok=True)
    model.save(model_path)
    
    print(f"\n>>> Model saved to: {model_path}")
    
    # Save training history
    history_df = pd.DataFrame(history.history)
    history_df.to_csv('models/training_history.csv', index=False)
    print(f">>> History saved to: models/training_history.csv")
    
    print("\n" + "=" * 60)
    print("SUCCESS! Model training completed without errors")
    print("=" * 60)

except Exception as e:
    print(f"\nERROR during training: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
