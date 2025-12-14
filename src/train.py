"""
Complete Training Pipeline
End-to-end training workflow for the Dual-Input LSTM model.
"""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
from pathlib import Path
import yaml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.preprocessing import DataPreprocessor
from models.dual_lstm import DualLSTMModel
from agents.kelly_agent import KellyCriterionAgent


def plot_training_history(history, save_path: str = None):
    """
    Plot training history.
    
    Args:
        history: Keras training history
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Loss
    axes[0].plot(history.history['loss'], label='Train Loss')
    axes[0].plot(history.history['val_loss'], label='Val Loss')
    axes[0].set_title('Model Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy
    axes[1].plot(history.history['accuracy'], label='Train Accuracy')
    axes[1].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[1].set_title('Model Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    # AUC
    axes[2].plot(history.history['auc'], label='Train AUC')
    axes[2].plot(history.history['val_auc'], label='Val AUC')
    axes[2].set_title('Model AUC')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('AUC')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Saved training plot to: {save_path}")
    
    plt.show()


def evaluate_with_kelly_agent(
    model: DualLSTMModel,
    X_price: np.ndarray,
    X_sentiment: np.ndarray,
    y_true: np.ndarray,
    tickers: list = None,
    capital: float = 100000
):
    """
    Evaluate model with Kelly Criterion agent.
    
    Args:
        model: Trained DualLSTMModel
        X_price: Price sequences
        X_sentiment: Sentiment features
        y_true: True labels
        tickers: List of ticker names
        capital: Available capital
    """
    print("\n🤖 Evaluating with Kelly Criterion Agent...")
    print("=" * 60)
    
    # Get predictions
    predictions = model.predict(X_price, X_sentiment)
    
    # Initialize agent
    agent = KellyCriterionAgent()
    
    # Sample evaluation on first 10 predictions
    n_samples = min(10, len(predictions))
    
    total_recommended = 0
    total_wait = 0
    
    for i in range(n_samples):
        ticker = tickers[i] if tickers and i < len(tickers) else f"STOCK_{i}"
        
        # Get recommendation
        rec = agent.get_recommendation(
            model_prob=predictions[i][0],
            sentiment_score=X_sentiment[i][0],  # Use first sentiment feature
            capital=capital,
            ticker=ticker,
            verbose=False
        )
        
        if rec['action'] != 'WAIT':
            total_recommended += 1
        else:
            total_wait += 1
    
    print(f"\n📊 Agent Decisions (on {n_samples} samples):")
    print(f"   ✅ Recommended to trade: {total_recommended}")
    print(f"   ⏸️  Recommended to wait: {total_wait}")


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("🚀 NIFTY50 DUAL-LSTM ENSEMBLE TRAINING PIPELINE")
    print("=" * 60)
    
    # ===== STEP 1: Find Data Files =====
    print("\n📂 Step 1: Loading data files...")
    
    raw_path = Path('data/raw')
    processed_path = Path('data/processed')
    
    market_files = list(raw_path.glob('market_data_*.csv'))
    sentiment_files = list(processed_path.glob('daily_sentiment_*.csv'))
    
    if not market_files:
        print("❌ No market data found!")
        print("💡 Run: python src/data_collection/market_data.py")
        sys.exit(1)
    
    if not sentiment_files:
        print("❌ No sentiment data found!")
        print("💡 Run: python src/sentiment/finbert_engine.py")
        sys.exit(1)
    
    # Get latest files
    latest_market = max(market_files, key=os.path.getctime)
    latest_sentiment = max(sentiment_files, key=os.path.getctime)
    
    print(f"✅ Market data: {latest_market.name}")
    print(f"✅ Sentiment data: {latest_sentiment.name}")
    
    # ===== STEP 2: Preprocess Data =====
    print("\n🔄 Step 2: Preprocessing data...")
    
    preprocessor = DataPreprocessor()
    
    train_data, val_data, test_data, price_features, sentiment_features = preprocessor.prepare_model_data(
        str(latest_market),
        str(latest_sentiment)
    )
    
    X_price_train, X_sent_train, y_train = train_data
    X_price_val, X_sent_val, y_val = val_data
    X_price_test, X_sent_test, y_test = test_data
    
    if X_price_val is None:
        print("⚠️  No validation data - using 20% of training for validation")
        # Split training data
        split_idx = int(0.8 * len(X_price_train))
        
        X_price_val = X_price_train[split_idx:]
        X_sent_val = X_sent_train[split_idx:]
        y_val = y_train[split_idx:]
        
        X_price_train = X_price_train[:split_idx]
        X_sent_train = X_sent_train[:split_idx]
        y_train = y_train[:split_idx]
    
    # ===== STEP 3: Build Model =====
    print("\n🏗️  Step 3: Building model...")
    
    model = DualLSTMModel()
    
    price_shape = (X_price_train.shape[1], X_price_train.shape[2])
    sentiment_shape = (X_sent_train.shape[1],)
    
    model.build_model(price_shape, sentiment_shape)
    
    # ===== STEP 4: Train Model =====
    print("\n🎓 Step 4: Training model...")
    
    history = model.train(
        X_price_train, X_sent_train, y_train,
        X_price_val, X_sent_val, y_val,
        model_name='best_model.keras'
    )
    
    # Plot training history
    plot_path = Path('data/models/training_history.png')
    plot_training_history(history, str(plot_path))
    
    # ===== STEP 5: Evaluate Model =====
    print("\n📊 Step 5: Evaluating model on validation set...")
    
    val_metrics = model.evaluate(X_price_val, X_sent_val, y_val)
    
    # ===== STEP 6: Test with Kelly Agent =====
    if X_price_test is not None:
        print("\n🧪 Step 6: Testing with Kelly Criterion Agent...")
        evaluate_with_kelly_agent(
            model, X_price_test, X_sent_test, y_test
        )
    
    # ===== STEP 7: Save Model =====
    print("\n💾 Step 7: Saving final model...")
    model.save_model(str(Path('data/models/dual_lstm_final.keras')))
    
    print("\n" + "=" * 60)
    print("✅ TRAINING PIPELINE COMPLETE!")
    print("=" * 60)
    print("\n📋 Summary:")
    print(f"   Training samples: {len(X_price_train)}")
    print(f"   Validation samples: {len(X_price_val)}")
    print(f"   Test samples: {len(X_price_test) if X_price_test is not None else 0}")
    print(f"   Final Val Accuracy: {val_metrics['accuracy']:.2%}")
    print(f"   Final Val AUC: {val_metrics['auc']:.4f}")
    print(f"\n💡 Next Steps:")
    print(f"   1. Check training plot: data/models/training_history.png")
    print(f"   2. Load model: data/models/best_model.keras")
    print(f"   3. Run inference on new data")
    print(f"   4. (Optional) Create Streamlit dashboard")


if __name__ == "__main__":
    main()
