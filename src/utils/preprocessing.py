"""
Data Preprocessing and Merging Utilities
Combines market data with sentiment data and prepares for LSTM training.
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List
import pickle


class DataPreprocessor:
    """
    Handles data merging, normalization, and sequence preparation for LSTM.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.processed_data_path = Path(self.config['paths']['processed_data'])
        self.lookback_days = self.config['model']['lstm']['lookback_days']
        
        self.price_scaler = MinMaxScaler()
        self.sentiment_scaler = MinMaxScaler()
        
    def merge_market_and_sentiment(
        self,
        market_df: pd.DataFrame,
        sentiment_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merge market data with daily sentiment scores.
        
        Args:
            market_df: DataFrame with market data and technical indicators
            sentiment_df: DataFrame with daily sentiment scores
            
        Returns:
            Merged DataFrame
        """
        print("\n🔗 Merging market data with sentiment...")
        print("=" * 60)
        
        # Ensure date columns are datetime
        if 'date' in sentiment_df.columns:
            sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
            sentiment_df = sentiment_df.set_index('date')
        
        # Market data should have date as index
        if not isinstance(market_df.index, pd.DatetimeIndex):
            market_df.index = pd.to_datetime(market_df.index)
        
        # Extract just the date part (ignore time)
        market_df['date_only'] = pd.to_datetime(market_df.index).date
        sentiment_df['date_only'] = pd.to_datetime(sentiment_df.index).date
        
        # Merge on date
        merged_df = market_df.merge(
            sentiment_df[['date_only', 'sentiment_mean', f'sentiment_ma_{self.config["sentiment"]["moving_average_days"]}']],
            on='date_only',
            how='left'
        )
        
        # Fill missing sentiment with 0 (neutral)
        merged_df['sentiment_mean'] = merged_df['sentiment_mean'].fillna(0)
        merged_df[f'sentiment_ma_{self.config["sentiment"]["moving_average_days"]}'] = \
            merged_df[f'sentiment_ma_{self.config["sentiment"]["moving_average_days"]}'].fillna(0)
        
        # Drop helper column
        merged_df = merged_df.drop('date_only', axis=1)
        
        print(f"✅ Merged data shape: {merged_df.shape}")
        print(f"   Records with sentiment: {merged_df['sentiment_mean'].notna().sum()}")
        print(f"   Date range: {merged_df.index.min()} to {merged_df.index.max()}")
        
        return merged_df
    
    def select_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
        """
        Select relevant features for model training.
        
        Args:
            df: Merged DataFrame
            
        Returns:
            Tuple of (df, price_features, sentiment_features)
        """
        # Price/Technical features (for LSTM)
        price_features = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'RSI_14',
            'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0',  # Bollinger Bands
            'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9',  # MACD
            'ATRr_14',  # ATR
            'SMA_20', 'SMA_50',  # Moving Averages
            'OBV'  # Volume indicator
        ]
        
        # Filter to only existing columns
        price_features = [f for f in price_features if f in df.columns]
        
        # Sentiment features (for Dense branch)
        sentiment_features = [
            'sentiment_mean',
            f'sentiment_ma_{self.config["sentiment"]["moving_average_days"]}'
        ]
        
        print(f"\n📊 Selected Features:")
        print(f"   Price/Technical: {len(price_features)} features")
        print(f"   Sentiment: {len(sentiment_features)} features")
        
        return df, price_features, sentiment_features
    
    def normalize_features(
        self,
        df: pd.DataFrame,
        price_features: List[str],
        sentiment_features: List[str],
        fit_scalers: bool = True
    ) -> pd.DataFrame:
        """
        Normalize features using MinMaxScaler.
        
        Args:
            df: DataFrame with features
            price_features: List of price feature names
            sentiment_features: List of sentiment feature names
            fit_scalers: Whether to fit scalers (True for training, False for inference)
            
        Returns:
            Normalized DataFrame
        """
        df_normalized = df.copy()
        
        # Normalize price features
        if fit_scalers:
            df_normalized[price_features] = self.price_scaler.fit_transform(df[price_features])
            print("✅ Fitted and transformed price features")
        else:
            df_normalized[price_features] = self.price_scaler.transform(df[price_features])
            print("✅ Transformed price features (using existing scaler)")
        
        # Normalize sentiment features
        if fit_scalers:
            df_normalized[sentiment_features] = self.sentiment_scaler.fit_transform(df[sentiment_features])
            print("✅ Fitted and transformed sentiment features")
        else:
            df_normalized[sentiment_features] = self.sentiment_scaler.transform(df[sentiment_features])
            print("✅ Transformed sentiment features (using existing scaler)")
        
        return df_normalized
    
    def create_sequences(
        self,
        df: pd.DataFrame,
        price_features: List[str],
        sentiment_features: List[str],
        target_col: str = 'Target'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training.
        
        Args:
            df: Normalized DataFrame
            price_features: List of price feature names
            sentiment_features: List of sentiment feature names
            target_col: Name of target column
            
        Returns:
            Tuple of (X_price, X_sentiment, y)
        """
        X_price_sequences = []
        X_sentiment_sequences = []
        y_targets = []
        
        # Create sliding windows
        for i in range(self.lookback_days, len(df)):
            # Price sequence: last N days
            price_seq = df.iloc[i - self.lookback_days:i][price_features].values
            X_price_sequences.append(price_seq)
            
            # Sentiment: current day's sentiment
            sentiment_vals = df.iloc[i][sentiment_features].values
            X_sentiment_sequences.append(sentiment_vals)
            
            # Target: current day's target
            y_targets.append(df.iloc[i][target_col])
        
        X_price = np.array(X_price_sequences)
        X_sentiment = np.array(X_sentiment_sequences)
        y = np.array(y_targets)
        
        print(f"\n📦 Created sequences:")
        print(f"   X_price shape: {X_price.shape} (samples, lookback, features)")
        print(f"   X_sentiment shape: {X_sentiment.shape} (samples, sentiment_features)")
        print(f"   y shape: {y.shape}")
        print(f"   Class distribution: {np.bincount(y.astype(int))}")
        
        return X_price, X_sentiment, y
    
    def split_train_val_test(
        self,
        df: pd.DataFrame,
        train_end: str | None = None,
        val_end: str | None = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets (time-based).
        
        Args:
            df: DataFrame to split
            train_end: End date for training (inclusive)
            val_end: End date for validation (inclusive)
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        if train_end is None:
            train_end = self.config['data_split']['train_end']
        if val_end is None:
            val_end = self.config['data_split']['validation_end']
        
        # Ensure dates are datetime
        train_end = pd.to_datetime(train_end)
        val_end = pd.to_datetime(val_end)
        
        # Split data
        train_df = df[df.index <= train_end].copy()
        val_df = df[(df.index > train_end) & (df.index <= val_end)].copy()
        test_df = df[df.index > val_end].copy()
        
        print(f"\n📅 Data Split:")
        print(f"   Train: {len(train_df)} samples ({train_df.index.min()} to {train_df.index.max()})")
        print(f"   Validation: {len(val_df)} samples ({val_df.index.min()} to {val_df.index.max()})" if len(val_df) > 0 else "   Validation: No data")
        print(f"   Test: {len(test_df)} samples ({test_df.index.min()} to {test_df.index.max()})" if len(test_df) > 0 else "   Test: No data")
        
        return train_df, val_df, test_df
    
    def prepare_model_data(
        self,
        market_file: str,
        sentiment_file: str,
        save_scalers: bool = True
    ) -> Tuple:
        """
        Complete preprocessing pipeline.
        
        Args:
            market_file: Path to market data CSV
            sentiment_file: Path to sentiment data CSV
            save_scalers: Whether to save fitted scalers
            
        Returns:
            Tuple of prepared train/val/test data
        """
        print("\n🚀 Starting data preprocessing pipeline...")
        print("=" * 60)
        
        # 1. Load data
        print("\n📂 Loading data files...")
        market_df = pd.read_csv(market_file, index_col=0, parse_dates=True)
        sentiment_df = pd.read_csv(sentiment_file, parse_dates=['date'])
        
        print(f"   Market data: {market_df.shape}")
        print(f"   Sentiment data: {sentiment_df.shape}")
        
        # 2. Merge data
        merged_df = self.merge_market_and_sentiment(market_df, sentiment_df)
        
        # 3. Select features
        merged_df, price_features, sentiment_features = self.select_features(merged_df)
        
        # 4. Split into train/val/test
        train_df, val_df, test_df = self.split_train_val_test(merged_df)
        
        # 5. Normalize features (fit on train, transform val/test)
        print("\n🔄 Normalizing features...")
        train_df_norm = self.normalize_features(train_df, price_features, sentiment_features, fit_scalers=True)
        
        if len(val_df) > 0:
            val_df_norm = self.normalize_features(val_df, price_features, sentiment_features, fit_scalers=False)
        else:
            val_df_norm = pd.DataFrame()
        
        if len(test_df) > 0:
            test_df_norm = self.normalize_features(test_df, price_features, sentiment_features, fit_scalers=False)
        else:
            test_df_norm = pd.DataFrame()
        
        # 6. Create sequences
        print("\n📊 Creating sequences...")
        X_price_train, X_sent_train, y_train = self.create_sequences(
            train_df_norm, price_features, sentiment_features
        )
        
        if len(val_df_norm) > 0:
            X_price_val, X_sent_val, y_val = self.create_sequences(
                val_df_norm, price_features, sentiment_features
            )
        else:
            X_price_val, X_sent_val, y_val = None, None, None
        
        if len(test_df_norm) > 0:
            X_price_test, X_sent_test, y_test = self.create_sequences(
                test_df_norm, price_features, sentiment_features
            )
        else:
            X_price_test, X_sent_test, y_test = None, None, None
        
        # 7. Save scalers and feature names
        if save_scalers:
            self._save_artifacts(price_features, sentiment_features)
        
        print("\n✅ Data preprocessing complete!")
        
        return (
            (X_price_train, X_sent_train, y_train),
            (X_price_val, X_sent_val, y_val),
            (X_price_test, X_sent_test, y_test),
            price_features,
            sentiment_features
        )
    
    def _save_artifacts(self, price_features: List[str], sentiment_features: List[str]):
        """Save scalers and feature names."""
        artifacts = {
            'price_scaler': self.price_scaler,
            'sentiment_scaler': self.sentiment_scaler,
            'price_features': price_features,
            'sentiment_features': sentiment_features,
            'lookback_days': self.lookback_days
        }
        
        filepath = self.processed_data_path / 'preprocessing_artifacts.pkl'
        with open(filepath, 'wb') as f:
            pickle.dump(artifacts, f)
        
        print(f"\n💾 Saved preprocessing artifacts to: {filepath}")
    
    def load_artifacts(self):
        """Load saved scalers and feature names."""
        filepath = self.processed_data_path / 'preprocessing_artifacts.pkl'
        
        with open(filepath, 'rb') as f:
            artifacts = pickle.load(f)
        
        self.price_scaler = artifacts['price_scaler']
        self.sentiment_scaler = artifacts['sentiment_scaler']
        
        print(f"📂 Loaded preprocessing artifacts from: {filepath}")
        
        return artifacts


def main():
    """Test preprocessing pipeline."""
    import sys
    from pathlib import Path
    
    preprocessor = DataPreprocessor()
    
    # Find latest data files
    raw_path = Path('data/raw')
    processed_path = Path('data/processed')
    
    market_files = list(raw_path.glob('market_data_*.csv'))
    sentiment_files = list(processed_path.glob('daily_sentiment_*.csv'))
    
    if not market_files:
        print("❌ No market data found! Run market_data.py first.")
        sys.exit(1)
    
    if not sentiment_files:
        print("❌ No sentiment data found! Run finbert_engine.py first.")
        sys.exit(1)
    
    # Get latest files
    import os
    latest_market = max(market_files, key=os.path.getctime)
    latest_sentiment = max(sentiment_files, key=os.path.getctime)
    
    print(f"📂 Using market data: {latest_market.name}")
    print(f"📂 Using sentiment data: {latest_sentiment.name}")
    
    # Run preprocessing
    train_data, val_data, test_data, price_feat, sent_feat = preprocessor.prepare_model_data(
        str(latest_market),
        str(latest_sentiment)
    )
    
    print("\n" + "=" * 60)
    print("📋 Preprocessing Summary:")
    print("=" * 60)
    print(f"Training data: {train_data[0].shape if train_data[0] is not None else 'None'}")
    print(f"Validation data: {val_data[0].shape if val_data[0] is not None else 'None'}")
    print(f"Test data: {test_data[0].shape if test_data[0] is not None else 'None'}")
    
    print("\n✅ Preprocessing complete!")


if __name__ == "__main__":
    main()
