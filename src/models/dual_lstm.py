"""
Dual-Input LSTM Model Architecture
Combines price/technical indicators (LSTM) with sentiment data (Dense).
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, Concatenate, 
    Bidirectional, BatchNormalization
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import yaml
import numpy as np
from pathlib import Path
from typing import Tuple


class DualLSTMModel:
    """
    Neuro-Symbolic Ensemble Model:
    - Branch A: Bi-Directional LSTM for time-series price data
    - Branch B: Dense network for sentiment features
    - Fusion: Combined prediction layer
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_config = self.config['model']
        self.models_path = Path(self.config['paths']['models'])
        self.models_path.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.history = None
    
    def build_model(
        self, 
        price_shape: Tuple[int, int],
        sentiment_shape: Tuple[int,]
    ) -> Model:
        """
        Build the dual-input model architecture.
        
        Args:
            price_shape: Shape of price input (lookback_days, num_features)
            sentiment_shape: Shape of sentiment input (num_sentiment_features,)
            
        Returns:
            Compiled Keras model
        """
        print("\n🏗️  Building Dual-Input LSTM Model...")
        print("=" * 60)
        
        lstm_config = self.model_config['lstm']
        sent_config = self.model_config['sentiment_branch']
        fusion_config = self.model_config['fusion']
        
        # ===== BRANCH A: Price/Technical Data (Time Series) =====
        price_input = Input(shape=price_shape, name='price_input')
        
        if lstm_config['bidirectional']:
            # Bi-Directional LSTM captures patterns from both past and future
            x1 = Bidirectional(
                LSTM(lstm_config['units_layer1'], return_sequences=True)
            )(price_input)
        else:
            x1 = LSTM(lstm_config['units_layer1'], return_sequences=True)(price_input)
        
        x1 = Dropout(lstm_config['dropout_rate'])(x1)
        x1 = BatchNormalization()(x1)
        
        # Second LSTM layer (no return_sequences, outputs final state)
        x1 = LSTM(lstm_config['units_layer2'])(x1)
        x1 = Dropout(lstm_config['dropout_rate'])(x1)
        
        # Dense layer for price branch
        x1 = Dense(16, activation='relu', name='price_dense')(x1)
        
        # ===== BRANCH B: Sentiment Data (Static Features) =====
        sentiment_input = Input(shape=sentiment_shape, name='sentiment_input')
        
        x2 = Dense(sent_config['dense_units'], activation='relu', name='sentiment_dense_1')(sentiment_input)
        x2 = Dropout(0.1)(x2)
        x2 = Dense(4, activation='relu', name='sentiment_dense_2')(x2)
        
        # ===== FUSION LAYER: Combine Both Branches =====
        combined = Concatenate(name='fusion_concat')([x1, x2])
        
        z = Dense(fusion_config['dense_units'], activation='relu', name='fusion_dense')(combined)
        z = Dropout(fusion_config['dropout_rate'])(z)
        z = Dense(8, activation='relu', name='fusion_dense_2')(z)
        
        # ===== OUTPUT LAYER =====
        # Sigmoid activation for binary classification (price up/down)
        output = Dense(1, activation='sigmoid', name='output')(z)
        
        # ===== BUILD MODEL =====
        model = Model(
            inputs=[price_input, sentiment_input],
            outputs=output,
            name='DualLSTM_Ensemble'
        )
        
        # Compile model
        train_config = self.model_config['training']
        model.compile(
            optimizer=train_config['optimizer'],
            loss=train_config['loss'],
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        
        print("\n📋 Model Architecture:")
        print("=" * 60)
        model.summary()
        
        self.model = model
        return model
    
    def get_callbacks(self, model_name: str = 'best_model.keras'):
        """
        Create training callbacks.
        
        Args:
            model_name: Name for saved model file
            
        Returns:
            List of Keras callbacks
        """
        train_config = self.model_config['training']
        
        callbacks = [
            # Early stopping to prevent overfitting
            EarlyStopping(
                monitor='val_loss',
                patience=train_config['early_stopping_patience'],
                restore_best_weights=True,
                verbose=1
            ),
            
            # Save best model
            ModelCheckpoint(
                filepath=str(self.models_path / model_name),
                monitor='val_auc',
                mode='max',
                save_best_only=True,
                verbose=1
            ),
            
            # Reduce learning rate on plateau
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        return callbacks
    
    def train(
        self,
        X_price_train: np.ndarray,
        X_sentiment_train: np.ndarray,
        y_train: np.ndarray,
        X_price_val: np.ndarray,
        X_sentiment_val: np.ndarray,
        y_val: np.ndarray,
        model_name: str = 'best_model.keras'
    ):
        """
        Train the model.
        
        Args:
            X_price_train: Training price sequences
            X_sentiment_train: Training sentiment features
            y_train: Training labels
            X_price_val: Validation price sequences
            X_sentiment_val: Validation sentiment features
            y_val: Validation labels
            model_name: Name for saved model
            
        Returns:
            Training history
        """
        if self.model is None:
            # Build model with correct shapes
            price_shape = (X_price_train.shape[1], X_price_train.shape[2])
            sentiment_shape = (X_sentiment_train.shape[1],)
            self.build_model(price_shape, sentiment_shape)
        
        train_config = self.model_config['training']
        
        print("\n🚀 Starting model training...")
        print("=" * 60)
        print(f"Training samples: {len(X_price_train)}")
        print(f"Validation samples: {len(X_price_val)}")
        print(f"Epochs: {train_config['epochs']}")
        print(f"Batch size: {train_config['batch_size']}")
        print("=" * 60)
        
        # Train model
        history = self.model.fit(
            [X_price_train, X_sentiment_train],
            y_train,
            validation_data=([X_price_val, X_sentiment_val], y_val),
            epochs=train_config['epochs'],
            batch_size=train_config['batch_size'],
            callbacks=self.get_callbacks(model_name),
            verbose=1
        )
        
        self.history = history
        
        print("\n✅ Training complete!")
        
        return history
    
    def predict(
        self,
        X_price: np.ndarray,
        X_sentiment: np.ndarray
    ) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X_price: Price sequences
            X_sentiment: Sentiment features
            
        Returns:
            Prediction probabilities
        """
        if self.model is None:
            raise ValueError("Model not built or loaded. Call build_model() or load_model() first.")
        
        predictions = self.model.predict([X_price, X_sentiment], verbose=0)
        return predictions
    
    def evaluate(
        self,
        X_price: np.ndarray,
        X_sentiment: np.ndarray,
        y_true: np.ndarray
    ) -> dict:
        """
        Evaluate model performance.
        
        Args:
            X_price: Price sequences
            X_sentiment: Sentiment features
            y_true: True labels
            
        Returns:
            Dictionary of metrics
        """
        if self.model is None:
            raise ValueError("Model not built or loaded.")
        
        # Get metrics
        results = self.model.evaluate(
            [X_price, X_sentiment],
            y_true,
            verbose=0
        )
        
        metrics = dict(zip(self.model.metrics_names, results))
        
        # Get predictions for additional metrics
        y_pred_prob = self.predict(X_price, X_sentiment)
        y_pred = (y_pred_prob > 0.5).astype(int)
        
        # Calculate additional metrics
        from sklearn.metrics import classification_report, confusion_matrix
        
        print("\n📊 Model Evaluation Results:")
        print("=" * 60)
        print(f"Loss: {metrics['loss']:.4f}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"AUC: {metrics['auc']:.4f}")
        
        print("\n📋 Classification Report:")
        print(classification_report(y_true, y_pred, target_names=['Down', 'Up']))
        
        print("\n📊 Confusion Matrix:")
        cm = confusion_matrix(y_true, y_pred)
        print(f"True Negatives:  {cm[0][0]}")
        print(f"False Positives: {cm[0][1]}")
        print(f"False Negatives: {cm[1][0]}")
        print(f"True Positives:  {cm[1][1]}")
        
        return metrics
    
    def save_model(self, filepath: str = None):
        """Save the model."""
        if filepath is None:
            filepath = str(self.models_path / 'dual_lstm_model.keras')
        
        self.model.save(filepath)
        print(f"💾 Model saved to: {filepath}")
    
    def load_model(self, filepath: str = None):
        """Load a saved model."""
        if filepath is None:
            filepath = str(self.models_path / 'best_model.keras')
        
        self.model = tf.keras.models.load_model(filepath)
        print(f"📂 Model loaded from: {filepath}")
        
        return self.model


def main():
    """Main execution function for testing."""
    model = DualLSTMModel()
    
    # Build model with example shapes
    price_shape = (30, 10)  # 30 days, 10 features
    sentiment_shape = (2,)   # 2 sentiment features
    
    model.build_model(price_shape, sentiment_shape)
    
    print("\n✅ Model architecture created successfully!")


if __name__ == "__main__":
    main()
