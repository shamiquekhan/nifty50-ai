"""
State-of-the-Art Bi-Directional LSTM with Attention Mechanism
Research-grade architecture for NIFTY50 prediction with Late Fusion.
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, Concatenate, 
    Bidirectional, BatchNormalization, Layer, Multiply, Permute, RepeatVector, Lambda
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import tensorflow.keras.backend as K
import yaml
import numpy as np
from pathlib import Path
from typing import Tuple


class AttentionLayer(Layer):
    """
    Attention mechanism for LSTM outputs.
    Learns which time steps are most important for prediction.
    """
    
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        # input_shape: (batch, timesteps, features)
        self.W = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], 1),
            initializer='random_normal',
            trainable=True
        )
        self.b = self.add_weight(
            name='attention_bias',
            shape=(input_shape[1], 1),
            initializer='zeros',
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, x):
        # x shape: (batch, timesteps, features)
        # Calculate attention scores
        e = K.tanh(K.dot(x, self.W) + self.b)  # (batch, timesteps, 1)
        a = K.softmax(e, axis=1)  # Attention weights
        
        # Apply attention weights
        output = x * a  # (batch, timesteps, features)
        return K.sum(output, axis=1)  # (batch, features)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


class BiLSTMAttentionModel:
    """
    State-of-the-Art Hybrid Architecture:
    - Bi-Directional LSTM with Attention for time-series (price data)
    - Dense network for sentiment (FinBERT outputs)
    - Late Fusion for combining both modalities
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
        Build Bi-Directional LSTM with Attention + Late Fusion.
        
        Architecture:
        1. Price Branch: BiLSTM → Attention → Dense (extracts temporal patterns)
        2. Sentiment Branch: Dense layers (processes FinBERT scores)
        3. Late Fusion: Concatenate both branches → Final prediction
        
        Args:
            price_shape: (lookback_days, num_features) e.g., (60, 5)
            sentiment_shape: (num_sentiment_features,) e.g., (1,)
            
        Returns:
            Compiled Keras model
        """
        print("\n🧠 Building State-of-the-Art BiLSTM + Attention Model...")
        print("=" * 80)
        print("📚 Research-Grade Features:")
        print("   ✅ Bi-Directional LSTM (past + future context)")
        print("   ✅ Attention Mechanism (learns important time steps)")
        print("   ✅ Late Fusion (optimal integration of price + sentiment)")
        print("   ✅ Batch Normalization (training stability)")
        print("=" * 80)
        
        lstm_config = self.model_config['lstm']
        sent_config = self.model_config['sentiment_branch']
        fusion_config = self.model_config['fusion']
        
        # ===== BRANCH A: Price/Technical Data (Time Series) =====
        price_input = Input(shape=price_shape, name='price_input')
        
        # First Bi-Directional LSTM layer with return_sequences=True
        # This captures patterns in both forward and backward time directions
        x1 = Bidirectional(
            LSTM(64, return_sequences=True, name='bilstm_1'),
            name='bidirectional_1'
        )(price_input)
        x1 = BatchNormalization()(x1)
        x1 = Dropout(0.2)(x1)
        
        # Second Bi-Directional LSTM layer
        x1 = Bidirectional(
            LSTM(32, return_sequences=True, name='bilstm_2'),
            name='bidirectional_2'
        )(x1)
        x1 = BatchNormalization()(x1)
        x1 = Dropout(0.2)(x1)
        
        # Attention Layer - learns which time steps matter most
        # E.g., earnings announcements or policy decisions
        x1 = AttentionLayer(name='attention')(x1)
        
        # Dense layer for price branch
        x1 = Dense(32, activation='relu', name='price_dense_1')(x1)
        x1 = Dropout(0.2)(x1)
        x1 = Dense(16, activation='relu', name='price_dense_2')(x1)
        
        # ===== BRANCH B: Sentiment Data (FinBERT Outputs) =====
        sentiment_input = Input(shape=sentiment_shape, name='sentiment_input')
        
        # Process sentiment scores through dense layers
        x2 = Dense(16, activation='relu', name='sentiment_dense_1')(sentiment_input)
        x2 = Dropout(0.1)(x2)
        x2 = Dense(8, activation='relu', name='sentiment_dense_2')(x2)
        
        # ===== LATE FUSION: Concatenate Both Intelligence Sources =====
        # Late fusion allows the model to learn optimal weighting of:
        # - Technical patterns (from BiLSTM+Attention)
        # - News sentiment (from FinBERT)
        combined = Concatenate(name='late_fusion')([x1, x2])
        
        # Fusion dense layers learn to combine both modalities
        z = Dense(32, activation='relu', name='fusion_dense_1')(combined)
        z = BatchNormalization()(z)
        z = Dropout(0.3)(z)
        z = Dense(16, activation='relu', name='fusion_dense_2')(z)
        z = Dropout(0.2)(z)
        
        # ===== OUTPUT LAYER =====
        # Sigmoid for binary classification: price UP (1) or DOWN (0)
        output = Dense(1, activation='sigmoid', name='prediction')(z)
        
        # ===== BUILD MODEL =====
        model = Model(
            inputs=[price_input, sentiment_input],
            outputs=output,
            name='BiLSTM_Attention_LateFusion'
        )
        
        # Compile with Adam optimizer
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ]
        )
        
        print("\n📋 Model Architecture:")
        print("=" * 80)
        model.summary()
        
        # Calculate total parameters
        total_params = model.count_params()
        print(f"\n📊 Total Parameters: {total_params:,}")
        print("=" * 80)
        
        self.model = model
        return model
    
    def get_callbacks(self, model_name: str = 'bilstm_attention.keras'):
        """
        Create training callbacks for optimal training.
        
        Args:
            model_name: Name for saved model file
            
        Returns:
            List of Keras callbacks
        """
        callbacks = [
            # Early stopping - stops if val_loss doesn't improve for 10 epochs
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1,
                mode='min'
            ),
            
            # Save best model based on validation AUC
            ModelCheckpoint(
                filepath=str(self.models_path / model_name),
                monitor='val_auc',
                mode='max',
                save_best_only=True,
                verbose=1
            ),
            
            # Reduce learning rate when validation loss plateaus
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1,
                mode='min'
            ),
            
            # TensorBoard logging (optional)
            tf.keras.callbacks.TensorBoard(
                log_dir=str(self.models_path / 'logs'),
                histogram_freq=1,
                write_graph=True
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
        epochs: int = 50,
        batch_size: int = 32,
        model_name: str = 'bilstm_attention.keras'
    ):
        """
        Train the BiLSTM+Attention model.
        
        Args:
            X_price_train: Training price sequences (batch, timesteps, features)
            X_sentiment_train: Training sentiment (batch, sentiment_features)
            y_train: Training labels (batch,)
            X_price_val: Validation price sequences
            X_sentiment_val: Validation sentiment
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size for training
            model_name: Name for saved model
            
        Returns:
            Training history
        """
        if self.model is None:
            # Build model with correct shapes
            price_shape = (X_price_train.shape[1], X_price_train.shape[2])
            sentiment_shape = (X_sentiment_train.shape[1],)
            self.build_model(price_shape, sentiment_shape)
        
        print("\n🚀 Starting Training...")
        print("=" * 80)
        print(f"📊 Dataset Statistics:")
        print(f"   Training samples: {len(X_price_train):,}")
        print(f"   Validation samples: {len(X_price_val):,}")
        print(f"   Price features: {X_price_train.shape[2]}")
        print(f"   Sentiment features: {X_sentiment_train.shape[1]}")
        print(f"   Lookback window: {X_price_train.shape[1]} days")
        print(f"\n⚙️  Training Configuration:")
        print(f"   Epochs: {epochs}")
        print(f"   Batch size: {batch_size}")
        print(f"   Optimizer: Adam (lr=0.001)")
        print("=" * 80)
        
        # Train model
        history = self.model.fit(
            [X_price_train, X_sentiment_train],
            y_train,
            validation_data=([X_price_val, X_sentiment_val], y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=self.get_callbacks(model_name),
            verbose=1,
            class_weight={0: 1.0, 1: 1.2}  # Slight bias towards UP class
        )
        
        self.history = history
        
        print("\n✅ Training Complete!")
        print(f"📁 Model saved to: {self.models_path / model_name}")
        
        return history
    
    def predict(
        self,
        X_price: np.ndarray,
        X_sentiment: np.ndarray
    ) -> np.ndarray:
        """
        Make predictions with trained model.
        
        Args:
            X_price: Price sequences
            X_sentiment: Sentiment features
            
        Returns:
            Prediction probabilities (probability of price going UP)
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
        Evaluate model performance on test set.
        
        Args:
            X_price: Test price sequences
            X_sentiment: Test sentiment features
            y_true: True labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not built or loaded.")
        
        # Get predictions
        y_pred_prob = self.predict(X_price, X_sentiment)
        y_pred = (y_pred_prob > 0.5).astype(int)
        
        # Calculate metrics
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, 
            f1_score, roc_auc_score, confusion_matrix
        )
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred_prob)
        cm = confusion_matrix(y_true, y_pred)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'confusion_matrix': cm,
            'true_positives': cm[1, 1],
            'false_positives': cm[0, 1],
            'true_negatives': cm[0, 0],
            'false_negatives': cm[1, 0]
        }
        
        print("\n📊 Model Evaluation Results:")
        print("=" * 80)
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"AUC-ROC:   {auc:.4f}")
        print("\nConfusion Matrix:")
        print(cm)
        print("=" * 80)
        
        return metrics
    
    def load_model(self, model_path: str):
        """Load pre-trained model."""
        self.model = tf.keras.models.load_model(
            model_path,
            custom_objects={'AttentionLayer': AttentionLayer}
        )
        print(f"✅ Model loaded from: {model_path}")
    
    def save_model(self, model_path: str):
        """Save model to file."""
        if self.model is None:
            raise ValueError("No model to save.")
        
        self.model.save(model_path)
        print(f"💾 Model saved to: {model_path}")
