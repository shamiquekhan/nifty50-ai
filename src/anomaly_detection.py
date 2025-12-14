"""
Isolation Forest Anomaly Detection - Red Flag System
Detects unusual market behavior: pump-and-dump, manipulation, flash crashes.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class AnomalyDetector:
    """
    White-Box Red Flag Detection System.
    Uses Isolation Forest to identify market anomalies without labeled data.
    """
    
    def __init__(self, contamination: float = 0.01):
        """
        Initialize anomaly detector.
        
        Args:
            contamination: Expected proportion of anomalies (default 1%)
        """
        self.contamination = contamination
        self.model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100,
            max_samples='auto'
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract features for anomaly detection.
        
        Features:
        - Volume deviation from average
        - Price volatility (daily range)
        - Return magnitude
        - Volume-to-price ratio
        - Bid-ask spread proxy (High-Low range)
        
        Args:
            df: Market data DataFrame
            
        Returns:
            Feature matrix
        """
        features = pd.DataFrame()
        
        # 1. Volume deviation (Z-score)
        features['volume_zscore'] = (
            (df['Volume'] - df['Volume'].rolling(20).mean()) / 
            df['Volume'].rolling(20).std()
        ).fillna(0)
        
        # 2. Daily volatility (High-Low range as % of Close)
        features['daily_volatility'] = (
            (df['High'] - df['Low']) / df['Close']
        ).fillna(0)
        
        # 3. Absolute return magnitude
        features['return_magnitude'] = (
            np.abs(df['Close'].pct_change())
        ).fillna(0)
        
        # 4. Volume-to-price ratio (unusual for large caps)
        features['volume_price_ratio'] = (
            df['Volume'] / df['Close']
        ).fillna(0)
        
        # 5. Price gap (open vs previous close)
        features['price_gap'] = (
            np.abs((df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1))
        ).fillna(0)
        
        # 6. Volume spike (current vs 20-day average)
        features['volume_spike'] = (
            df['Volume'] / df['Volume'].rolling(20).mean()
        ).fillna(1)
        
        # 7. Price range expansion
        features['range_expansion'] = (
            (df['High'] - df['Low']) / 
            (df['High'] - df['Low']).rolling(20).mean()
        ).fillna(1)
        
        return features.values
    
    def fit(self, df: pd.DataFrame):
        """
        Train anomaly detector on historical data.
        
        Args:
            df: Historical market data
        """
        print("\n🔍 Training Anomaly Detector (Isolation Forest)...")
        print("=" * 70)
        
        # Prepare features
        X = self.prepare_features(df)
        
        # Remove NaN rows
        valid_idx = ~np.isnan(X).any(axis=1)
        X_clean = X[valid_idx]
        
        # Standardize features
        X_scaled = self.scaler.fit_transform(X_clean)
        
        # Train Isolation Forest
        self.model.fit(X_scaled)
        self.is_fitted = True
        
        # Calculate baseline anomaly rate
        predictions = self.model.predict(X_scaled)
        anomaly_rate = (predictions == -1).sum() / len(predictions)
        
        print(f"✅ Anomaly Detector Trained")
        print(f"   Training samples: {len(X_clean):,}")
        print(f"   Features: {X_clean.shape[1]}")
        print(f"   Anomaly rate: {anomaly_rate*100:.2f}%")
        print(f"   Expected contamination: {self.contamination*100:.2f}%")
        print("=" * 70)
    
    def detect(
        self, 
        df: pd.DataFrame, 
        return_scores: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies in market data.
        
        Args:
            df: Market data DataFrame
            return_scores: If True, return anomaly scores
            
        Returns:
            Tuple of (predictions, scores)
            predictions: 1 = normal, -1 = anomaly
            scores: Anomaly scores (lower = more anomalous)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Prepare features
        X = self.prepare_features(df)
        
        # Handle NaN
        valid_idx = ~np.isnan(X).any(axis=1)
        X_clean = X[valid_idx]
        
        # Standardize
        X_scaled = self.scaler.transform(X_clean)
        
        # Predict
        predictions = np.ones(len(X))  # Default to normal
        scores = np.zeros(len(X))
        
        # Fill valid indices
        predictions[valid_idx] = self.model.predict(X_scaled)
        scores[valid_idx] = self.model.score_samples(X_scaled)
        
        if return_scores:
            return predictions, scores
        return predictions
    
    def check_latest(
        self, 
        df: pd.DataFrame, 
        lookback: int = 5
    ) -> Dict:
        """
        Check recent data for red flags.
        
        Args:
            df: Market data DataFrame
            lookback: Number of recent days to check
            
        Returns:
            Dictionary with red flag analysis
        """
        if not self.is_fitted:
            # Fit on all data except last day
            self.fit(df[:-1])
        
        # Get recent data
        recent_df = df.tail(lookback)
        
        # Detect anomalies
        predictions, scores = self.detect(recent_df)
        
        # Analyze results
        latest_prediction = predictions[-1]
        latest_score = scores[-1]
        recent_anomalies = (predictions == -1).sum()
        
        # Categorize severity
        if latest_prediction == -1:
            if latest_score < -0.5:
                severity = "CRITICAL"
                color = "🔴"
            elif latest_score < -0.3:
                severity = "HIGH"
                color = "🟠"
            else:
                severity = "MODERATE"
                color = "🟡"
        else:
            severity = "NORMAL"
            color = "🟢"
        
        # Identify specific red flags
        red_flags = []
        latest_data = df.iloc[-1]
        
        # Check volume spike
        avg_volume = df['Volume'].tail(20).mean()
        if latest_data['Volume'] > avg_volume * 3:
            red_flags.append(f"Volume spike: {latest_data['Volume']/avg_volume:.1f}x average")
        
        # Check price volatility
        daily_range = (latest_data['High'] - latest_data['Low']) / latest_data['Close']
        avg_range = ((df['High'] - df['Low']) / df['Close']).tail(20).mean()
        if daily_range > avg_range * 2:
            red_flags.append(f"High volatility: {daily_range*100:.1f}% daily range")
        
        # Check unusual return
        latest_return = (latest_data['Close'] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]
        if abs(latest_return) > 0.05:
            red_flags.append(f"Large move: {latest_return*100:+.1f}% return")
        
        # Check gap
        if 'Open' in df.columns:
            gap = abs((latest_data['Open'] - df['Close'].iloc[-2]) / df['Close'].iloc[-2])
            if gap > 0.03:
                red_flags.append(f"Price gap: {gap*100:.1f}% overnight move")
        
        result = {
            'is_anomaly': latest_prediction == -1,
            'severity': severity,
            'color': color,
            'anomaly_score': latest_score,
            'recent_anomalies': recent_anomalies,
            'lookback': lookback,
            'red_flags': red_flags,
            'message': self._generate_message(severity, red_flags)
        }
        
        return result
    
    def _generate_message(self, severity: str, red_flags: List[str]) -> str:
        """Generate human-readable message."""
        if severity == "CRITICAL":
            msg = "⚠️ CRITICAL RED FLAG: Extreme market anomaly detected. "
            msg += "Possible manipulation or flash crash. AVOID TRADING."
        elif severity == "HIGH":
            msg = "⚠️ HIGH RISK: Unusual market behavior detected. "
            msg += "Exercise extreme caution."
        elif severity == "MODERATE":
            msg = "⚠️ MODERATE ALERT: Some unusual activity detected. "
            msg += "Monitor closely before trading."
        else:
            msg = "✅ NORMAL: Market behavior within expected range."
        
        if red_flags:
            msg += "\n\nSpecific Flags:\n"
            for flag in red_flags:
                msg += f"  • {flag}\n"
        
        return msg
    
    def get_anomaly_timeline(
        self, 
        df: pd.DataFrame, 
        min_days: int = 30
    ) -> pd.DataFrame:
        """
        Get timeline of anomalies with context.
        
        Args:
            df: Market data DataFrame
            min_days: Minimum days required
            
        Returns:
            DataFrame with anomaly timeline
        """
        if len(df) < min_days:
            raise ValueError(f"Need at least {min_days} days of data")
        
        if not self.is_fitted:
            self.fit(df)
        
        # Detect all anomalies
        predictions, scores = self.detect(df)
        
        # Create timeline
        timeline = pd.DataFrame({
            'date': df.index if isinstance(df.index, pd.DatetimeIndex) else range(len(df)),
            'close': df['Close'].values,
            'volume': df['Volume'].values,
            'is_anomaly': predictions == -1,
            'anomaly_score': scores
        })
        
        # Add context
        timeline['return'] = df['Close'].pct_change()
        timeline['volume_zscore'] = (
            (df['Volume'] - df['Volume'].rolling(20).mean()) / 
            df['Volume'].rolling(20).std()
        )
        
        # Filter to anomalies only
        anomalies = timeline[timeline['is_anomaly']].copy()
        
        return anomalies
    
    def summarize_risk(self, df: pd.DataFrame) -> Dict:
        """
        Generate comprehensive risk summary.
        
        Args:
            df: Market data DataFrame
            
        Returns:
            Risk summary dictionary
        """
        if not self.is_fitted:
            self.fit(df)
        
        # Detect anomalies
        predictions, scores = self.detect(df)
        
        # Calculate statistics
        total_days = len(predictions)
        anomaly_days = (predictions == -1).sum()
        anomaly_rate = anomaly_days / total_days
        
        # Recent anomalies (last 30 days)
        recent_anomalies = (predictions[-30:] == -1).sum()
        
        # Average anomaly score
        avg_score = scores.mean()
        worst_score = scores.min()
        
        # Risk classification
        if anomaly_rate > 0.05:
            risk_level = "HIGH"
        elif anomaly_rate > 0.02:
            risk_level = "MODERATE"
        else:
            risk_level = "LOW"
        
        summary = {
            'total_days': total_days,
            'anomaly_days': anomaly_days,
            'anomaly_rate': anomaly_rate,
            'recent_anomalies_30d': recent_anomalies,
            'avg_anomaly_score': avg_score,
            'worst_anomaly_score': worst_score,
            'risk_level': risk_level
        }
        
        return summary
