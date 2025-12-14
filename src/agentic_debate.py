"""
Agentic Debate System - Bull vs Bear Consensus Protocol
Multi-agent deliberation for transparent trading decisions.

Architecture:
1. Analyst Agent: Runs BiLSTM+Attention model predictions
2. Bull Agent: Argues for buying (growth perspective)
3. Bear Agent: Argues for caution (risk perspective)
4. Moderator Agent: Synthesizes arguments into consensus
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum


class AgentRole(Enum):
    """Agent roles in the debate."""
    ANALYST = "ANALYST"
    BULL = "BULL"
    BEAR = "BEAR"
    MODERATOR = "MODERATOR"


@dataclass
class Argument:
    """Structured argument from an agent."""
    agent: AgentRole
    claim: str
    evidence: List[str]
    confidence: float  # 0.0 to 1.0
    weight: float = 1.0  # Importance weight


@dataclass
class Consensus:
    """Final consensus from debate."""
    action: str  # BUY, SHORT, WAIT
    confidence: float
    position_size: float
    bull_score: float
    bear_score: float
    arguments: List[Argument]
    final_reasoning: str


class AnalystAgent:
    """
    Analyst Agent: Runs model predictions and provides objective analysis.
    """
    
    def __init__(self, model):
        """
        Initialize with trained model.
        
        Args:
            model: Trained BiLSTM+Attention model
        """
        self.model = model
        self.role = AgentRole.ANALYST
    
    def analyze(
        self, 
        price_data: np.ndarray,
        sentiment_data: np.ndarray,
        ticker: str
    ) -> Argument:
        """
        Run model prediction and create argument.
        
        Args:
            price_data: Price sequences
            sentiment_data: Sentiment features
            ticker: Stock ticker
            
        Returns:
            Argument with model prediction
        """
        # Get model prediction
        prediction = self.model.predict(
            price_data.reshape(1, *price_data.shape),
            sentiment_data.reshape(1, *sentiment_data.shape)
        )[0][0]
        
        # Create evidence
        evidence = [
            f"Model probability: {prediction*100:.1f}%",
            f"Prediction: {'UP' if prediction > 0.5 else 'DOWN'}",
            f"Confidence: {'HIGH' if abs(prediction - 0.5) > 0.25 else 'MODERATE' if abs(prediction - 0.5) > 0.15 else 'LOW'}"
        ]
        
        # Formulate claim
        if prediction > 0.75:
            claim = f"Strong bullish signal for {ticker}. Model shows {prediction*100:.1f}% probability of upward move."
        elif prediction > 0.6:
            claim = f"Moderate bullish signal for {ticker}. Model leans positive at {prediction*100:.1f}%."
        elif prediction > 0.4:
            claim = f"Neutral signal for {ticker}. Model uncertainty at {prediction*100:.1f}%."
        else:
            claim = f"Bearish signal for {ticker}. Model shows {(1-prediction)*100:.1f}% probability of downward move."
        
        return Argument(
            agent=self.role,
            claim=claim,
            evidence=evidence,
            confidence=abs(prediction - 0.5) * 2,  # Convert to 0-1 scale
            weight=1.5  # Analyst has high weight
        )


class BullAgent:
    """
    Bull Agent: Argues for growth and buying opportunities.
    Focuses on positive sentiment, momentum, technical strength.
    """
    
    def __init__(self):
        self.role = AgentRole.BULL
    
    def argue(
        self,
        df: pd.DataFrame,
        sentiment_score: float,
        model_prob: float,
        analyst_arg: Argument
    ) -> Argument:
        """
        Build bullish argument.
        
        Args:
            df: Market data DataFrame
            sentiment_score: Sentiment score
            model_prob: Model probability
            analyst_arg: Analyst's argument
            
        Returns:
            Bullish argument
        """
        evidence = []
        bull_points = 0
        
        # 1. Check sentiment
        if sentiment_score > 0.2:
            evidence.append(f"✅ Strong positive sentiment: {sentiment_score:+.3f}")
            bull_points += 2
        elif sentiment_score > 0:
            evidence.append(f"✅ Positive sentiment: {sentiment_score:+.3f}")
            bull_points += 1
        
        # 2. Check model probability
        if model_prob > 0.7:
            evidence.append(f"✅ High model confidence: {model_prob*100:.1f}%")
            bull_points += 2
        elif model_prob > 0.5:
            evidence.append(f"✅ Model favors upside: {model_prob*100:.1f}%")
            bull_points += 1
        
        # 3. Check recent momentum
        if len(df) >= 5:
            recent_returns = df['Close'].pct_change().tail(5)
            avg_return = recent_returns.mean()
            if avg_return > 0.005:
                evidence.append(f"✅ Positive momentum: {avg_return*100:+.2f}% avg 5-day return")
                bull_points += 1
        
        # 4. Check RSI (if available)
        if 'RSI_14' in df.columns:
            latest_rsi = df['RSI_14'].iloc[-1]
            if 30 < latest_rsi < 50:
                evidence.append(f"✅ RSI in buy zone: {latest_rsi:.1f} (oversold recovery)")
                bull_points += 1
            elif 50 <= latest_rsi < 70:
                evidence.append(f"✅ RSI in bullish territory: {latest_rsi:.1f}")
                bull_points += 1
        
        # 5. Check volume trend
        if len(df) >= 20:
            recent_volume = df['Volume'].tail(5).mean()
            avg_volume = df['Volume'].tail(20).mean()
            if recent_volume > avg_volume * 1.2:
                evidence.append(f"✅ Strong buying volume: {recent_volume/avg_volume:.1f}x average")
                bull_points += 1
        
        # 6. Check price above moving average
        if 'SMA_50' in df.columns:
            latest_price = df['Close'].iloc[-1]
            sma_50 = df['SMA_50'].iloc[-1]
            if latest_price > sma_50:
                pct_above = ((latest_price - sma_50) / sma_50) * 100
                evidence.append(f"✅ Price above SMA(50): +{pct_above:.1f}%")
                bull_points += 1
        
        # Formulate claim based on bull points
        if bull_points >= 5:
            claim = "STRONG BUY: Multiple bullish indicators align. Excellent risk-reward setup."
            confidence = 0.9
        elif bull_points >= 3:
            claim = "BUY: Several bullish factors support upside potential."
            confidence = 0.7
        elif bull_points >= 1:
            claim = "CAUTIOUS BUY: Some bullish signs present, but limited conviction."
            confidence = 0.5
        else:
            claim = "INSUFFICIENT BULLISH CASE: Lack of strong buy signals."
            confidence = 0.2
            evidence.append("⚠️ Recommend waiting for better setup")
        
        return Argument(
            agent=self.role,
            claim=claim,
            evidence=evidence if evidence else ["No significant bullish factors"],
            confidence=confidence,
            weight=1.0
        )


class BearAgent:
    """
    Bear Agent: Argues for caution and risk management.
    Focuses on red flags, volatility, negative sentiment, anomalies.
    """
    
    def __init__(self, anomaly_detector=None):
        self.role = AgentRole.BEAR
        self.anomaly_detector = anomaly_detector
    
    def argue(
        self,
        df: pd.DataFrame,
        sentiment_score: float,
        model_prob: float,
        anomaly_result: Dict = None
    ) -> Argument:
        """
        Build bearish argument (risk perspective).
        
        Args:
            df: Market data DataFrame
            sentiment_score: Sentiment score
            model_prob: Model probability
            anomaly_result: Anomaly detection results
            
        Returns:
            Risk-focused argument
        """
        evidence = []
        risk_points = 0
        
        # 1. Check for anomalies (RED FLAGS)
        if anomaly_result and anomaly_result.get('is_anomaly'):
            severity = anomaly_result.get('severity', 'UNKNOWN')
            evidence.append(f"🔴 ANOMALY DETECTED: {severity} risk")
            for flag in anomaly_result.get('red_flags', []):
                evidence.append(f"  ⚠️ {flag}")
            risk_points += 3
        
        # 2. Check negative sentiment
        if sentiment_score < -0.2:
            evidence.append(f"🔴 Strong negative sentiment: {sentiment_score:+.3f}")
            risk_points += 2
        elif sentiment_score < 0:
            evidence.append(f"🟡 Negative sentiment: {sentiment_score:+.3f}")
            risk_points += 1
        
        # 3. Check model bearishness
        if model_prob < 0.3:
            evidence.append(f"🔴 Model predicts downside: {(1-model_prob)*100:.1f}% DOWN probability")
            risk_points += 2
        elif model_prob < 0.5:
            evidence.append(f"🟡 Model leans bearish: {model_prob*100:.1f}% UP probability")
            risk_points += 1
        
        # 4. Check volatility
        if len(df) >= 20:
            recent_vol = df['Close'].pct_change().tail(20).std()
            if recent_vol > 0.03:
                evidence.append(f"🔴 High volatility: {recent_vol*100:.2f}% daily std dev")
                risk_points += 2
            elif recent_vol > 0.02:
                evidence.append(f"🟡 Elevated volatility: {recent_vol*100:.2f}% daily std dev")
                risk_points += 1
        
        # 5. Check overbought RSI
        if 'RSI_14' in df.columns:
            latest_rsi = df['RSI_14'].iloc[-1]
            if latest_rsi > 70:
                evidence.append(f"🔴 Overbought: RSI = {latest_rsi:.1f}")
                risk_points += 2
            elif latest_rsi > 65:
                evidence.append(f"🟡 Near overbought: RSI = {latest_rsi:.1f}")
                risk_points += 1
        
        # 6. Check recent losses
        if len(df) >= 5:
            recent_returns = df['Close'].pct_change().tail(5)
            losing_days = (recent_returns < 0).sum()
            if losing_days >= 4:
                evidence.append(f"🔴 Downtrend: {losing_days}/5 losing days")
                risk_points += 2
            elif losing_days >= 3:
                evidence.append(f"🟡 Weakness: {losing_days}/5 losing days")
                risk_points += 1
        
        # 7. Check volume anomaly
        if len(df) >= 20:
            latest_volume = df['Volume'].iloc[-1]
            avg_volume = df['Volume'].tail(20).mean()
            if latest_volume > avg_volume * 3:
                evidence.append(f"🔴 Extreme volume spike: {latest_volume/avg_volume:.1f}x (potential manipulation)")
                risk_points += 2
        
        # Formulate claim
        if risk_points >= 6:
            claim = "STRONG SELL/AVOID: Critical risk factors present. Do NOT trade."
            confidence = 0.95
        elif risk_points >= 4:
            claim = "HIGH RISK: Multiple warning signs. Avoid or reduce exposure."
            confidence = 0.8
        elif risk_points >= 2:
            claim = "MODERATE RISK: Some concerns present. Reduce position size."
            confidence = 0.6
        else:
            claim = "ACCEPTABLE RISK: No major red flags detected."
            confidence = 0.3
            evidence.append("✅ Risk profile within normal bounds")
        
        return Argument(
            agent=self.role,
            claim=claim,
            evidence=evidence if evidence else ["No significant risk factors"],
            confidence=confidence,
            weight=1.2  # Bear agent has slightly higher weight (risk management priority)
        )


class ModeratorAgent:
    """
    Moderator Agent: Synthesizes all arguments into final consensus.
    Balances bull/bear perspectives with model predictions.
    """
    
    def __init__(self):
        self.role = AgentRole.MODERATOR
    
    def deliberate(
        self,
        analyst_arg: Argument,
        bull_arg: Argument,
        bear_arg: Argument,
        capital: float = 100000
    ) -> Consensus:
        """
        Synthesize arguments into consensus decision.
        
        Args:
            analyst_arg: Analyst's argument
            bull_arg: Bull's argument
            bear_arg: Bear's argument
            capital: Available capital
            
        Returns:
            Consensus decision
        """
        # Calculate weighted scores
        bull_score = bull_arg.confidence * bull_arg.weight
        bear_score = bear_arg.confidence * bear_arg.weight
        analyst_score = analyst_arg.confidence * analyst_arg.weight
        
        # Normalize scores
        total_weight = bull_arg.weight + bear_arg.weight + analyst_arg.weight
        bull_normalized = bull_score / total_weight
        bear_normalized = bear_score / total_weight
        
        # Decision logic
        net_sentiment = bull_normalized - bear_normalized
        
        # Critical risk override
        if bear_arg.confidence >= 0.9:
            action = "WAIT"
            confidence = bear_arg.confidence
            position_size = 0.0
            reasoning = f"RISK OVERRIDE: {bear_arg.claim}\n\nBear Agent vetoes trade due to critical risks."
        
        # Strong bull consensus
        elif net_sentiment > 0.3 and analyst_arg.confidence > 0.5:
            action = "BUY"
            confidence = min(bull_arg.confidence, analyst_arg.confidence)
            # Position size based on confidence
            kelly_fraction = min(confidence * 0.1, 0.05)  # Max 5% of capital
            position_size = capital * kelly_fraction
            reasoning = f"CONSENSUS BUY: {bull_arg.claim}\n\nBull argument dominates. Analyst supports with {analyst_arg.confidence*100:.0f}% confidence."
        
        # Moderate bull with acceptable risk
        elif net_sentiment > 0.1 and bear_arg.confidence < 0.6:
            action = "BUY"
            confidence = 0.6
            kelly_fraction = 0.02  # Conservative 2%
            position_size = capital * kelly_fraction
            reasoning = f"CAUTIOUS BUY: {bull_arg.claim}\n\nBullish tilt but risks noted. Position size reduced."
        
        # Bear dominance
        elif net_sentiment < -0.2:
            action = "SHORT"
            confidence = bear_arg.confidence
            kelly_fraction = min(confidence * 0.05, 0.03)
            position_size = capital * kelly_fraction
            reasoning = f"SHORT SIGNAL: {bear_arg.claim}\n\nRisk factors dominate. Consider short position or protective puts."
        
        # Neutral / Wait
        else:
            action = "WAIT"
            confidence = 0.5
            position_size = 0.0
            reasoning = f"NO CONSENSUS: Bull and Bear arguments balanced.\n\n{bull_arg.claim}\n\nVS\n\n{bear_arg.claim}\n\nWait for clearer signal."
        
        return Consensus(
            action=action,
            confidence=confidence,
            position_size=position_size,
            bull_score=bull_normalized,
            bear_score=bear_normalized,
            arguments=[analyst_arg, bull_arg, bear_arg],
            final_reasoning=reasoning
        )


class DebateSystem:
    """
    Complete Agentic Debate System.
    Orchestrates multi-agent deliberation for trading decisions.
    """
    
    def __init__(self, model, anomaly_detector=None):
        """
        Initialize debate system.
        
        Args:
            model: Trained BiLSTM+Attention model
            anomaly_detector: Anomaly detection system
        """
        self.analyst = AnalystAgent(model)
        self.bull = BullAgent()
        self.bear = BearAgent(anomaly_detector)
        self.moderator = ModeratorAgent()
        self.anomaly_detector = anomaly_detector
    
    def run_debate(
        self,
        ticker: str,
        df: pd.DataFrame,
        price_data: np.ndarray,
        sentiment_data: np.ndarray,
        sentiment_score: float,
        capital: float = 100000
    ) -> Consensus:
        """
        Run complete debate protocol.
        
        Args:
            ticker: Stock ticker
            df: Market data DataFrame
            price_data: Price sequences for model
            sentiment_data: Sentiment features for model
            sentiment_score: Raw sentiment score
            capital: Available capital
            
        Returns:
            Final consensus
        """
        print(f"\n{'='*80}")
        print(f"🎯 AGENTIC DEBATE: {ticker}")
        print(f"{'='*80}")
        
        # Step 1: Analyst runs model
        analyst_arg = self.analyst.analyze(price_data, sentiment_data, ticker)
        model_prob = float(self.analyst.model.predict(
            price_data.reshape(1, *price_data.shape),
            sentiment_data.reshape(1, *sentiment_data.shape)
        )[0][0])
        
        print(f"\n📊 ANALYST ({analyst_arg.confidence*100:.0f}% confidence):")
        print(f"   {analyst_arg.claim}")
        for ev in analyst_arg.evidence:
            print(f"   {ev}")
        
        # Step 2: Check for anomalies
        anomaly_result = None
        if self.anomaly_detector:
            anomaly_result = self.anomaly_detector.check_latest(df, lookback=5)
        
        # Step 3: Bull argues for growth
        bull_arg = self.bull.argue(df, sentiment_score, model_prob, analyst_arg)
        print(f"\n🐂 BULL AGENT ({bull_arg.confidence*100:.0f}% confidence):")
        print(f"   {bull_arg.claim}")
        for ev in bull_arg.evidence:
            print(f"   {ev}")
        
        # Step 4: Bear argues for caution
        bear_arg = self.bear.argue(df, sentiment_score, model_prob, anomaly_result)
        print(f"\n🐻 BEAR AGENT ({bear_arg.confidence*100:.0f}% confidence):")
        print(f"   {bear_arg.claim}")
        for ev in bear_arg.evidence:
            print(f"   {ev}")
        
        # Step 5: Moderator synthesizes
        consensus = self.moderator.deliberate(analyst_arg, bull_arg, bear_arg, capital)
        
        print(f"\n⚖️ MODERATOR CONSENSUS:")
        print(f"{'='*80}")
        print(f"ACTION: {consensus.action}")
        print(f"CONFIDENCE: {consensus.confidence*100:.0f}%")
        print(f"POSITION SIZE: ₹{consensus.position_size:,.0f} ({consensus.position_size/capital*100:.1f}% of capital)")
        print(f"\nREASONING:")
        print(consensus.final_reasoning)
        print(f"{'='*80}")
        
        return consensus
