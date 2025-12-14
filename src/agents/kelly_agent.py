"""
Kelly Criterion Risk Management Agent
Acts as a logic-based agent to determine optimal position sizing.
"""

import yaml
import numpy as np
from typing import Tuple


class KellyCriterionAgent:
    """
    Risk Management Agent using Kelly Criterion.
    
    The Kelly Criterion calculates the optimal bet size based on:
    - Win probability (from model)
    - Payoff odds
    - Sentiment confirmation
    
    This acts as an "Agentic" layer that debates whether to trade.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.risk_config = self.config['risk']
        self.min_confidence = self.risk_config['min_confidence']
        self.kelly_fraction = self.risk_config['kelly_fraction']
        self.conflict_penalty = self.risk_config['sentiment_conflict_penalty']
    
    def kelly_formula(self, p: float, b: float = 1.0) -> float:
        """
        Calculate Kelly Criterion fraction.
        
        Formula: f = (bp - q) / b
        where:
        - f = fraction of capital to bet
        - b = odds received (default 1:1 for stocks)
        - p = probability of winning
        - q = probability of losing (1-p)
        
        Args:
            p: Probability of winning (0-1)
            b: Odds received (default 1.0 for 1:1)
            
        Returns:
            Kelly fraction (0-1)
        """
        q = 1 - p
        f = (b * p - q) / b
        
        # Kelly can be negative (don't bet) or > 1 (too aggressive)
        # We'll clamp it to [0, 1]
        return max(0.0, min(1.0, f))
    
    def check_sentiment_alignment(
        self, 
        model_prob: float, 
        sentiment_score: float
    ) -> Tuple[bool, str]:
        """
        Check if model prediction aligns with sentiment.
        
        Args:
            model_prob: Model's predicted probability (0-1)
            sentiment_score: Sentiment score (-1 to +1)
            
        Returns:
            Tuple of (is_aligned, message)
        """
        # Bullish signal: model_prob > 0.5 (predicting UP)
        model_bullish = model_prob > 0.5
        
        # Sentiment bullish: sentiment_score > 0
        sentiment_bullish = sentiment_score > 0
        
        # Check alignment
        if model_bullish == sentiment_bullish:
            return True, "✅ ALIGNED: Technical and Sentiment agree"
        else:
            return False, "⚠️  CONFLICT: Technical/Sentiment divergence detected"
    
    def calculate_position_size(
        self,
        model_prob: float,
        sentiment_score: float,
        capital: float,
        ticker: str = "STOCK"
    ) -> Tuple[float, str, dict]:
        """
        Main decision function: Calculate position size using Kelly Criterion.
        
        Args:
            model_prob: Model's predicted probability of price increase (0-1)
            sentiment_score: Sentiment score from news (-1 to +1)
            capital: Available capital
            ticker: Stock ticker for logging
            
        Returns:
            Tuple of (position_size, action, debug_info)
        """
        debug_info = {
            'ticker': ticker,
            'model_prob': model_prob,
            'sentiment_score': sentiment_score,
            'capital': capital,
        }
        
        # ===== STEP 1: Confidence Gate =====
        # Model must be confident enough to trade
        if model_prob < self.min_confidence and model_prob > (1 - self.min_confidence):
            debug_info['reason'] = f"Low confidence (need >{self.min_confidence} or <{1-self.min_confidence})"
            return 0.0, "WAIT", debug_info
        
        # ===== STEP 2: Sentiment Alignment Check =====
        is_aligned, alignment_msg = self.check_sentiment_alignment(model_prob, sentiment_score)
        debug_info['alignment'] = alignment_msg
        
        confidence_multiplier = 1.0
        if not is_aligned:
            # Apply penalty when technical and sentiment disagree
            confidence_multiplier = self.conflict_penalty
            print(f"⚠️  {ticker}: {alignment_msg}")
            print(f"   Reducing position size to {self.conflict_penalty*100}%")
        
        # ===== STEP 3: Calculate Kelly Fraction =====
        # Determine if we're betting on UP or DOWN
        if model_prob > 0.5:
            # Betting on price increase
            p_win = model_prob
            action_direction = "BUY"
        else:
            # Betting on price decrease (short)
            p_win = 1 - model_prob
            action_direction = "SHORT"
        
        # Calculate raw Kelly fraction
        kelly_f = self.kelly_formula(p_win)
        
        # Apply Half-Kelly for safety (or custom fraction)
        safe_kelly = kelly_f * self.kelly_fraction
        
        # Apply sentiment penalty if conflicted
        final_kelly = safe_kelly * confidence_multiplier
        
        debug_info['kelly_raw'] = kelly_f
        debug_info['kelly_safe'] = safe_kelly
        debug_info['kelly_final'] = final_kelly
        debug_info['confidence_multiplier'] = confidence_multiplier
        
        # ===== STEP 4: Calculate Position Size =====
        position_size = final_kelly * capital
        
        # Ensure minimum position size (e.g., at least $100 or 0)
        min_position = 100
        if position_size < min_position:
            debug_info['reason'] = f"Position size too small (<${min_position})"
            return 0.0, "WAIT", debug_info
        
        debug_info['position_size'] = position_size
        debug_info['action'] = action_direction
        
        return position_size, action_direction, debug_info
    
    def get_recommendation(
        self,
        model_prob: float,
        sentiment_score: float,
        capital: float,
        ticker: str = "STOCK",
        verbose: bool = True
    ) -> dict:
        """
        Get complete trading recommendation with explanation.
        
        Args:
            model_prob: Model prediction probability
            sentiment_score: Sentiment score
            capital: Available capital
            ticker: Stock ticker
            verbose: Print detailed explanation
            
        Returns:
            Dictionary with recommendation details
        """
        position_size, action, debug_info = self.calculate_position_size(
            model_prob, sentiment_score, capital, ticker
        )
        
        recommendation = {
            'ticker': ticker,
            'action': action,
            'position_size': position_size,
            'position_percent': (position_size / capital * 100) if capital > 0 else 0,
            'model_probability': model_prob,
            'sentiment_score': sentiment_score,
            'kelly_fraction': debug_info.get('kelly_final', 0),
            'is_aligned': 'ALIGNED' in debug_info.get('alignment', ''),
            'debug': debug_info
        }
        
        if verbose:
            self._print_recommendation(recommendation)
        
        return recommendation
    
    def _print_recommendation(self, rec: dict):
        """Print formatted recommendation."""
        print("\n" + "=" * 60)
        print(f"🤖 KELLY AGENT RECOMMENDATION: {rec['ticker']}")
        print("=" * 60)
        print(f"📊 Model Probability: {rec['model_probability']:.2%}")
        print(f"📰 Sentiment Score:   {rec['sentiment_score']:+.3f}")
        print(f"🎯 Action:            {rec['action']}")
        print(f"💰 Position Size:     ${rec['position_size']:,.2f} ({rec['position_percent']:.1f}% of capital)")
        print(f"📈 Kelly Fraction:    {rec['kelly_fraction']:.2%}")
        print(f"🔄 Alignment:         {'✅ YES' if rec['is_aligned'] else '⚠️  NO'}")
        
        if 'reason' in rec['debug']:
            print(f"💡 Reason:            {rec['debug']['reason']}")
        
        print("=" * 60)


def main():
    """Test the Kelly Agent."""
    agent = KellyCriterionAgent()
    
    capital = 100000  # $100k capital
    
    # Test Case 1: High confidence + aligned sentiment (STRONG BUY)
    print("\n🧪 Test Case 1: High Confidence + Positive Sentiment")
    agent.get_recommendation(
        model_prob=0.75,      # 75% chance of price increase
        sentiment_score=0.6,  # Positive sentiment
        capital=capital,
        ticker="RELIANCE"
    )
    
    # Test Case 2: High confidence but conflicting sentiment (CAUTIOUS BUY)
    print("\n🧪 Test Case 2: High Confidence + Negative Sentiment (CONFLICT)")
    agent.get_recommendation(
        model_prob=0.70,      # 70% chance of price increase
        sentiment_score=-0.4, # Negative sentiment (CONFLICT!)
        capital=capital,
        ticker="TCS"
    )
    
    # Test Case 3: Low confidence (NO TRADE)
    print("\n🧪 Test Case 3: Low Confidence")
    agent.get_recommendation(
        model_prob=0.55,      # 55% - too uncertain
        sentiment_score=0.2,
        capital=capital,
        ticker="INFY"
    )
    
    # Test Case 4: Strong SHORT signal
    print("\n🧪 Test Case 4: Strong SHORT Signal")
    agent.get_recommendation(
        model_prob=0.20,      # 20% chance of increase = 80% chance of decrease
        sentiment_score=-0.7, # Very negative sentiment
        capital=capital,
        ticker="SBIN"
    )


if __name__ == "__main__":
    main()
