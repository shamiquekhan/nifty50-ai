"""
Advanced Quantitative Analytics Module
Time-series analysis: volatility clustering, fat tails, mean reversion, regime shifts
Stochastic calculus: GBM, drift, diffusion, Brownian motion
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


class QuantAnalytics:
    """Advanced quant analytics for trading systems."""
    
    def __init__(self):
        """Initialize quant analytics engine."""
        pass
    
    def analyze_returns(self, prices: pd.Series) -> Dict:
        """
        Analyze return distribution for fat tails, skewness, kurtosis.
        
        Args:
            prices: Price series
            
        Returns:
            Dictionary with distribution statistics
        """
        # Calculate returns
        returns = prices.pct_change().dropna()
        
        # Descriptive stats
        mean_return = returns.mean()
        volatility = returns.std()
        
        # Fat tails analysis
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)
        excess_kurtosis = kurtosis - 3  # Normal distribution has kurtosis of 3
        
        # Jarque-Bera test for normality
        jb_stat, jb_pvalue = stats.jarque_bera(returns)
        
        # VaR (Value at Risk) at 95% confidence
        var_95 = np.percentile(returns, 5)
        
        # CVaR (Conditional VaR) - Expected Shortfall
        cvar_95 = returns[returns <= var_95].mean()
        
        # Fat tail indicator
        if excess_kurtosis > 2:
            tail_type = "FAT TAILS (High Risk)"
        elif excess_kurtosis < -1:
            tail_type = "THIN TAILS (Low Risk)"
        else:
            tail_type = "NORMAL TAILS"
        
        return {
            'mean_return': mean_return,
            'volatility': volatility,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'excess_kurtosis': excess_kurtosis,
            'jb_stat': jb_stat,
            'jb_pvalue': jb_pvalue,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'tail_type': tail_type,
            'is_normal': jb_pvalue > 0.05
        }
    
    def detect_volatility_clustering(self, prices: pd.Series, window: int = 20) -> Dict:
        """
        Detect volatility clustering - periods of high/low volatility persistence.
        
        Args:
            prices: Price series
            window: Rolling window for volatility calculation
            
        Returns:
            Dictionary with clustering statistics
        """
        returns = prices.pct_change().dropna()
        
        # Rolling volatility
        rolling_vol = returns.rolling(window=window).std()
        
        # Volatility of volatility (vol clustering indicator)
        vol_of_vol = rolling_vol.std()
        
        # Current vs average volatility
        current_vol = rolling_vol.iloc[-1] if len(rolling_vol) > 0 else 0
        avg_vol = rolling_vol.mean()
        
        # Clustering coefficient (autocorrelation of squared returns)
        squared_returns = returns ** 2
        clustering_coef = squared_returns.autocorr(lag=1) if len(squared_returns) > 1 else 0
        
        # Regime detection
        if current_vol > avg_vol * 1.5:
            regime = "HIGH VOLATILITY"
            risk_level = "ELEVATED"
        elif current_vol < avg_vol * 0.7:
            regime = "LOW VOLATILITY"
            risk_level = "NORMAL"
        else:
            regime = "MODERATE VOLATILITY"
            risk_level = "STABLE"
        
        return {
            'current_vol': current_vol,
            'avg_vol': avg_vol,
            'vol_of_vol': vol_of_vol,
            'clustering_coef': clustering_coef,
            'regime': regime,
            'risk_level': risk_level,
            'vol_ratio': current_vol / avg_vol if avg_vol > 0 else 1
        }
    
    def test_mean_reversion(self, prices: pd.Series) -> Dict:
        """
        Test for mean reversion using ADF test and Hurst exponent.
        
        Args:
            prices: Price series
            
        Returns:
            Dictionary with mean reversion statistics
        """
        from statsmodels.tsa.stattools import adfuller
        
        # Augmented Dickey-Fuller test
        try:
            adf_result = adfuller(prices.dropna(), autolag='AIC')
            adf_stat = adf_result[0]
            adf_pvalue = adf_result[1]
            is_stationary = adf_pvalue < 0.05
        except:
            adf_stat = 0
            adf_pvalue = 1
            is_stationary = False
        
        # Half-life of mean reversion (for stationary series)
        returns = prices.pct_change().dropna()
        if len(returns) > 1:
            # Simple AR(1) approximation
            lagged = returns.shift(1).dropna()
            current = returns[1:]
            if len(lagged) > 0 and len(current) > 0:
                corr = np.corrcoef(lagged, current)[0, 1]
                if corr < 1 and corr > -1:
                    half_life = -np.log(2) / np.log(abs(corr))
                else:
                    half_life = np.inf
            else:
                half_life = np.inf
        else:
            half_life = np.inf
        
        # Mean reversion strength
        if is_stationary and half_life < 30:
            reversion_type = "STRONG MEAN REVERSION"
            trading_signal = "GOOD FOR MEAN REVERSION STRATEGIES"
        elif is_stationary:
            reversion_type = "WEAK MEAN REVERSION"
            trading_signal = "MODERATE FOR MEAN REVERSION"
        else:
            reversion_type = "TRENDING (NO REVERSION)"
            trading_signal = "GOOD FOR MOMENTUM STRATEGIES"
        
        return {
            'adf_stat': adf_stat,
            'adf_pvalue': adf_pvalue,
            'is_stationary': is_stationary,
            'half_life': half_life,
            'reversion_type': reversion_type,
            'trading_signal': trading_signal
        }
    
    def detect_regime_shifts(self, prices: pd.Series, lookback: int = 60) -> Dict:
        """
        Detect market regime shifts using rolling statistics.
        
        Args:
            prices: Price series
            lookback: Lookback period for regime detection
            
        Returns:
            Dictionary with regime information
        """
        returns = prices.pct_change().dropna()
        
        # Rolling mean and std
        rolling_mean = returns.rolling(window=lookback).mean()
        rolling_std = returns.rolling(window=lookback).std()
        
        # Current regime
        current_mean = rolling_mean.iloc[-1] if len(rolling_mean) > 0 else 0
        current_std = rolling_std.iloc[-1] if len(rolling_std) > 0 else 0
        
        # Trend identification
        if current_mean > 0.001:
            trend = "BULL MARKET"
        elif current_mean < -0.001:
            trend = "BEAR MARKET"
        else:
            trend = "SIDEWAYS MARKET"
        
        # Volatility regime
        avg_std = rolling_std.mean() if len(rolling_std) > 0 else 0
        if current_std > avg_std * 1.5:
            vol_regime = "HIGH VOLATILITY REGIME"
        elif current_std < avg_std * 0.7:
            vol_regime = "LOW VOLATILITY REGIME"
        else:
            vol_regime = "NORMAL VOLATILITY REGIME"
        
        # Combined regime
        regime = f"{trend} + {vol_regime}"
        
        # Trading recommendation
        if "BULL" in trend and "LOW" in vol_regime:
            recommendation = "OPTIMAL FOR LONG POSITIONS"
        elif "BEAR" in trend and "HIGH" in vol_regime:
            recommendation = "HIGH RISK - REDUCE EXPOSURE"
        elif "SIDEWAYS" in trend:
            recommendation = "RANGE-BOUND - MEAN REVERSION"
        else:
            recommendation = "MIXED SIGNALS - CAUTIOUS"
        
        return {
            'trend': trend,
            'vol_regime': vol_regime,
            'combined_regime': regime,
            'current_mean': current_mean,
            'current_std': current_std,
            'recommendation': recommendation
        }
    
    def estimate_gbm_parameters(self, prices: pd.Series) -> Dict:
        """
        Estimate Geometric Brownian Motion (GBM) parameters: drift and diffusion.
        dS = μ*S*dt + σ*S*dW
        
        Args:
            prices: Price series
            
        Returns:
            Dictionary with GBM parameters
        """
        log_returns = np.log(prices / prices.shift(1)).dropna()
        
        # Drift (μ) - annualized
        drift = log_returns.mean() * 252
        
        # Diffusion (σ) - annualized volatility
        diffusion = log_returns.std() * np.sqrt(252)
        
        # Sharpe ratio approximation
        risk_free_rate = 0.06  # Assume 6% risk-free rate
        sharpe = (drift - risk_free_rate) / diffusion if diffusion > 0 else 0
        
        # Drift interpretation
        if drift > 0.15:
            drift_type = "STRONG POSITIVE DRIFT"
        elif drift > 0:
            drift_type = "POSITIVE DRIFT"
        elif drift > -0.15:
            drift_type = "NEGATIVE DRIFT"
        else:
            drift_type = "STRONG NEGATIVE DRIFT"
        
        # Diffusion interpretation
        if diffusion > 0.40:
            vol_type = "HIGH VOLATILITY (Risky)"
        elif diffusion > 0.25:
            vol_type = "MODERATE VOLATILITY"
        else:
            vol_type = "LOW VOLATILITY (Stable)"
        
        return {
            'drift': drift,
            'diffusion': diffusion,
            'sharpe_ratio': sharpe,
            'drift_type': drift_type,
            'vol_type': vol_type,
            'annualized_return': drift,
            'annualized_vol': diffusion
        }
    
    def comprehensive_analysis(self, prices: pd.Series) -> Dict:
        """
        Run all quantitative analyses on price series.
        
        Args:
            prices: Price series
            
        Returns:
            Complete analysis dictionary
        """
        return {
            'returns_analysis': self.analyze_returns(prices),
            'volatility_clustering': self.detect_volatility_clustering(prices),
            'mean_reversion': self.test_mean_reversion(prices),
            'regime_shifts': self.detect_regime_shifts(prices),
            'gbm_parameters': self.estimate_gbm_parameters(prices)
        }
