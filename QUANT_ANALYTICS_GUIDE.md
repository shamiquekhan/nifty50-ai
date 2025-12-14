# Advanced Quantitative Analytics Guide

## Overview
The dashboard now includes **Advanced Quant Analytics** section that showcases sophisticated quantitative finance concepts recruiters look for.

---

## 🎯 What's New

### 1. **Return Distribution & Fat Tails Analysis**
**Concepts Demonstrated:**
- **Skewness**: Measures asymmetry in return distribution
  - Negative skew = Large losses more likely than large gains
  - Positive skew = Large gains more likely than large losses
  
- **Excess Kurtosis**: Detects fat tails (extreme events)
  - Kurtosis > 3 (excess > 0) = Fat tails → More extreme moves than normal distribution
  - Normal distribution has kurtosis = 3
  
- **Jarque-Bera Test**: Statistical test for normality
  - p-value < 0.05 = Non-normal distribution (fat tails confirmed)
  
- **VaR (Value at Risk)**: Maximum expected loss at 95% confidence
- **CVaR (Conditional VaR)**: Expected loss when VaR is exceeded

**Why It Matters:**
- Markets have fat tails - extreme events occur more often than models predict
- Understanding tail risk is critical for risk management
- Normal distribution assumptions fail in real markets

---

### 2. **Volatility Clustering Detection**
**Concepts Demonstrated:**
- **Volatility Clustering**: High volatility periods follow high volatility
- **Autocorrelation of Squared Returns**: Measures clustering strength
- **Volatility of Volatility**: Second-order risk metric
- **Regime Detection**: Identifies HIGH/LOW/MODERATE volatility periods

**Why It Matters:**
- GARCH models exploit volatility clustering
- Position sizing should adapt to volatility regimes
- Options pricing requires volatility forecasting

**Interview Question You Can Now Answer:**
> *"Why do we square returns to detect volatility clustering?"*
> 
> **Answer**: "Squaring returns removes the sign (direction) and amplifies large moves. High autocorrelation in squared returns (e.g., 0.3+) indicates volatility persistence - when markets are turbulent, they stay turbulent. This violates constant volatility assumptions in Black-Scholes."

---

### 3. **Mean Reversion Testing**
**Concepts Demonstrated:**
- **ADF (Augmented Dickey-Fuller) Test**: Tests for stationarity
  - p-value < 0.05 = Stationary series (mean-reverting)
  - Stationary = Price reverts to mean over time
  
- **Half-Life**: Time for price to revert halfway to mean
  - Short half-life (< 30 days) = Strong mean reversion
  - Long half-life = Weak reversion or trending behavior

**Why It Matters:**
- Mean reversion strategies require stationary series
- Non-stationary = Use momentum strategies instead
- Half-life determines optimal holding period

**Interview Question You Can Now Answer:**
> *"How do you test if a stock is mean-reverting?"*
> 
> **Answer**: "I use the ADF test on the price series. If p-value < 0.05, I reject the null hypothesis of a unit root, confirming stationarity. Then I estimate half-life using AR(1) autocorrelation: half_life = -log(2)/log(ρ). Short half-life indicates fast mean reversion, ideal for pairs trading or stat arb."

---

### 4. **Market Regime Detection**
**Concepts Demonstrated:**
- **Trend Regime**: BULL / BEAR / SIDEWAYS
  - Based on rolling mean return direction
  
- **Volatility Regime**: HIGH / LOW / NORMAL
  - Based on rolling standard deviation vs historical average
  
- **Combined Regime**: Links trend + volatility for trading recommendations

**Why It Matters:**
- Different strategies work in different regimes
- Bull + Low Vol = Best for leveraged longs
- Bear + High Vol = Reduce exposure or hedge
- Sideways = Mean reversion opportunities

---

### 5. **Stochastic Calculus - GBM Parameters**
**Concepts Demonstrated:**
- **Geometric Brownian Motion (GBM)**: dS = μ·S·dt + σ·S·dW
  - **Drift (μ)**: Expected return direction (annualized)
  - **Diffusion (σ)**: Volatility from randomness (Brownian motion)
  
- **Sharpe Ratio**: (μ - r_f) / σ
  - Risk-adjusted return quality
  - > 1 = Good, > 2 = Excellent

**Why It Matters:**
- GBM is foundation of options pricing (Black-Scholes)
- Drift = Deterministic component (trend)
- Diffusion = Stochastic component (uncertainty)
- Understanding drift/diffusion intuition separates good quants from great ones

**Interview Question You Can Now Answer:**
> *"Explain Geometric Brownian Motion in simple terms."*
> 
> **Answer**: "GBM models stock prices as having two components: drift (μ) is the expected direction the stock moves over time, like a river current. Diffusion (σ) is the random volatility that makes prices jiggle unpredictably, like waves on water. The Brownian motion term (dW) represents pure randomness - it's why we can't predict exact prices, only distributions. This is the basis of options pricing because it captures both trend and uncertainty."

---

## 📊 Dashboard Sections Added

### Section 1: Return Distribution
- **Visual Cards**: Skewness, Excess Kurtosis, Tail Type, VaR 95%
- **Color Coding**: 
  - Red = Fat tails detected (high risk)
  - Green = Normal tails
- **Interpretation**: Automatic explanation of tail risk

### Section 2: Volatility Clustering
- **Visual Cards**: Current Vol, Avg Vol, Regime, Clustering Coefficient
- **Color Coding**:
  - Red = High volatility regime
  - Green = Low volatility regime
  - Orange = Moderate regime
- **Interpretation**: Clustering strength and risk level

### Section 3: Mean Reversion
- **Visual Cards**: ADF Statistic, P-Value, Stationarity Status, Half-Life
- **Color Coding**:
  - Green = Stationary (mean-reverting)
  - Red = Non-stationary (trending)
- **Trading Signal**: Automatic strategy recommendation

### Section 4: Market Regime
- **Visual Cards**: Trend Regime, Volatility Regime
- **Border Colors**: Match regime type
- **Recommendation Box**: Combined trading advice

### Section 5: GBM Parameters
- **Visual Cards**: Drift (μ), Diffusion (σ), Sharpe Ratio
- **Color Coding**:
  - Green = Positive drift / Good Sharpe
  - Red = Negative drift / Poor Sharpe
  - Orange = Moderate diffusion
- **Interpretation**: Explains drift/diffusion in context

---

## 🎓 Interview Edge

### What You Can Now Demonstrate:

1. **Time-Series Intuition Over Theory**
   - ✅ You understand volatility clustering, not just GARCH formulas
   - ✅ You recognize fat tails and their implications
   - ✅ You can test for mean reversion and stationarity
   - ✅ You detect regime shifts in live data

2. **Stochastic Calculus as Intuition**
   - ✅ You explain GBM drift/diffusion in simple words
   - ✅ You understand why randomness shapes pricing
   - ✅ You know Brownian motion isn't just theoretical
   - ✅ You can estimate real-world stochastic parameters

3. **Practical Application**
   - ✅ You built this analytics engine from scratch
   - ✅ You visualize complex concepts clearly
   - ✅ You connect theory to trading decisions
   - ✅ You use proper statistical tests (ADF, Jarque-Bera)

---

## 🚀 Usage

### In Dashboard:
1. Navigate to **ADVANCED QUANT ANALYTICS** section (below backtest)
2. Select any stock from the dropdown
3. See all 5 analytics sections with live calculations
4. Read interpretations and trading signals

### In Code:
```python
from src.quant_analytics import QuantAnalytics

qa = QuantAnalytics()
prices = stock_data['Close']

# Run all analyses
analysis = qa.comprehensive_analysis(prices)

# Or individual tests
returns_analysis = qa.analyze_returns(prices)
vol_clustering = qa.detect_volatility_clustering(prices)
mean_reversion = qa.test_mean_reversion(prices)
regime = qa.detect_regime_shifts(prices)
gbm = qa.estimate_gbm_parameters(prices)
```

---

## 🎯 Key Metrics Explained

| Metric | Normal Range | What It Means |
|--------|-------------|---------------|
| **Skewness** | -0.5 to +0.5 | Symmetric returns |
| **Excess Kurtosis** | -1 to +2 | Normal tails |
| **Clustering Coef** | > 0.3 | Strong volatility persistence |
| **ADF p-value** | < 0.05 | Stationary (mean-reverting) |
| **Half-Life** | < 30 days | Fast mean reversion |
| **Drift (μ)** | 0.05 - 0.20 | Reasonable annual return |
| **Diffusion (σ)** | 0.15 - 0.35 | Normal volatility |
| **Sharpe Ratio** | > 1.0 | Good risk-adjusted return |

---

## 📈 Example Interpretations

### Scenario 1: HDFCBANK
```
Return Distribution:
- Skewness: -0.342 (slight negative skew)
- Excess Kurtosis: 3.15 (FAT TAILS)
- VaR 95%: -2.34% (max daily loss at 95% confidence)

Interpretation: "Fat tails detected - expect 2-3% daily moves 
more often than normal distribution predicts. Negative skew 
means large losses slightly more likely than large gains."
```

### Scenario 2: RELIANCE
```
Volatility Clustering:
- Current Vol: 1.82%
- Avg Vol: 1.54%
- Clustering Coef: 0.387 (HIGH)
- Regime: HIGH VOLATILITY

Interpretation: "High volatility clustering - when volatility 
spikes, it persists. Current vol 1.2x average. Reduce position 
sizing in this regime."
```

### Scenario 3: INFY
```
Mean Reversion:
- ADF Statistic: -3.654
- P-Value: 0.0043 (< 0.05)
- Stationary: YES
- Half-Life: 18.3 days

Trading Signal: "STRONG MEAN REVERSION - Good for mean 
reversion strategies. Price reverts to mean in ~18 days."
```

---

## 🔥 Bottom Line

**Before this feature:**
- You could say "I know GARCH models"

**After this feature:**
- You can say "I built a volatility clustering detector that identifies regime shifts and auto-adjusts position sizing based on vol-of-vol metrics"
- You can explain why squared returns autocorrelation matters
- You can test for stationarity and estimate half-life on the fly
- You understand GBM parameters as intuition, not just formulas

**Recruiter sees:**
- Real quantitative thinking
- Practical application of theory
- Production-ready analytics code
- Clear communication of complex concepts

This is what separates portfolio-worthy projects from tutorial-following.

---

## 📚 Next Steps

1. **Test with different stocks** - See how NIFTY50 stocks vary in regimes
2. **Study the interpretations** - Understand why each metric matters
3. **Practice explaining** - Use this in interviews
4. **Extend the analytics** - Add your own metrics (e.g., Hurst exponent, Omega ratio)

---

**Built with:** Python, Pandas, NumPy, SciPy, Statsmodels  
**Deployed on:** Streamlit Dashboard  
**Location:** `src/quant_analytics.py` + Dashboard Section  
**Commit:** `4ee5ab5` - "Add Advanced Quant Analytics: volatility clustering, fat tails, mean reversion, regime shifts, GBM parameters"
