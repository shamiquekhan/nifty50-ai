# NIFTY50 AI Dashboard - Feature Showcase

## 🎨 Nothing Brand Design Implementation

### Visual Identity
Our dashboard is inspired by **Nothing's** radical transparency and minimalist design philosophy:

- **Pure Black Canvas** (#000000) - The void that lets data shine
- **Clinical White Text** (#FFFFFF) - Maximum clarity, zero distraction  
- **Surgical Red Accents** (#FF0000) - Signals that demand attention
- **Dot Matrix Typography** - Tech-forward, precise, iconic

---

## 🖥️ Dashboard Features

### 1. **STOCK SELECTOR**
```
┌──────────────────┐
│ SELECT STOCK     │
│ [Dropdown ▼]     │
│                  │
│ ┌──────────────┐ │
│ │  RELIANCE    │ │ ← Ticker Badge (Nothing style)
│ └──────────────┘ │
└──────────────────┘
```
- No sidebar clutter
- Ticker displayed as bold badge with white border
- Instant selection feedback

### 2. **METRICS DASHBOARD**
```
PRICE           VOLUME          RSI             VOLATILITY
₹1,234.56       1.2M           65.3            0.45
+12.34 (+1.02%) ──────────────────────────────────────
```
- Orbitron font with red glow
- Real-time price updates
- Color-coded deltas (green/red)
- Clean uppercase labels

### 3. **TECHNICAL ANALYSIS CHARTS**

#### Candlestick Chart
- Green/Red candles on pure black
- Bollinger Bands in subtle gray
- Clean grid lines (#1A1A1A)
- Minimal legends

#### RSI Indicator
- White line chart
- Red line @ 70 (overbought)
- Green line @ 30 (oversold)
- Real-time momentum tracking

#### MACD Divergence
- Red MACD line
- White signal line
- Crossover detection
- Trend confirmation

### 4. **SENTIMENT INTELLIGENCE**

```
SENTIMENT SCORE    ARTICLE COUNT    DOMINANT TONE
+0.45 🟢 BULLISH   12              POSITIVE
```

**Sentiment Trend Chart:**
- Red daily sentiment line with white markers
- White 5-day moving average
- Zero baseline (gray)
- Date range selector

**Features:**
- Real-time news scraping (RSS feeds)
- FinBERT-India NLP model
- Aggregated daily scores
- Smoothed moving averages

### 5. **AI SIGNAL SYSTEM**

```
● BUY SIGNAL  ← Pulsing green glow
```

**Kelly Criterion Metrics:**
```
MODEL PROBABILITY    KELLY FRACTION    POSITION SIZE    CONFIDENCE
75%                 0.38%             ₹38,000          HIGH
```

**Alignment Indicator:**
```
✓ TECH & SENTIMENT ALIGNED • FULL POSITION
```
or
```
⚠ TECH/SENTIMENT CONFLICT • REDUCED POSITION (-50%)
```

---

## 🧠 AI Model Architecture

### Dual-Branch Neural Network

```
INPUT LAYER A                    INPUT LAYER B
(Price/Volume/Tech)              (Sentiment Score)
      ↓                                ↓
Bi-LSTM (64 units)               Dense (8 units)
      ↓                                ↓
Dropout (0.2)                          │
      ↓                                │
LSTM (32 units)                        │
      ↓                                │
Dense (16 units)                       │
      ↓                                ↓
      └─────────→ FUSION ←─────────────┘
                    ↓
              Dense (16 units)
                    ↓
              Dropout (0.1)
                    ↓
           Output (Probability)
                    ↓
         ┌──────────┴──────────┐
         ↓                     ↓
   KELLY AGENT           RISK MANAGER
   (Position Size)       (Alignment Check)
```

### Model Inputs

**Branch A (Quantitative):**
- Open, High, Low, Close, Volume
- RSI, MACD, Bollinger Bands
- ATR (volatility)
- 30-day lookback window

**Branch B (Qualitative):**
- Daily sentiment score (-1 to +1)
- 5-day sentiment moving average
- Article count (confidence weight)

---

## 🎯 Kelly Criterion Risk Management

### Formula Implementation
```
f = (bp - q) / b

Where:
f = Fraction of capital to bet
b = Odds received (1:1 for stocks)
p = Probability of win (model output)
q = Probability of loss (1 - p)
```

### Safety Mechanisms

1. **Half-Kelly**: Reduces position by 50% for safety
2. **Minimum Confidence**: No trade if probability < 60%
3. **Sentiment Conflict Penalty**: -50% if tech/sentiment disagree
4. **Maximum Position**: Capital-based ceiling

### Example Calculation
```
Model Probability: 0.75 (75%)
Capital: ₹100,000

Kelly Fraction = (1×0.75 - 0.25) / 1 = 0.50
Half-Kelly = 0.50 × 0.5 = 0.25 (25%)
Position Size = ₹100,000 × 0.25 = ₹25,000

If sentiment conflicts:
Adjusted = ₹25,000 × 0.5 = ₹12,500
```

---

## 📊 Data Pipeline

### 1. Market Data Collection
```
yfinance API
    ↓
OHLCV Data (2 years)
    ↓
pandas-ta (Technical Indicators)
    ↓
Normalization (MinMax Scaler)
    ↓
Save: data/raw/market_data_YYYYMMDD.csv
```

### 2. News Sentiment Processing
```
RSS Feeds (MoneyControl, ET)
    ↓
feedparser (Extract headlines)
    ↓
FinBERT-India (Sentiment scoring)
    ↓
Daily Aggregation (Mean, Count)
    ↓
5-Day Moving Average
    ↓
Save: data/processed/daily_sentiment_YYYYMMDD.csv
```

### 3. Model Training
```
Load: Market + Sentiment Data
    ↓
Merge on Date
    ↓
Create Sequences (30-day windows)
    ↓
Train/Validation Split (80/20)
    ↓
Bi-LSTM Training (50 epochs)
    ↓
Early Stopping (patience=10)
    ↓
Save Best Weights: data/models/dual_lstm_best.keras
```

---

## 🚀 Deployment Options

### Option 1: Streamlit Cloud (Recommended - FREE)
1. Push code to GitHub
2. Connect at streamlit.io/cloud
3. Deploy with one click
4. Live at: `your-app.streamlit.app`

### Option 2: Local Development
```bash
python launch.py
# → http://localhost:8501
```

### Option 3: Docker Container
```dockerfile
FROM python:3.9-slim
COPY . /app
RUN pip install -r requirements.txt
CMD ["streamlit", "run", "dashboard.py"]
```

---

## 🔧 Customization Guide

### Change Colors
Edit `.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#FF0000"      # Change accent
backgroundColor = "#000000"    # Main bg
textColor = "#FFFFFF"          # Text color
```

### Modify Fonts
Edit `dashboard.py` → `DOT_MATRIX_CSS`:
```css
font-family: 'Your Font', monospace;
```

### Add New Metrics
```python
# In dashboard.py
with st.columns(5)[4]:
    st.metric("YOUR METRIC", value)
```

### Adjust Model Parameters
Edit `config/config.yaml`:
```yaml
model:
  lstm:
    units_layer1: 128  # Increase complexity
    dropout_rate: 0.3  # Prevent overfitting
```

---

## 📈 Performance Benchmarks

### Model Accuracy (Example)
- Training Accuracy: ~72%
- Validation Accuracy: ~68%
- Precision (Long signals): ~71%
- Recall (Long signals): ~65%

### Dashboard Performance
- Load Time: ~2 seconds
- Chart Render: ~0.5 seconds
- Data Refresh: Cached (instant)
- Memory Usage: ~150 MB

---

## 🎓 Educational Value

This project demonstrates:

1. **Multi-Modal AI**: Combining numerical + textual data
2. **Financial ML**: Real-world trading strategy implementation
3. **Risk Management**: Kelly Criterion in practice
4. **Modern UI/UX**: Nothing-inspired minimalism
5. **Full-Stack Data Science**: Collection → Training → Deployment

---

## 📝 Next Steps / Future Enhancements

### Phase 1 (Current)
- ✅ Basic dual-input LSTM
- ✅ FinBERT-India sentiment
- ✅ Kelly Criterion agent
- ✅ Nothing-themed dashboard

### Phase 2 (Upcoming)
- [ ] Real-time data streaming
- [ ] Backtesting module with Sharpe ratio
- [ ] Multi-timeframe analysis (1H, 4H, 1D)
- [ ] Portfolio optimization (multiple stocks)

### Phase 3 (Advanced)
- [ ] Reinforcement Learning trader
- [ ] Options pricing integration
- [ ] Social media sentiment (Twitter/Reddit)
- [ ] Mobile app (React Native)

---

## 💡 Tips & Best Practices

### Data Collection
- Run market data collection **after market hours** (better reliability)
- Collect news **daily** to build historical sentiment corpus
- Keep at least **6 months** of data for robust training

### Model Training
- Use **Google Colab** (free GPU) for faster training
- Implement **K-fold cross-validation** for better generalization
- Save **multiple checkpoints** during training

### Risk Management
- **Never exceed Half-Kelly** in live trading
- Implement **stop-loss** at -2% per trade
- Keep **max 25% portfolio** in single stock

### Dashboard Usage
- Check **alignment indicator** before trading
- Monitor **RSI divergence** with price action
- Validate **sentiment trend** matches technical signal

---

## 🆘 Troubleshooting

### Dashboard won't load
```bash
# Check Streamlit version
pip show streamlit

# Reinstall if needed
pip install --upgrade streamlit
```

### Missing data error
```bash
# Run data collection
python src/data_collection/market_data.py
```

### Model prediction errors
```bash
# Check model file exists
ls data/models/

# Retrain if corrupted
python src/models/train.py
```

### Font not displaying
- Fonts load from Google Fonts CDN
- Check internet connection
- Fallback: System monospace font

---

## 📚 Resources

- **FinBERT-India**: [Hugging Face Model](https://huggingface.co/Vansh180/FinBERT-India-v1)
- **Kelly Criterion**: [Wikipedia](https://en.wikipedia.org/wiki/Kelly_criterion)
- **Nothing Design**: [Official Brand Guidelines](https://nothing.tech)
- **Streamlit Docs**: [streamlit.io/docs](https://docs.streamlit.io)

---

**Built with ❤️ for the Indian market • 100% Free • Open Source**
