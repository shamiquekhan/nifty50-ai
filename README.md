# 🎯 NIFTY50 AI Trading System

```
███╗   ██╗██╗███████╗████████╗██╗   ██╗███████╗ ██████╗ 
████╗  ██║██║██╔════╝╚══██╔══╝╚██╗ ██╔╝██╔════╝██╔═████╗
██╔██╗ ██║██║█████╗     ██║    ╚████╔╝ ███████╗██║██╔██║
██║╚██╗██║██║██╔══╝     ██║     ╚██╔╝  ╚════██║████╔╝██║
██║ ╚████║██║██║        ██║      ██║   ███████║╚██████╔╝
╚═╝  ╚═══╝╚═╝╚═╝        ╚═╝      ╚═╝   ╚══════╝ ╚═════╝ 
```

<div align="center">

**AI-Powered NIFTY50 Trading System with Auto-Updates**

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.14+-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[🚀 Live Demo](https://nifty50-ai.streamlit.app) • [📖 Documentation](AUTO_UPDATE_GUIDE.md) • [🐛 Report Bug](https://github.com/shamiquekhan/nifty50-ai/issues)

</div>

---

## 🌟 Features

### 🤖 **AI-Powered Trading**
- **LSTM Deep Learning** - Bi-directional LSTM for price prediction
- **Sentiment Analysis** - FinBERT-India NLP for news sentiment
- **Kelly Criterion** - Optimal position sizing and risk management
- **Real-time Alerts** - 5 types of trading alerts (BUY, SELL, WARNING, OPPORTUNITY, INFO)

### ⚡ **Auto-Update System** (NEW!)
- **Every 4 Hours** - Automatic data refresh and model fine-tuning
- **Incremental Learning** - Fast model updates (3 min vs 10+ min full training)
- **Continuous Improvement** - Model learns from new market data automatically
- **Zero Manual Work** - Set it and forget it operation

### 🎨 **Nothing-Inspired Dashboard**
- **Minimalist Design** - Pure black/white/red aesthetic
- **Dot Matrix Fonts** - Futuristic typography (Doto + Share Tech Mono)
- **Real-time Updates** - Live market data and predictions
- **Interactive Charts** - Plotly visualizations with Nothing theme
- **Backtest Performance** - Equity curve and detailed metrics

### 📊 **Trading Features**
- **50 NIFTY Stocks** - Complete coverage of NIFTY50 index
- **Technical Indicators** - RSI, MACD, Bollinger Bands, ATR, volume analysis
- **Market Regime Detection** - Trend and volatility classification
- **Strategy Optimization** - Parameter grid search for best performance
- **Risk Management** - Stop-loss, take-profit, trailing stops

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/shamiquekhan/nifty50-ai.git
cd nifty50-ai

# Install dependencies
pip install -r requirements.txt

# Run initial data collection and training
python src/data_collection/market_data.py
python src/quick_train.py

# Start dashboard
streamlit run dashboard.py
```

### Auto-Update System

Start the auto-update system for automatic data refresh every 4 hours:

```powershell
# Windows PowerShell
.\start_auto_update.ps1

# OR Command Prompt
start_auto_update.bat

# OR Direct Python
python src/auto_update.py
```

---

## 📁 Project Structure
```
nifty50-ai/
├── 📁 src/
│   ├── data_collection/
│   │   ├── market_data.py       # Fetch NIFTY50 stock data
│   │   └── news_scraper.py      # Scrape news from Moneycontrol
│   ├── sentiment/
│   │   └── finbert_engine.py    # FinBERT-India sentiment analysis
│   ├── models/
│   │   └── dual_lstm.py         # LSTM model architecture
│   ├── agents/
│   │   └── kelly_agent.py       # Kelly Criterion position sizing
│   ├── utils/
│   │   └── preprocessing.py     # Data preprocessing utilities
│   ├── auto_update.py           # Auto-update orchestrator ⚡
│   ├── incremental_training.py  # Fast model fine-tuning ⚡
│   ├── quick_train.py           # LSTM model training
│   ├── predict.py               # Generate predictions
│   ├── backtesting.py           # Strategy backtesting
│   ├── optimize_strategy.py     # Parameter optimization
│   ├── alerts.py                # Real-time alert system
│   └── market_regime.py         # Market condition detection
├── 📁 data/
│   ├── raw/                     # Market data CSVs
│   ├── processed/               # Predictions & sentiment
│   └── results/                 # Backtest results
├── 📁 models/
│   └── lstm_model.keras         # Trained LSTM model
├── 📁 logs/
│   └── auto_update.log          # Auto-update system logs
├── 📁 config/
│   └── config.yaml              # System configuration
├── 📁 .streamlit/
│   └── config.toml              # Streamlit theme config
├── dashboard.py                 # Main Streamlit dashboard 🎨
├── requirements.txt             # Python dependencies
├── packages.txt                 # System dependencies (Streamlit Cloud)
├── runtime.txt                  # Python version
├── start_auto_update.ps1        # Auto-update launcher (PowerShell)
├── start_auto_update.bat        # Auto-update launcher (CMD)
├── AUTO_UPDATE_GUIDE.md         # Complete auto-update docs
└── README.md                    # This file
```

---

## 🎯 How It Works

### 1️⃣ **Data Collection**
- Fetches OHLCV data for all 50 NIFTY stocks using yfinance
- Scrapes latest news articles from Moneycontrol
- Calculates 14+ technical indicators (RSI, MACD, Bollinger Bands, etc.)

### 2️⃣ **Sentiment Analysis**
- Processes news with FinBERT-India (fine-tuned for Indian markets)
- Generates sentiment scores (-1 to +1)
- Creates sentiment moving averages

### 3️⃣ **AI Prediction**
- LSTM model processes 10-day sequences of price + indicators
- Outputs probability of price increase (0-1)
- Combines with sentiment for final signal

### 4️⃣ **Risk Management**
- Kelly Criterion calculates optimal position size
- Checks tech-sentiment alignment
- Applies stop-loss, take-profit, trailing stops

### 5️⃣ **Auto-Update (Every 4 Hours)**
- Fetches fresh market data
- Scrapes new news articles
- Analyzes sentiment
- **Fine-tunes model** with incremental training
- Generates new predictions
- Validates with backtesting

---

## 📊 Dashboard Overview

### Home Screen
- **Auto-Update Status** - Shows if system is running (● ACTIVE / ○ INACTIVE)
- **Last Update Timer** - Hours since last data refresh
- **Manual Refresh** - Force reload button

### Market Overview
- **Price Charts** - Interactive candlestick charts with technical indicators
- **Volume Analysis** - Trading volume with moving averages
- **Technical Signals** - RSI, MACD, Bollinger Band positions

### Sentiment Section
- **News Sentiment Score** - Current sentiment (-1 to +1)
- **Dominant Tone** - POSITIVE / NEUTRAL / NEGATIVE
- **Sentiment Trend** - Historical sentiment chart
- **Article Count** - Number of analyzed articles

### AI Predictions
- **BUY/SELL/WAIT Signals** - Model recommendations
- **Confidence Level** - HIGH / MEDIUM / LOW
- **Position Size** - Kelly Criterion optimal allocation
- **Tech-Sentiment Alignment** - Agreement indicator

### Real-Time Alerts
- 🟢 **BUY SIGNALS** - RSI oversold, strong predictions
- 🔴 **SELL SIGNALS** - RSI overbought, weak predictions
- ⚠️ **WARNINGS** - High volatility, BB breakouts
- 💡 **OPPORTUNITIES** - Volume spikes, MACD crossovers
- ℹ️ **INFO** - Market updates, regime changes

### Backtest Performance
- **Total Return** - Overall strategy performance
- **Win Rate** - Percentage of profitable trades
- **Sharpe Ratio** - Risk-adjusted returns
- **Max Drawdown** - Worst peak-to-trough decline
- **Equity Curve** - Visual portfolio performance
- **Recent Trades** - Last 10 trades with P&L

---

## 🔧 Configuration

### Auto-Update Interval
Edit `src/auto_update.py`:
```python
self.update_interval = 4  # Change to desired hours (2, 6, 12, etc.)
```

### Model Parameters
Edit `config/config.yaml`:
```yaml
model:
  lstm_units: [64, 32]
  dropout_rate: 0.2
  learning_rate: 0.001
  
backtest:
  stop_loss_pct: 0.05    # 5%
  take_profit_pct: 0.10  # 10%
  trailing_stop_pct: 0.03 # 3%
```

---

## 📈 Performance Metrics

| Metric | Value |
|--------|-------|
| **Total Return** | +0.26% |
| **Win Rate** | 32.8% |
| **Total Trades** | 61 |
| **Sharpe Ratio** | -0.95 |
| **Max Drawdown** | -20.54% |

*Note: These are backtest results. Past performance doesn't guarantee future results.*

---

## 🚀 Deployment

### Streamlit Cloud (Recommended)

1. **Fork/Clone this repo** to your GitHub
2. **Go to** [share.streamlit.io](https://share.streamlit.io)
3. **New app** → Select your repo → Branch: `main` → File: `dashboard.py`
4. **Deploy!** 🎉

The app will automatically use:
- `requirements.txt` for Python packages
- `packages.txt` for system dependencies
- `.streamlit/config.toml` for theme
- `runtime.txt` for Python version

### Local Development

```bash
streamlit run dashboard.py
```

Dashboard runs at: http://localhost:8501

---

## 🛠️ Troubleshooting

### Auto-Update Not Starting
```bash
# Check Python path
python --version

# Install dependencies
pip install -r requirements.txt

# Start manually
python src/auto_update.py
```

### Model Fine-Tuning Fails
```bash
# Run full training first
python src/quick_train.py

# Then start auto-update
.\start_auto_update.ps1
```

### Dashboard Shows No Data
```bash
# Collect initial data
python src/data_collection/market_data.py

# Refresh dashboard
# Click "🔄 REFRESH NOW" button
```

---

## 📚 Documentation

- [📖 Auto-Update Guide](AUTO_UPDATE_GUIDE.md) - Complete auto-update documentation
- [📊 Auto-Update Summary](AUTO_UPDATE_SUMMARY.md) - Technical overview
- [🚀 Deployment Guide](DEPLOYMENT_GUIDE.md) - Cloud deployment instructions
- [✅ Final Complete](FINAL_COMPLETE.md) - All features summary

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details

---

## 🙏 Acknowledgments

- **FinBERT-India** - [Vansh180/FinBERT-India-v1](https://huggingface.co/Vansh180/FinBERT-India-v1)
- **Nothing Brand** - Design inspiration
- **yfinance** - Market data API
- **Streamlit** - Dashboard framework

---

## 📞 Contact

**Shamique Khan**
- GitHub: [@shamiquekhan](https://github.com/shamiquekhan)
- Email: shamiquekhan18@gmail.com
- Project: [nifty50-ai](https://github.com/shamiquekhan/nifty50-ai)

---

<div align="center">

**⭐ Star this repo if you find it useful!**

Made with ❤️ for the Indian Stock Market

</div>
```

### Manual Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure settings in `config/config.yaml`

3. Run data collection:
```bash
python src/data_collection/market_data.py
python src/data_collection/news_scraper.py
```

4. Train the model:
```bash
python src/models/train.py
```

5. **Launch Dashboard** (Nothing Design):
```bash
# Quick launch (any OS)
python launch.py

# Or manually
streamlit run dashboard.py
```

Access at: `http://localhost:8501`

### Deploy to Streamlit Cloud (FREE):
See [DEPLOYMENT.md](DEPLOYMENT.md) for complete deployment guide.

### Design Documentation:
See [DESIGN.md](DESIGN.md) for Nothing brand design system details.

## Cost: $0
- Google Colab for GPU training
- Free APIs (yfinance, RSS feeds)
- Open-source models (FinBERT-India)
