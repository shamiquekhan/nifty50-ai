# 🚀 QUICK REFERENCE CARD

## Launch Commands

```powershell
# Windows Quick Launch
.\launch.ps1

# Python Launcher (any OS)
python launch.py

# Manual Launch
streamlit run dashboard.py
```

**URL:** http://localhost:8501

---

## Design System

### Colors
- `#000000` Black (Background)
- `#FFFFFF` White (Text)
- `#FF0000` Red (Accent)

### Fonts
- **Orbitron** (Headers, Metrics)
- **Share Tech Mono** (Body, Labels)

### Key Files
- `dashboard.py` - Main app
- `.streamlit/config.toml` - Theme
- `DESIGN.md` - Full guide

---

## Project Structure

```
Nifty50Qualtml/
├── dashboard.py          ← Main dashboard
├── launch.py/ps1         ← Launchers
├── requirements.txt      ← Dependencies
├── config/
│   └── config.yaml       ← Settings
├── src/
│   ├── data_collection/  ← Market & news data
│   ├── sentiment/        ← FinBERT engine
│   ├── models/           ← LSTM model
│   ├── agents/           ← Kelly agent
│   └── utils/            ← Preprocessing
├── data/
│   ├── raw/              ← Downloaded data
│   ├── processed/        ← Clean datasets
│   └── models/           ← Saved weights
└── .streamlit/
    └── config.toml       ← Theme config
```

---

## Data Pipeline

```
1. Market Data
   python src/data_collection/market_data.py

2. News Scraping
   python src/data_collection/news_scraper.py

3. Sentiment Analysis
   python src/sentiment/finbert_engine.py

4. Train Model
   python src/models/train.py

5. Generate Predictions
   python src/predict.py

6. Launch Dashboard
   streamlit run dashboard.py
```

---

## Nothing Design Elements

### ✓ Implemented
- [x] Pure black background
- [x] No sidebar layout
- [x] Dot matrix fonts (Orbitron + Share Tech Mono)
- [x] Red glow effects
- [x] Uppercase metrics
- [x] Clean ticker badge
- [x] Dark chart theme
- [x] Matrix grid background

### Typography Hierarchy
```
TITLE:    4rem Orbitron 900 (Glowing)
SECTION:  1.8rem Orbitron 700 (Red underline)
METRIC:   2.5rem Orbitron 700 (White)
LABEL:    0.9rem Share Tech Mono (Gray)
BODY:     1rem Share Tech Mono (White)
```

---

## Deployment

### Streamlit Cloud (Free)
1. Push to GitHub
2. Go to streamlit.io/cloud
3. New app → Select repo
4. Main file: `dashboard.py`
5. Deploy!

**Result:** `https://your-app.streamlit.app`

---

## Customization

### Change Accent Color
`.streamlit/config.toml`:
```toml
primaryColor = "#00FF00"
```

### Modify CSS
`dashboard.py` → `DOT_MATRIX_CSS` variable

### Add Metrics
```python
st.metric("LABEL", value, delta)
```

---

## Documentation Map

| Read This For... | File |
|-----------------|------|
| Getting started | `README.md` |
| Design details | `DESIGN.md` |
| Feature list | `FEATURES.md` |
| Deployment | `DEPLOYMENT.md` |
| Nothing design | `NOTHING_DESIGN_COMPLETE.md` |

---

## Troubleshooting

**Dashboard won't load?**
```bash
pip install --upgrade streamlit
```

**Missing data?**
```bash
python src/data_collection/market_data.py
```

**Fonts not showing?**
- Check internet (loads from Google Fonts)
- Clear browser cache
- Try different browser

---

## Key Metrics Displayed

- **Price** + Change %
- **Volume**
- **RSI** (Momentum)
- **Volatility** (ATR)
- **Sentiment Score**
- **Model Probability**
- **Kelly Fraction**
- **Position Size**
- **Confidence Level**

---

## AI Components

1. **Bi-Directional LSTM** - Price pattern recognition
2. **FinBERT-India** - News sentiment analysis
3. **Kelly Criterion** - Position sizing
4. **Risk Agent** - Alignment checking

---

## Cost Breakdown

| Component | Cost |
|-----------|------|
| Market Data (yfinance) | ₹0 |
| News Data (RSS) | ₹0 |
| FinBERT Model (HF) | ₹0 |
| Streamlit Hosting | ₹0 |
| Google Colab (Training) | ₹0 |
| **TOTAL** | **₹0** |

---

## Performance

- Load Time: ~2s
- Chart Render: ~0.5s
- Memory: ~150MB
- Model Accuracy: ~68-72%

---

**Need help?** Read the full docs in respective .md files!
