"""
Quick Start Guide: Step-by-step execution
Run these commands to build the complete pipeline.
"""

# ===== STEP 1: Install Dependencies =====
print("""
STEP 1: Install Dependencies
-----------------------------
pip install -r requirements.txt

This installs:
- yfinance, pandas-ta (market data)
- transformers, torch (FinBERT-India)
- tensorflow (LSTM model)
- streamlit (dashboard)
""")

# ===== STEP 2: Collect Market Data =====
print("""
STEP 2: Collect Market Data
----------------------------
python src/data_collection/market_data.py

This will:
- Download 2 years of NIFTY50 stock data
- Calculate technical indicators (RSI, MACD, Bollinger Bands)
- Save to: data/raw/market_data_YYYYMMDD_HHMMSS.csv
""")

# ===== STEP 3: Collect News Data =====
print("""
STEP 3: Collect News Data
--------------------------
python src/data_collection/news_scraper.py

This will:
- Scrape RSS feeds from MoneyControl & Economic Times
- Save to: data/raw/news_raw_YYYYMMDD_HHMMSS.csv
""")

# ===== STEP 4: Analyze Sentiment =====
print("""
STEP 4: Analyze Sentiment with FinBERT-India
----------------------------------------------
python src/sentiment/finbert_engine.py

This will:
- Load FinBERT-India model from Hugging Face
- Analyze sentiment of all news articles
- Aggregate daily sentiment scores
- Save to: data/processed/daily_sentiment_YYYYMMDD_HHMMSS.csv
""")

# ===== STEP 5: Train Model =====
print("""
STEP 5: Train the Dual-LSTM Model
----------------------------------
python src/train.py

This will:
- Merge market data with sentiment
- Preprocess and create sequences
- Train the dual-input LSTM model
- Save best model to: data/models/best_model.keras
- Generate training plots
""")

# ===== STEP 6: Make Predictions =====
print("""
STEP 6: Make Predictions
-------------------------
python src/predict.py

This will:
- Load trained model
- Make predictions on latest data
- Apply Kelly Criterion agent
- Save recommendations to: data/processed/predictions.csv
""")

# ===== STEP 7: Test Kelly Agent (Optional) =====
print("""
STEP 7 (Optional): Test Kelly Agent
------------------------------------
python src/agents/kelly_agent.py

This runs test cases to demonstrate the risk management agent.
""")

print("""
=" * 60)
✅ QUICK START GUIDE COMPLETE
=" * 60)

💡 Pro Tips:
1. Run steps 2-4 weekly to get fresh data
2. Retrain model (step 5) monthly
3. Use Google Colab for free GPU training
4. Adjust config.yaml for custom settings

📊 Next Steps:
- Build Streamlit dashboard
- Backtest strategy
- Add more technical indicators
- Expand to full NIFTY50 (currently top 10)
""")
