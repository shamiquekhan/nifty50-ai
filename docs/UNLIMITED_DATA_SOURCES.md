# Unlimited Free Data Sources Guide
## Industry Secrets for ₹0 Institutional-Grade Data

This guide explains how to access **unlimited, free, production-quality** data sources for the NIFTY50 AI Trading System.

---

## 🎯 Why This Matters

**Standard Approach (❌ Limited):**
- NewsAPI.org: 100 requests/day, $449/month for historical data
- Alpha Vantage: 5 API calls/minute
- Yahoo Finance: Soft limits, unreliable for production

**Professional Approach (✅ Unlimited):**
- Shoonya API: **Unlimited** real-time + historical data
- RSS Feeds: **Unlimited** news articles, updated every 15 minutes
- Zero brokerage brokers: **Free accounts** with institutional APIs

---

## 📊 1. Market Data - Unlimited OHLCV

### Option A: Shoonya (Finvasia) API - **RECOMMENDED**

**Why:** Zero brokerage broker with unlimited free API access. This is what professional quants use.

**Setup Steps:**

1. **Open Free Account:**
   - Visit: https://shoonya.com
   - Sign up (₹0 brokerage, ₹0 AMC)
   - Complete KYC (Aadhaar + PAN)

2. **Get API Credentials:**
   - Login to Shoonya web platform
   - Go to Settings → API
   - Generate: User ID, Password, Vendor Code, API Key

3. **Configure Environment:**
   ```bash
   # Create .env file
   echo "SHOONYA_USER_ID=your_user_id" >> .env
   echo "SHOONYA_PASSWORD=your_password" >> .env
   echo "SHOONYA_VENDOR_CODE=your_vendor_code" >> .env
   echo "SHOONYA_API_KEY=your_api_key" >> .env
   ```

4. **Install Library:**
   ```bash
   pip install shoonya
   # OR
   pip install NorenRestApiPy
   ```

5. **Test Connection:**
   ```python
   from src.data_collection.shoonya_api import get_nifty50_data
   
   # Get 1 year of RELIANCE data
   df = get_nifty50_data('RELIANCE', days=365)
   print(f"Downloaded {len(df)} days of data")
   ```

**Benefits:**
- ✅ **Unlimited** API calls (no rate limits)
- ✅ **Real-time** tick-by-tick data
- ✅ **Historical** data going back years
- ✅ **WebSocket** support for live streaming
- ✅ **Option chain** data included
- ✅ **₹0 cost** - completely free

### Option B: yfinance - **FALLBACK**

**Why:** Reliable for backtesting, but has soft rate limits for production.

**Usage:**
```python
import yfinance as yf

# Automatically used if Shoonya not configured
df = yf.download('RELIANCE.NS', period='1y', interval='1d')
```

**Limitations:**
- ⚠️ Soft rate limits (delays between requests)
- ⚠️ No official API (scraping Yahoo Finance)
- ⚠️ Can break if Yahoo changes website

---

## 📰 2. News Data - Unlimited Articles

### RSS Feeds - **COMPLETELY FREE & UNLIMITED**

**Why:** Official RSS feeds have no rate limits. This is how Bloomberg/Reuters get news.

**Active Sources:**

| Source | URL | Update Frequency | Content Type |
|--------|-----|------------------|--------------|
| MoneyControl Market | `https://www.moneycontrol.com/rss/marketreports.xml` | Real-time | Market commentary |
| MoneyControl Business | `https://www.moneycontrol.com/rss/business.xml` | Real-time | Corporate news |
| Economic Times Stocks | `https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms` | Real-time | Stock movements |
| Economic Times Economy | `https://economictimes.indiatimes.com/news/economy/rssfeeds/1373380680.cms` | Real-time | Macro news |
| LiveMint Markets | `https://www.livemint.com/rss/markets` | Real-time | Market analysis |
| LiveMint Companies | `https://www.livemint.com/rss/companies` | Real-time | Company updates |

**Setup:**

1. **Run News Harvester:**
   ```bash
   # Single harvest
   python src/data_collection/rss_news_harvester.py
   
   # Continuous mode (every 15 minutes)
   python src/data_collection/rss_news_harvester.py --continuous --interval 15
   ```

2. **Check Results:**
   ```bash
   # View harvested news
   ls -lh data/news/news_database.csv
   
   # Get news for specific ticker
   python src/data_collection/rss_news_harvester.py --ticker RELIANCE
   ```

3. **Automate (Optional):**
   ```bash
   # Add to crontab (Linux/Mac)
   */15 * * * * cd /path/to/project && python src/data_collection/rss_news_harvester.py
   
   # Or use Windows Task Scheduler
   # Run every 15 minutes: python src/data_collection/rss_news_harvester.py
   ```

**Benefits:**
- ✅ **Unlimited** articles (no API keys needed)
- ✅ **Real-time** updates (15-minute lag)
- ✅ **Structured** data (title, summary, link, published date)
- ✅ **Automatic** ticker extraction
- ✅ **Deduplication** built-in
- ✅ **₹0 cost** - RSS is free forever

**Example Output:**
```
Harvested 47 new articles:
- moneycontrol_market: 12 articles (RELIANCE, TCS, INFY mentioned)
- economic_times_stocks: 18 articles (HDFCBANK, ICICIBANK mentioned)
- livemint_markets: 17 articles (General market news)

Total database: 3,452 unique articles
```

---

## 📈 3. Fundamental Data - Company Metrics

### Screener.in - **FREE EXPORT**

**Why:** Best free source for P/E, ROE, Debt/Equity ratios.

**Method:**
1. Visit: https://www.screener.in/company/RELIANCE/
2. Click "Export to Excel" at bottom
3. Parse Excel file for ratios

**Programmatic Access:**
```python
import requests
import pandas as pd

def get_fundamental_data(ticker: str) -> dict:
    url = f"https://www.screener.in/company/{ticker}/"
    # Add headers to avoid blocking
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    
    # Parse HTML for key ratios
    # (Implementation in src/data_collection/fundamentals.py)
    
    return {
        'PE_ratio': 24.5,
        'ROE': 15.3,
        'Debt_to_Equity': 0.42
    }
```

### NSE Corporate Announcements - **OFFICIAL DATA**

**URL:** https://www.nseindia.com/companies-listing/corporate-filings-announcements

**Content:**
- Board meetings
- Dividend declarations
- Bonus announcements
- Results

---

## 💰 4. Macro Economic Data

### RBI Database - **OFFICIAL GOVERNMENT DATA**

**URL:** https://dbie.rbi.org.in/

**Available Data:**
- Repo Rate
- Inflation (CPI, WPI)
- INR/USD exchange rate
- GDP growth

### NIFTY Indices - **INDEX METRICS**

**URL:** https://www.niftyindices.com/reports/historical-data

**Download:**
- NIFTY50 P/E ratio
- NIFTY50 P/B ratio
- Monthly historical data

---

## 🚀 Quick Start - Get All Data Now

**Step 1: Install Dependencies**
```bash
pip install -r requirements.txt
```

**Step 2: Run Data Collection Pipeline**
```bash
# Market data (yfinance fallback)
python src/data_collection/market_data.py

# News harvester (unlimited RSS)
python src/data_collection/rss_news_harvester.py

# Sentiment analysis
python src/sentiment/finbert_engine.py
```

**Step 3: Verify Data**
```bash
# Check market data
ls -lh data/market_data_*.csv

# Check news data
ls -lh data/news/news_database.csv

# Check sentiment
ls -lh data/daily_sentiment_*.csv
```

---

## 📊 Data Storage Strategy

**Local Database (Recommended: DuckDB)**

```python
import duckdb

# Create database
conn = duckdb.connect('data/nifty50.duckdb')

# Import CSV data
conn.execute("""
    CREATE TABLE market_data AS 
    SELECT * FROM read_csv_auto('data/market_data_*.csv')
""")

conn.execute("""
    CREATE TABLE news AS 
    SELECT * FROM read_csv_auto('data/news/news_database.csv')
""")

# Fast queries
result = conn.execute("""
    SELECT * FROM market_data 
    WHERE Ticker = 'RELIANCE' 
    AND Date >= '2024-01-01'
""").fetchdf()
```

**Benefits:**
- ⚡ **Fast** - 10x faster than Pandas
- 💾 **Efficient** - Handles millions of rows
- 🔍 **SQL** - Familiar query language
- 💰 **Free** - Open source

---

## 🎯 Pro Tips

### 1. Build Proprietary Dataset
Run RSS harvester continuously for 30 days → You'll have a dataset better than paid services.

### 2. Shoonya for Live Trading
Use Shoonya API for:
- Live price updates (WebSocket)
- Order execution (if deploying live trading)
- Option chain data (for Put-Call Ratio sentiment)

### 3. Backup Everything
```bash
# Automated daily backup
python src/backup_data.py --destination s3://your-bucket
```

### 4. Monitor Data Quality
```python
# Check for missing dates
from src.data_quality import DataQualityChecker

checker = DataQualityChecker()
checker.check_missing_dates('data/market_data_*.csv')
checker.check_outliers('Volume')
```

---

## 🔒 Security Best Practices

**Never commit API keys:**
```bash
# Add to .gitignore
echo ".env" >> .gitignore
echo "config/shoonya_config.yaml" >> .gitignore
```

**Use environment variables:**
```python
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('SHOONYA_API_KEY')
```

---

## 📈 Expected Data Volume

**After 1 Month:**
- Market Data: ~1,500 rows × 50 stocks = 75,000 records
- News Articles: ~3,000 unique articles
- Sentiment Scores: ~30 daily scores
- Storage: ~50 MB

**After 1 Year:**
- Market Data: ~250,000 records
- News Articles: ~36,000 articles
- Storage: ~500 MB

All this data is **100% free** using the sources above.

---

## 🎓 Summary

| Data Type | Source | Cost | Rate Limit | Quality |
|-----------|--------|------|------------|---------|
| **Market OHLCV** | Shoonya API | ₹0 | **Unlimited** | ⭐⭐⭐⭐⭐ |
| **Market OHLCV** | yfinance | ₹0 | Soft limits | ⭐⭐⭐⭐ |
| **News** | RSS Feeds | ₹0 | **Unlimited** | ⭐⭐⭐⭐⭐ |
| **Fundamentals** | Screener.in | ₹0 | Manual export | ⭐⭐⭐⭐ |
| **Macro Data** | RBI, NSE | ₹0 | None | ⭐⭐⭐⭐⭐ |

**Total Monthly Cost: ₹0**

**vs. Paid Services:**
- Bloomberg Terminal: $24,000/year
- Refinitiv Eikon: $22,000/year
- NewsAPI Pro: $5,388/year

You're getting institutional-grade data for **FREE** using industry secrets. 🚀

---

**Questions?**
- Check `logs/data_collection.log` for detailed debug info
- Review `config/data_sources.yaml` for all settings
- See `src/data_collection/` for implementation code
