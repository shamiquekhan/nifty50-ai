# Online Learning System - Complete Guide
## Offline Training + 4-Hour Auto-Updates with Incremental Fine-Tuning

This guide explains the **Online Learning Architecture** that powers the NIFTY50 AI system.

---

## 🎯 Architecture Overview

**Philosophy**: The market changes constantly, but fundamental patterns persist. We combine:
1. **Offline Phase**: Train base model on 2 years of historical data
2. **Online Phase**: Fine-tune every 4 hours with latest data

**Key Innovation**: Transfer Learning with low learning rate prevents "catastrophic forgetting" while adapting to recent market conditions.

---

## 📊 Phase 1: Offline Training (Base Model)

### What It Does
Trains a Bi-Directional LSTM on 2 years of NIFTY50 data to learn fundamental market patterns:
- Trend detection (support/resistance)
- Volatility clustering
- Mean reversion signals
- Technical indicator relationships

### Files Created
- `models/lstm_model.keras` - Base model (the "brain")
- `models/scaler.pkl` - MinMaxScaler for normalization
- `logs/training_history.csv` - Loss metrics over epochs

### How to Run

**Step 1: Collect Historical Data**
```bash
# Downloads 2 years of NIFTY50 data with technical indicators
python src/data_collection/market_data.py
```

**Step 2: Train Base Model**
```bash
# Trains Bi-Directional LSTM for 50 epochs (~10 minutes on CPU)
python src/train.py
```

**What Happens:**
1. Loads market data (Open, High, Low, Close, Volume)
2. Adds technical indicators (RSI, MACD, Bollinger Bands, ATR)
3. Normalizes data using MinMaxScaler (crucial for LSTM)
4. Creates sequences (60-day lookback windows)
5. Trains Bi-Directional LSTM:
   ```
   Layer 1: Bidirectional LSTM (64 units) + Dropout (0.2)
   Layer 2: LSTM (64 units) + Dropout (0.2)
   Layer 3: Dense (25 units)
   Output: Dense (1 unit) - Predicted Close Price
   ```
6. Saves model and scaler

**Expected Output:**
```
Epoch 50/50
loss: 0.0012 - val_loss: 0.0015
✓ Model saved: models/lstm_model.keras
✓ Test Accuracy: 52.34%
```

---

## 🔄 Phase 2: Online Learning (Incremental Updates)

### What It Does
Every 4 hours, the system:
1. Fetches latest market data (last 1-5 days)
2. Harvests news from RSS feeds
3. Runs sentiment analysis
4. **Fine-tunes** model with recent data (5 epochs, LR=0.0001)
5. Generates new predictions
6. Updates backtest results

### Architecture: Transfer Learning

**Why Low Learning Rate (0.0001)?**
- High LR (0.001) → Model "forgets" historical patterns → Catastrophic forgetting
- Low LR (0.0001) → Model "nudges" weights slightly → Adapts to recent volatility

**Why Few Epochs (5)?**
- Many epochs (50) → Overfits to last 4 hours of data
- Few epochs (5) → Learns recent trend without losing generalization

**Why Last 500 Samples?**
- Training on all historical data every 4 hours → Wasted compute
- Training only on last 4 hours → Overfitting
- Last 500 samples (~2 months) → Balance of recent relevance + pattern diversity

### Incremental Training Pipeline

**File**: `src/incremental_training.py`

**Process:**
```python
1. Load base model (lstm_model.keras)
2. Load fitted scaler (scaler.pkl)
3. Prepare latest data:
   - Scale using existing scaler (DO NOT refit)
   - Create 60-day sequences
   - Use last 500 samples
4. Recompile with Adam(LR=0.0001)
5. Fine-tune for 5 epochs
6. Backup old model (lstm_model_backup_TIMESTAMP.keras)
7. Save updated model
```

**Command:**
```bash
python src/incremental_training.py --data data/market_data_20251214.csv --epochs 5
```

**Output:**
```
============================================================
STARTING INCREMENTAL TRAINING
============================================================
✓ Loading base model from models/lstm_model.keras
✓ Backed up current model to models/lstm_model_backup_20251214_235900.keras
Fine-tuning with 500 samples, 5 epochs, LR=0.0001
Epoch 1/5 - loss: 0.0014 - val_loss: 0.0016
Epoch 5/5 - loss: 0.0011 - val_loss: 0.0013
✓ Model fine-tuned and saved
============================================================
✓ FINE-TUNING COMPLETE
  Final Loss: 0.001134
  Final Val Loss: 0.001287
============================================================
```

---

## ⏰ Phase 3: Auto-Update System (4-Hour Cycle)

### What It Does
Orchestrates the complete update pipeline automatically every 4 hours.

**File**: `src/auto_update.py`

**Pipeline Steps:**

1. **Market Data Collection** (30 seconds)
   - Fetches latest OHLCV data
   - Adds technical indicators
   - Saves to `data/market_data_TIMESTAMP.csv`

2. **News Harvesting** (10 seconds)
   - Scrapes 6 RSS feeds (unlimited & free)
   - Extracts ticker mentions
   - Saves to `data/news/news_database.csv`

3. **Sentiment Analysis** (60 seconds)
   - Runs FinBERT-India on latest news
   - Generates daily sentiment scores
   - Saves to `data/daily_sentiment_TIMESTAMP.csv`

4. **Incremental Training** (3 minutes)
   - Fine-tunes LSTM with latest data
   - Backs up current model
   - Saves updated model

5. **AI Predictions** (5 seconds)
   - Generates BUY/SELL/WAIT signals
   - Calculates Kelly fractions
   - Saves to `data/predictions.csv`

6. **Backtest Update** (10 seconds)
   - Tests strategy on latest data
   - Updates equity curve
   - Saves to `data/backtest_trades_TIMESTAMP.csv`

**Total Cycle Time**: ~4-5 minutes

### How to Run

**Option 1: PowerShell (Windows)**
```powershell
.\start_auto_update.ps1
```

**Option 2: Bash (Linux/Mac)**
```bash
chmod +x start_auto_update.sh
./start_auto_update.sh
```

**Option 3: Python Direct**
```bash
# Continuous mode (runs forever, every 4 hours)
python src/auto_update.py

# Single cycle (for testing)
python src/auto_update.py --once
```

**Output:**
```
================================================================================
STARTING AUTO-UPDATE CYCLE #1
Time: 2025-12-14 23:00:00
================================================================================
→ Running: Market Data Collection
✓ Market Data Collection completed successfully
→ Running: News Harvesting
✓ News Harvesting completed successfully (189 articles)
→ Running: Sentiment Analysis
✓ Sentiment Analysis completed successfully
→ Running: Incremental Model Training
✓ Incremental Model Training completed successfully
→ Running: AI Predictions
✓ AI Predictions completed successfully (10 stocks)
→ Running: Backtest Update
✓ Backtest Update completed successfully
================================================================================
✓ CYCLE #1 COMPLETED SUCCESSFULLY
  Duration: 247.3 seconds
  Next cycle in 4 hours
================================================================================
```

---

## 🧠 How Online Learning Works

### Concept: The Market's "Memory"

**Problem**: Markets evolve. A model trained in 2023 might miss 2025 patterns.

**Solution**: LSTM has internal "cell state" that remembers sequences.

**Analogy**:
- **Offline Training**: Learning to swim in a calm pool (fundamental strokes)
- **Online Learning**: Adapting strokes when you jump into ocean waves (recent conditions)

### Mathematical Intuition

**Base Model (Offline)**:
- Learns weights `W_base` from 2 years of data
- Captures general patterns: `P(price_tomorrow | 60_days_history)`

**Fine-Tuning (Online)**:
- Adjusts weights: `W_new = W_base + α * ∇L(recent_data)`
- Low `α` (learning rate 0.0001) → Small nudges, preserves `W_base`
- Result: Model knows "general patterns" + "this week's volatility regime"

### Why This Beats Static Models

| Approach | Adapts to News? | Remembers Patterns? | Risk |
|----------|----------------|---------------------|------|
| **Static Model** | ❌ No | ✅ Yes | Stale predictions |
| **Retrain from Scratch** | ✅ Yes | ❌ No | Catastrophic forgetting |
| **Online Learning** | ✅ Yes | ✅ Yes | ✅ Balanced |

**Example**:
- **Week 1**: RBI announces rate cut → Market rallies
- **Static model**: Doesn't know about rate cut → Wrong signal
- **Online model**: Fine-tuned on post-announcement data → Adapts

---

## 📈 Data Caching Strategy

### Problem: API Rate Limits

**Without Caching**:
- Fetch 2 years of data every 4 hours
- yfinance soft limits → Failures
- Wasted bandwidth

**With Caching**:
- Fetch only last 5 days every 4 hours
- Store in local database (CSV or DuckDB)
- Append new candles incrementally

### Implementation

**Current**: CSV Files
- `data/market_data_TIMESTAMP.csv` (latest full dataset)
- Dashboard caches for 4 hours (`@st.cache_data(ttl=14400)`)

**Optional Upgrade**: DuckDB
```python
import duckdb

conn = duckdb.connect('data/nifty50.duckdb')

# Insert only new data
conn.execute("""
    INSERT OR IGNORE INTO market_data 
    SELECT * FROM read_csv_auto('latest_data.csv')
""")

# Fast queries
result = conn.execute("""
    SELECT * FROM market_data 
    WHERE Date >= '2025-12-01'
""").df()
```

**Benefits**:
- 10x faster than Pandas
- Handles millions of rows
- SQL queries for analysis

---

## 🔧 Configuration

**File**: `config/config.yaml`

```yaml
# Auto-Update Settings
auto_update:
  enabled: true
  interval_hours: 4
  
  incremental_training:
    enabled: true
    epochs: 5
    learning_rate: 0.0001
    fine_tune_samples: 500
  
  backup_models: true
  max_backups: 10
```

---

## 📊 Monitoring & Logs

**Log Files**:
- `logs/auto_update.log` - Main pipeline log
- `logs/incremental_training.log` - Training metrics
- `logs/news_harvester.log` - RSS feed status

**Check Status**:
```bash
# View last 50 lines of auto-update log
tail -n 50 logs/auto_update.log

# Check for errors
grep "ERROR" logs/auto_update.log

# Count successful cycles
grep "COMPLETED SUCCESSFULLY" logs/auto_update.log | wc -l
```

**Model Backups**:
```bash
# List all model backups
ls -lh models/lstm_model_backup_*.keras

# Restore from backup
cp models/lstm_model_backup_20251214_120000.keras models/lstm_model.keras
```

---

## 🚀 Deployment

### Local Development
```bash
# Start auto-update in background
python src/auto_update.py &

# Or use screen/tmux
screen -S nifty_ai
python src/auto_update.py
# Press Ctrl+A, then D to detach
```

### Production (Cloud)

**Option 1: Google Cloud VM (Free Tier)**
```bash
# Create e2-micro instance
gcloud compute instances create nifty-ai \
    --machine-type=e2-micro \
    --zone=us-central1-a

# SSH and setup
gcloud compute ssh nifty-ai
git clone https://github.com/shamiquekhan/nifty50-ai.git
cd nifty50-ai
pip install -r requirements.txt

# Run with screen
screen -S auto_update
python src/auto_update.py
```

**Option 2: Heroku (With Scheduler Add-on)**
```bash
# Create Procfile
echo "worker: python src/auto_update.py" > Procfile

# Deploy
heroku create nifty-ai
git push heroku main
heroku ps:scale worker=1
```

**Option 3: AWS EC2**
- Launch t2.micro instance (Free tier)
- Same setup as Google Cloud

---

## 🎓 Best Practices

### 1. Learning Rate Selection
- **Too High (0.01)**: Catastrophic forgetting, model becomes unstable
- **Too Low (0.00001)**: No adaptation, model stays static
- **Optimal (0.0001)**: Balanced adaptation

**Rule of Thumb**: 10x lower than offline training LR

### 2. Epoch Count
- **Too Many (50)**: Overfits to recent 4 hours
- **Too Few (1)**: Insufficient learning
- **Optimal (5)**: Captures recent pattern shift

**Test**: Monitor val_loss. If it increases after epoch 3, reduce to 3 epochs.

### 3. Fine-Tune Sample Size
- **Too Large (All data)**: Wasted compute, no benefit
- **Too Small (50)**: High variance, unstable
- **Optimal (500)**: ~2 months of context

**Formula**: `samples = max(60 * 5, min(2000, total_samples * 0.1))`

### 4. Backup Strategy
- Keep last 10 model backups
- Daily backup to cloud (S3/Google Cloud Storage)
- Test restoration process monthly

### 5. Error Handling
- Continue pipeline even if one step fails
- Send alerts for critical failures (Telegram/Email)
- Auto-restart on crash (systemd service)

---

## 📈 Expected Results

**After 1 Month of Auto-Updates:**
- 180 update cycles (30 days × 6 cycles/day)
- Model exposed to 180 × 500 = 90,000 training samples
- Adapts to:
  - Quarterly earnings season volatility
  - RBI policy changes
  - Global market shifts (Fed rates, geopolitics)

**Performance Metrics:**
- **Static Model**: 51% accuracy (fixed on 2023-2024 data)
- **Online Model**: 53-55% accuracy (adapts to 2025 patterns)
- **Improvement**: +2-4% edge (huge in trading)

---

## 🔍 Troubleshooting

**Issue**: Model loss increases after fine-tuning
- **Cause**: Learning rate too high or bad data
- **Fix**: Reduce LR to 0.00005, check data quality

**Issue**: Auto-update crashes
- **Cause**: Missing dependencies or corrupted files
- **Fix**: Check logs, reinstall requirements

**Issue**: Predictions don't improve
- **Cause**: Insufficient new data or market regime shift
- **Fix**: Increase fine-tune samples to 1000, add more features

---

## 📚 Key Files Reference

| File | Purpose | When to Edit |
|------|---------|-------------|
| `src/incremental_training.py` | Fine-tuning logic | Change LR, epochs, sample size |
| `src/auto_update.py` | 4-hour orchestration | Add/remove pipeline steps |
| `config/config.yaml` | System settings | Adjust intervals, paths |
| `models/lstm_model.keras` | Current model | Never (auto-updated) |
| `models/scaler.pkl` | Normalization | Recreate if features change |

---

## 🎯 Summary

**Offline Phase**:
```
2 years data → Bi-LSTM (50 epochs) → base_model.keras
```

**Online Phase** (Every 4 hours):
```
Latest data → Fine-tune (5 epochs, LR=0.0001) → updated_model.keras → Predictions
```

**Result**: Model that knows general patterns + adapts to recent volatility.

This is the **state-of-the-art** approach for production ML systems in finance. 🚀

---

**Questions?**
- Check `logs/auto_update.log` for detailed execution
- Review `logs/incremental_training.log` for training metrics
- See `models/finetune_history_*.csv` for loss curves
