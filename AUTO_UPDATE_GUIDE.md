# AUTO-UPDATE SYSTEM GUIDE

## 🚀 Automated Data & Model Updates Every 4 Hours

Your NIFTY50 AI system now has **fully automated updates** that keep everything fresh and accurate.

---

## ✨ What Gets Updated Automatically

Every 4 hours, the system runs:

1. **📊 Market Data Collection** - Latest OHLCV data for all NIFTY50 stocks
2. **📰 News Scraping** - Recent news articles from Moneycontrol
3. **🧠 Sentiment Analysis** - FinBERT-India processes news sentiment
4. **🤖 Model Fine-Tuning** - LSTM model updates with new data (incremental training)
5. **🎯 Fresh Predictions** - AI generates new buy/sell signals
6. **📈 Backtest Validation** - Validates model performance

---

## 🎯 How to Start Auto-Update System

### Method 1: PowerShell (Recommended for Windows)
```powershell
.\start_auto_update.ps1
```

### Method 2: Command Prompt
```cmd
start_auto_update.bat
```

### Method 3: Python Direct
```bash
python src/auto_update.py
```

---

## 📊 Dashboard Auto-Refresh

The dashboard automatically refreshes data every 4 hours with:

- **Auto-Update Status Banner** - Shows if auto-update is running
- **Last Update Timer** - Displays how long ago data was updated
- **Manual Refresh Button** - Force refresh anytime with "🔄 REFRESH NOW"

---

## 🔧 System Features

### Incremental Fine-Tuning
- Uses **lower learning rate** (0.0001) for stable updates
- Only trains on **last 30 days** of new data
- **Preserves model knowledge** from full training
- **Backs up old model** before updating
- Faster than full retraining (5 epochs vs 50+)

### Smart Scheduling
- Runs **every 4 hours** automatically
- **Logs all operations** to `logs/auto_update.log`
- **Error recovery** - continues even if one step fails
- **Success tracking** - reports completion rate

### Data Freshness
- Dashboard cache: **4 hours**
- Market data: **Real-time** (updated every 4h)
- News: **Latest articles** (updated every 4h)
- Model: **Continuously fine-tuned**

---

## 📁 File Structure

```
src/
├── auto_update.py           # Main auto-update orchestrator
├── incremental_training.py  # Incremental model fine-tuning
├── data_collection/
│   ├── market_data.py      # Market data fetcher
│   └── news_scraper.py     # News scraper
└── sentiment/
    └── finbert_engine.py   # Sentiment analyzer

logs/
└── auto_update.log         # Update history and logs

models/
├── lstm_model.keras        # Current model (auto-updated)
└── lstm_model_backup_*.keras  # Backups before each update
```

---

## 🎮 Usage Examples

### Start Auto-Update in Background
```powershell
# Windows PowerShell
Start-Process powershell -ArgumentList "-File start_auto_update.ps1" -WindowStyle Hidden
```

### Check Auto-Update Logs
```bash
# View last 50 lines
Get-Content logs/auto_update.log -Tail 50
```

### Manual Force Update
```bash
# Run all update steps once
python src/auto_update.py
```

---

## 📊 Dashboard Integration

The dashboard shows:

1. **● AUTO-UPDATE: ACTIVE** (green) - System is running
2. **○ AUTO-UPDATE: INACTIVE** (gray) - System needs to be started
3. **LAST DATA UPDATE** - Hours since last refresh
4. **🔄 REFRESH NOW** - Manual cache clear + reload

---

## ⚙️ Configuration

### Change Update Interval

Edit `src/auto_update.py`:

```python
self.update_interval = 4  # Change to desired hours (e.g., 2, 6, 12)
```

### Adjust Fine-Tuning Epochs

Edit `src/incremental_training.py`:

```python
success = fine_tune_model(X_new, y_new, epochs=5)  # Change epochs
```

### Modify Learning Rate

Edit `src/incremental_training.py`:

```python
optimizer=keras.optimizers.Adam(learning_rate=0.0001)  # Adjust LR
```

---

## 🔍 Monitoring

### Check if Auto-Update is Running
```powershell
Get-Process python | Where-Object {$_.CommandLine -like "*auto_update*"}
```

### View Live Logs
```powershell
Get-Content logs/auto_update.log -Wait
```

### Stop Auto-Update
```
Press Ctrl+C in the auto-update terminal window
```

---

## 🎯 Benefits

✅ **Always Fresh Data** - No stale predictions  
✅ **Hands-Free Operation** - Set it and forget it  
✅ **Continuous Learning** - Model improves over time  
✅ **Error Resilience** - System recovers from failures  
✅ **Full Logging** - Track all updates and errors  
✅ **Model Backups** - Never lose previous versions  
✅ **Real-time Accuracy** - Predictions based on latest data  

---

## 🚨 Troubleshooting

### Auto-Update Not Starting
```bash
# Check Python path
python --version

# Verify dependencies
python -m pip install -r requirements.txt
```

### Model Fine-Tuning Fails
- Ensure `models/lstm_model.keras` exists (run full training first)
- Check if sufficient new data is available (needs 30+ days)

### Dashboard Shows "INACTIVE"
- Start auto-update system first
- Check if `auto_update.py` process is running

---

## 📈 Performance

- **Update Cycle Time**: ~5-10 minutes (depending on data volume)
- **CPU Usage**: Low (runs in background)
- **Memory Usage**: ~2-3 GB during fine-tuning
- **Disk Space**: ~100 MB per model backup

---

## 🎉 Quick Start Checklist

- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Run initial full training: `python src/quick_train.py`
- [ ] Start auto-update: `.\start_auto_update.ps1`
- [ ] Launch dashboard: `streamlit run dashboard.py`
- [ ] Verify "AUTO-UPDATE: ACTIVE" shows in dashboard

---

**System Status**: ✅ FULLY AUTOMATED  
**Next Update**: Automatic in 4 hours  
**Manual Control**: Always available via dashboard
