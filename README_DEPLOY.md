# 🚀 Deploy to Streamlit Cloud

## Quick Deploy Steps

### 1. **Push to GitHub** (Already Done ✓)
Your code is already on GitHub at: `https://github.com/shamiquekhan/nifty50-ai`

### 2. **Go to Streamlit Cloud**
Visit: https://share.streamlit.io/

### 3. **Deploy Your App**

1. Click **"New app"**
2. Select your repository: `shamiquekhan/nifty50-ai`
3. Set main file path: `dashboard.py`
4. Click **"Deploy"**

That's it! Your app will be live at: `https://nifty50-ai.streamlit.app`

---

## ⚙️ Configuration (Already Set Up)

### Files Included:
- ✅ `requirements.txt` - All Python dependencies
- ✅ `.streamlit/config.toml` - Streamlit configuration
- ✅ `packages.txt` - System packages
- ✅ `.streamlit/secrets.toml` - Template for secrets

---

## 🔐 Optional: Add Secrets (For Shoonya API)

If you want to use Shoonya API for unlimited free data:

1. In Streamlit Cloud dashboard, go to **Settings → Secrets**
2. Add:
```toml
shoonya_user = "YOUR_USER_ID"
shoonya_password = "YOUR_PASSWORD"
shoonya_totp_key = "YOUR_TOTP_KEY"
```

---

## 📊 What Will Work Immediately:

- ✅ **Live NSE Prices** (via yfinance)
- ✅ **Advanced Quant Analytics**
- ✅ **RSS News Harvesting** (6 unlimited sources)
- ✅ **AI Predictions Dashboard**
- ✅ **Kelly Criterion Risk Management**
- ✅ **Nothing Brand Design**

---

## 🎯 Your Live URL:

After deployment: **https://nifty50-ai.streamlit.app**

Share it with the world! 🌍

---

## 🆘 Troubleshooting

**If deployment fails:**
1. Check logs in Streamlit Cloud dashboard
2. Verify all files are pushed to GitHub
3. Ensure `dashboard.py` is in root directory
4. Check requirements.txt for version conflicts

**For large models (TensorFlow/Torch):**
- Streamlit Cloud has memory limits (1GB free tier)
- Consider removing heavy ML dependencies if not actively training
- Use pre-trained model files uploaded separately

---

## 📝 Notes

- **Auto-refresh**: Dashboard caches live prices for 60 seconds
- **Data**: Will use demo data initially (run training pipeline to generate real predictions)
- **Updates**: Push to GitHub → Auto-deploys to Streamlit Cloud
- **Free Tier**: Unlimited public apps, no credit card needed

**Enjoy your live trading dashboard! 🚀📈**
