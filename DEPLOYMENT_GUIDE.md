# NIFTY50 AI Trading System - Streamlit Cloud Deployment

## 🚀 Deployment Guide

### Prerequisites
1. GitHub account
2. Streamlit Cloud account (sign up at https://streamlit.io/cloud)
3. Your repository pushed to GitHub

### Step 1: Prepare Repository

Ensure these files are in your repository:
- ✅ `dashboard.py`
- ✅ `requirements.txt`
- ✅ `.streamlit/config.toml`
- ✅ `config/config.yaml`
- ✅ All source files in `src/`

### Step 2: Create .gitignore

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
.venv

# Jupyter
.ipynb_checkpoints

# Data files (optional - remove if you want to include sample data)
data/raw/*.csv
data/processed/*.csv
data/results/*.png
data/results/*.csv

# Models
models/*.keras
models/*.h5

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Secrets
.env
secrets.toml
```

### Step 3: Update requirements.txt

Ensure all dependencies are listed:
```
yfinance>=0.2.33
pandas>=2.0.0
numpy>=1.24.0
pandas-ta>=0.3.14b
transformers>=4.35.0
torch>=2.0.0
feedparser>=6.0.10
tensorflow>=2.14.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
plotly>=5.17.0
streamlit>=1.28.0
pyyaml>=6.0
requests>=2.31.0
beautifulsoup4>=4.12.0
```

### Step 4: Create Streamlit Cloud App

1. Go to https://streamlit.io/cloud
2. Click "New app"
3. Connect your GitHub repository
4. Select your repository and branch
5. Set main file path: `dashboard.py`
6. Click "Deploy"

### Step 5: Configure Secrets (if needed)

If you need API keys or secrets:

1. Go to app settings → Secrets
2. Add in TOML format:
```toml
# API Keys (if using paid data sources)
[api]
alpha_vantage_key = "your_key_here"
news_api_key = "your_key_here"

# Database (if using cloud storage)
[database]
connection_string = "your_connection_here"
```

### Step 6: Set Environment Variables

In `.streamlit/config.toml`:
```toml
[server]
headless = true
port = 8501
enableCORS = false
enableXsrfProtection = false

[theme]
primaryColor = "#D71921"
backgroundColor = "#000000"
secondaryBackgroundColor = "#1A1A1A"
textColor = "#FFFFFF"
font = "monospace"

[browser]
gatherUsageStats = false
```

### Automated Data Updates

Create `.streamlit/tasks.yaml` for scheduled data collection:
```yaml
tasks:
  - name: update_market_data
    schedule: "0 18 * * 1-5"  # 6 PM on weekdays
    command: "python src/data_collection/market_data.py"
  
  - name: update_news
    schedule: "0 */6 * * *"  # Every 6 hours
    command: "python src/data_collection/news_scraper.py"
  
  - name: generate_predictions
    schedule: "30 18 * * 1-5"  # 6:30 PM on weekdays
    command: "python src/predict.py"
```

### Free Tier Limitations

Streamlit Cloud free tier includes:
- ✅ 1 GB RAM
- ✅ 1 CPU core
- ✅ Unlimited public apps
- ⚠️ Apps sleep after 7 days of inactivity
- ⚠️ Limited to 1 concurrent viewer on free plan

### Optimization for Free Tier

1. **Reduce Memory Usage:**
   - Cache data aggressively
   - Limit historical data range
   - Use lightweight models

2. **Faster Loading:**
   - Precompute indicators
   - Store processed data
   - Lazy load heavy libraries

3. **Keep App Awake:**
   - Use UptimeRobot (free service)
   - Ping your app every 5 minutes

### Alternative Deployment Options

#### 1. Railway.app
```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and deploy
railway login
railway init
railway up
```

#### 2. Render.com
- Create `render.yaml`:
```yaml
services:
  - type: web
    name: nifty50-ai
    env: python
    buildCommand: "pip install -r requirements.txt"
    startCommand: "streamlit run dashboard.py --server.port $PORT"
```

#### 3. Heroku
```bash
# Create Procfile
echo "web: streamlit run dashboard.py --server.port=$PORT --server.address=0.0.0.0" > Procfile

# Deploy
heroku create nifty50-ai
git push heroku main
```

### Monitoring & Analytics

Add Google Analytics to `dashboard.py`:
```python
# In the <head> section
st.markdown("""
<script async src="https://www.googletagmanager.com/gtag/js?id=YOUR_GA_ID"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'YOUR_GA_ID');
</script>
""", unsafe_allow_html=True)
```

### Custom Domain

1. Go to Streamlit Cloud settings
2. Add custom domain
3. Update DNS records:
   - Type: CNAME
   - Name: www
   - Value: [your-app].streamlit.app

### Troubleshooting

**App won't start:**
- Check logs in Streamlit Cloud
- Verify all dependencies in requirements.txt
- Ensure data files are available

**Out of memory:**
- Reduce cache TTL
- Limit data size
- Use sampling for large datasets

**Slow performance:**
- Optimize data loading
- Use st.cache_data extensively
- Reduce chart complexity

### Production Checklist

- [ ] All dependencies in requirements.txt
- [ ] .gitignore properly configured
- [ ] Secrets properly managed
- [ ] Error handling implemented
- [ ] Data caching optimized
- [ ] Mobile-responsive design
- [ ] Analytics tracking (optional)
- [ ] Custom domain configured (optional)

### Cost Optimization

**Serverless Options:**
- Streamlit Cloud: Free for public apps
- Railway: $5/month for starter
- Render: Free tier available
- Vercel: Free for personal projects

**Upgrade When Needed:**
- Streamlit Cloud Pro: $20/month
- Railway Pro: $20/month  
- Render Standard: $7/month

### Support

For deployment issues:
- Streamlit Forum: https://discuss.streamlit.io
- GitHub Issues: Create issue in your repo
- Email: your-email@example.com

---

## 📊 Deployment Success Metrics

Track these after deployment:
- [ ] Uptime > 99%
- [ ] Page load < 3 seconds
- [ ] Data refresh working
- [ ] Alerts functioning
- [ ] No memory errors
- [ ] Mobile access working

**Your app will be live at:** `https://[your-app-name].streamlit.app`

🎉 **Congratulations on deploying your AI trading system!**
