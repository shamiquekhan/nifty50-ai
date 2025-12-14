# NIFTY50 AI Dashboard - Deployment Guide

## Streamlit Cloud Deployment (FREE)

### 1. Prepare Your Repository

Ensure your GitHub repository has:
- `dashboard.py` (main app file)
- `requirements.txt`
- `.streamlit/config.toml`
- All `src/` modules

### 2. Deploy to Streamlit Cloud

1. **Sign up**: Go to [streamlit.io/cloud](https://streamlit.io/cloud)
2. **Connect GitHub**: Link your GitHub account
3. **New app**: Click "New app"
4. **Configure**:
   - Repository: `your-username/Nifty50Qualtml`
   - Branch: `main`
   - Main file path: `dashboard.py`
5. **Deploy**: Click "Deploy!"

### 3. App URL
Your app will be live at: `https://your-app-name.streamlit.app`

### 4. Update Data
The app will automatically reload when you push updates to GitHub.

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
streamlit run dashboard.py
```

Access at: `http://localhost:8501`

## Design Features

### Nothing Brand Identity Integration:
- ⚫ Pure black background (#000000)
- ⚪ White text (#FFFFFF) 
- 🔴 Red accent (#FF0000)
- 🔲 Dot matrix fonts (Orbitron, Share Tech Mono)
- ⬛ Minimalist grid-based layout
- 💡 Glow effects on interactive elements

### No Sidebar Design:
- All controls in main view
- Maximized chart space
- Ticker badge with Nothing-style border
- Matrix grid background pattern

## Performance Tips

1. **Caching**: Data loading is cached with `@st.cache_data`
2. **Lazy Loading**: Charts render only when data is available
3. **Efficient Updates**: Use `st.rerun()` for partial updates

## Customization

Edit `.streamlit/config.toml` to adjust:
- Primary color (accent)
- Background colors
- Font family
- Server settings
