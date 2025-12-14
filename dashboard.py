"""
Streamlit Dashboard for NIFTY50 Ensemble Model
Interactive visualization inspired by Nothing Brand Identity
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent))

# Optional imports - dashboard works without them
try:
    from src.models.dual_lstm import DualLSTMModel
    from src.agents.kelly_agent import KellyCriterionAgent
    from src.utils.preprocessing import DataPreprocessor
    from src.backtesting import Backtester
    MODELS_AVAILABLE = True
except ImportError as e:
    MODELS_AVAILABLE = False
    print(f"Note: Some modules not available: {e}")

# Import live price fetcher
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("Note: yfinance not available - install with: pip install yfinance")

# ==================== NOTHING BRAND DESIGN SYSTEM ====================
# Color Palette (Nothing Brand Identity + Dot Matrix Red)
NOTHING_COLORS = {
    'black': '#000000',
    'white': '#FFFFFF',
    'red': '#D71921',  # Dot matrix red (matching your example)
    'gray_dark': '#1A1A1A',
    'gray_mid': '#333333',
    'gray_light': '#808080',
}

# Dot Matrix Font CSS (Inspired by Nothing's Glyph Interface)
DOT_MATRIX_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Doto:wght@400;600;900&family=Share+Tech+Mono&display=swap');
    
    /* Hide Streamlit branding and sidebar */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    [data-testid="stSidebar"] {display: none;}
    
    /* Global Dark Theme */
    .stApp {
        background-color: #000000;
        color: #FFFFFF;
        font-family: 'Share Tech Mono', monospace;
    }
    
    /* Main Title - True Dot Matrix Style */
    .main-title {
        font-family: 'Doto', monospace;
        font-size: 4rem;
        font-weight: 900;
        color: #D71921;
        text-align: center;
        letter-spacing: 0.5rem;
        text-transform: uppercase;
        margin: 2rem 0;
        line-height: 1.2;
    }
    
    /* Subtitle */
    .subtitle {
        font-family: 'Doto', monospace;
        font-size: 1rem;
        font-weight: 600;
        color: #D71921;
        text-align: center;
        letter-spacing: 0.3rem;
        margin-bottom: 3rem;
    }
    
    /* Section Headers */
    .section-header {
        font-family: 'Doto', monospace;
        font-size: 1.8rem;
        font-weight: 600;
        color: #D71921;
        border-bottom: 2px solid #D71921;
        padding-bottom: 0.5rem;
        margin: 2rem 0 1rem 0;
        letter-spacing: 0.3rem;
    }
    
    /* Metrics - Nothing Style */
    [data-testid="stMetricValue"] {
        font-family: 'Doto', monospace;
        font-size: 2.5rem !important;
        font-weight: 600 !important;
        color: #FFFFFF !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-family: 'Doto', monospace;
        font-size: 0.9rem !important;
        font-weight: 600 !important;
        color: #D71921 !important;
        letter-spacing: 0.15rem;
        text-transform: uppercase;
    }
    
    [data-testid="stMetricDelta"] {
        font-family: 'Share Tech Mono', monospace;
        font-size: 1.1rem !important;
    }
    
    /* Alert/Warning Boxes - Dot Matrix Style */
    .stAlert, [data-testid="stExpander"], [data-testid="stAlertContainer"] {
        background-color: #1A1A1A !important;
        border: 2px solid #D71921 !important;
        border-radius: 0px !important;
        color: #FFFFFF !important;
        font-family: 'Doto', monospace !important;
        font-weight: 600 !important;
        letter-spacing: 0.15rem !important;
    }
    
    /* Warning alerts */
    [data-testid="stAlertContainer"] [data-testid="stAlertContentWarning"] {
        background-color: #1A1A1A !important;
        color: #D71921 !important;
    }
    
    /* Success alerts */
    [data-testid="stAlertContainer"] [class*="success"] {
        background-color: #1A1A1A !important;
        color: #00FF00 !important;
        border-color: #00FF00 !important;
    }
    
    /* Info alerts */
    [data-testid="stAlertContainer"] [class*="info"] {
        background-color: #1A1A1A !important;
        color: #808080 !important;
        border-color: #808080 !important;
    }
    
    /* Alert text styling */
    [data-testid="stMarkdownContainer"] p {
        font-family: 'Doto', monospace !important;
        font-weight: 600 !important;
        letter-spacing: 0.15rem !important;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #D71921;
        color: #FFFFFF;
        font-family: 'Doto', monospace;
        border: none;
        border-radius: 0px;
        padding: 0.75rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        letter-spacing: 0.15rem;
        text-transform: uppercase;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        background-color: #FFFFFF;
        color: #000000;
    }
    
    /* Selectbox/Inputs */
    .stSelectbox, .stTextInput {
        font-family: 'Share Tech Mono', monospace;
    }
    
    /* Ticker Badge */
    .ticker-badge {
        display: inline-block;
        background-color: #D71921;
        color: #FFFFFF;
        font-family: 'Doto', monospace;
        font-weight: 900;
        padding: 0.5rem 1.5rem;
        font-size: 1.5rem;
        letter-spacing: 0.3rem;
        border: 2px solid #FFFFFF;
        margin: 1rem 0;
    }
    
    /* Signal Indicators */
    .signal-buy {
        color: #00FF00;
        font-family: 'Doto', monospace;
        font-weight: 900;
        font-size: 2rem;
    }
    
    .signal-sell {
        color: #D71921;
        font-family: 'Doto', monospace;
        font-weight: 900;
        font-size: 2rem;
    }
    
    /* Matrix Grid Background */
    .matrix-bg {
        background-image: 
            repeating-linear-gradient(0deg, transparent, transparent 2px, #1A1A1A 2px, #1A1A1A 4px),
            repeating-linear-gradient(90deg, transparent, transparent 2px, #1A1A1A 2px, #1A1A1A 4px);
        background-size: 20px 20px;
    }
</style>
"""

st.set_page_config(
    page_title="NIFTY50 AI • By Shamique Khan",
    page_icon="⚫",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ===== LIVE PRICE FETCHER =====
@st.cache_data(ttl=60)  # Cache for 60 seconds
def get_live_price(ticker: str) -> float:
    """Fetch live price from Yahoo Finance (NSE)"""
    if not YFINANCE_AVAILABLE:
        return None
    
    try:
        yahoo_ticker = f"{ticker}.NS"
        stock = yf.Ticker(yahoo_ticker)
        
        # Try multiple methods to get current price
        info = stock.info
        price = (
            info.get('currentPrice') or 
            info.get('regularMarketPrice') or 
            info.get('previousClose')
        )
        
        if price is None:
            # Fallback: latest close from history
            hist = stock.history(period='1d')
            if not hist.empty:
                price = hist['Close'].iloc[-1]
        
        return float(price) if price else None
    except Exception as e:
        print(f"Error fetching live price for {ticker}: {e}")
        return None

@st.cache_data(ttl=60)
def get_live_prices_batch(tickers: list) -> dict:
    """Fetch live prices for multiple tickers"""
    if not YFINANCE_AVAILABLE:
        return {}
    
    prices = {}
    for ticker in tickers:
        price = get_live_price(ticker)
        if price:
            prices[ticker] = price
    return prices

# Inject custom CSS
st.markdown(DOT_MATRIX_CSS, unsafe_allow_html=True)

# Main Title - Nothing Style
st.markdown('<h1 class="main-title">NIFTY50 AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">NEURO-SYMBOLIC TRADING • KELLY CRITERION RISK MANAGEMENT</p>', unsafe_allow_html=True)

# Real-time Alerts Section
try:
    import sys
    sys.path.append('src')
    from alerts import AlertEngine
    
    alert_engine = AlertEngine()
except:
    alert_engine = None

# Load latest data (Auto-refresh every 4 hours)
@st.cache_data(ttl=14400)  # Cache for 4 hours (14400 seconds)
def load_data():
    """Load latest market and prediction data. Auto-refreshes every 4 hours."""
    # Check if data exists, if not generate demo data
    market_files = list(Path('data/raw').glob('market_data_*.csv'))
    if not market_files:
        # Generate demo data on first run
        import subprocess
        subprocess.run([sys.executable, 'generate_demo_data_now.py'], 
                      capture_output=True, text=True, timeout=60)
        market_files = list(Path('data/raw').glob('market_data_*.csv'))
    
    # Market data
    if market_files:
        latest_market = max(market_files, key=lambda x: x.stat().st_ctime)
        market_df = pd.read_csv(latest_market, index_col=0, parse_dates=True)
    else:
        market_df = pd.DataFrame()
    
    # Predictions
    pred_file = Path('data/processed/predictions.csv')
    if pred_file.exists():
        predictions_df = pd.read_csv(pred_file)
    else:
        predictions_df = pd.DataFrame()
    
    # Sentiment
    sentiment_files = list(Path('data/processed').glob('daily_sentiment_*.csv'))
    if sentiment_files:
        latest_sentiment = max(sentiment_files, key=lambda x: x.stat().st_ctime)
        sentiment_df = pd.read_csv(latest_sentiment, parse_dates=['date'])
    else:
        sentiment_df = pd.DataFrame()
    
    return market_df, predictions_df, sentiment_df


market_df, predictions_df, sentiment_df = load_data()

# ===== AUTO-UPDATE STATUS BANNER =====
st.markdown("---")
auto_update_col1, auto_update_col2, auto_update_col3 = st.columns([2, 2, 1])

with auto_update_col1:
    # Check if auto-update is running
    import psutil
    auto_update_running = any('auto_update.py' in ' '.join(p.cmdline()) for p in psutil.process_iter(['cmdline']) if p.info['cmdline'])
    
    if auto_update_running:
        st.markdown('<p style="color: #00FF00; font-family: Doto; font-size: 1.2rem;">● AUTO-UPDATE: ACTIVE</p>', unsafe_allow_html=True)
        st.caption("System updates every 4 hours automatically")
    else:
        st.markdown('<p style="color: #808080; font-family: Doto; font-size: 1.2rem;">○ AUTO-UPDATE: INACTIVE</p>', unsafe_allow_html=True)
        st.caption("Start auto-update: `python src/auto_update.py`")

with auto_update_col2:
    # Show last update time
    if not market_df.empty:
        market_files = list(Path('data/raw').glob('market_data_*.csv'))
        if market_files:
            latest_file = max(market_files, key=lambda x: x.stat().st_ctime)
            last_update = datetime.fromtimestamp(latest_file.stat().st_ctime)
            hours_ago = (datetime.now() - last_update).total_seconds() / 3600
            
            st.metric("LAST DATA UPDATE", f"{hours_ago:.1f}h ago")
            st.caption(last_update.strftime("%Y-%m-%d %H:%M:%S"))

with auto_update_col3:
    # Manual refresh button
    if st.button("🔄 REFRESH NOW", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

st.markdown("---")

# Real-time Alerts Display
if not market_df.empty and alert_engine:
    alerts_df = alert_engine.get_all_alerts(market_df, predictions_df)
    
    if not alerts_df.empty:
        st.markdown('<div class="section-header">🚨 LIVE ALERTS</div>', unsafe_allow_html=True)
        
        # Group by severity
        buy_signals = alerts_df[alerts_df['severity'] == 'BUY_SIGNAL']
        warnings = alerts_df[alerts_df['severity'] == 'WARNING']
        opportunities = alerts_df[alerts_df['severity'] == 'OPPORTUNITY']
        
        alert_cols = st.columns(3)
        
        with alert_cols[0]:
            if not buy_signals.empty:
                st.success(f"🟢 **BUY SIGNALS** ({len(buy_signals)})")
                for _, alert in buy_signals.head(3).iterrows():
                    st.markdown(f"• {alert['message']}")
        
        with alert_cols[1]:
            if not warnings.empty:
                st.warning(f"🟡 **WARNINGS** ({len(warnings)})")
                for _, alert in warnings.head(3).iterrows():
                    st.markdown(f"• {alert['message']}")
        
        with alert_cols[2]:
            if not opportunities.empty:
                st.info(f"🟢 **OPPORTUNITIES** ({len(opportunities)})")
                for _, alert in opportunities.head(3).iterrows():
                    st.markdown(f"• {alert['message']}")
        
        st.markdown("---")

# ===== TOP STOCK RECOMMENDATIONS SECTION =====
if not predictions_df.empty:
    st.markdown('<div class="section-header">⭐ TOP STOCK RECOMMENDATIONS</div>', unsafe_allow_html=True)
    
    # Filter BUY signals and sort by model probability
    buy_stocks = predictions_df[predictions_df['action'] == 'BUY'].copy()
    buy_stocks = buy_stocks.sort_values('model_probability', ascending=False)
    
    if not buy_stocks.empty:
        # Fetch live prices for top recommendations
        top_tickers = buy_stocks.head(3)['ticker'].tolist()
        live_prices = get_live_prices_batch(top_tickers)
        
        # Top 3 recommendations
        top_3 = buy_stocks.head(3)
        
        rec_cols = st.columns(3)
        
        for idx, (_, stock) in enumerate(top_3.iterrows()):
            with rec_cols[idx]:
                # Create recommendation card
                rank = idx + 1
                ticker = stock['ticker']
                prob = stock['model_probability']
                
                # Use LIVE price if available, otherwise fallback to stored price
                price = live_prices.get(ticker, stock['latest_price'])
                price_label = "LIVE PRICE" if ticker in live_prices else "LAST PRICE"
                price_color = "#00FF00" if ticker in live_prices else "#FFA500"
                
                kelly = stock['kelly_fraction']
                position = stock['position_size']
                rsi = stock['rsi']
                
                # Confidence level
                if prob > 0.75:
                    confidence = "🟢 VERY HIGH"
                    conf_color = "#00FF00"
                elif prob > 0.70:
                    confidence = "🟢 HIGH"
                    conf_color = "#00FF00"
                else:
                    confidence = "🟡 MODERATE"
                    conf_color = "#FFA500"
                
                # Card HTML
                card_html = f'''<div style="background: linear-gradient(135deg, #1A1A1A 0%, #0D0D0D 100%); border: 2px solid #D71921; border-radius: 10px; padding: 20px; margin: 10px 0; box-shadow: 0 0 20px rgba(215, 25, 33, 0.3);">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                        <span style="background: #D71921; color: #FFFFFF; font-family: 'Doto', monospace; font-weight: 900; padding: 5px 15px; border-radius: 5px; font-size: 1.2rem;">#{rank}</span>
                        <span style="color: {conf_color}; font-family: 'Doto', monospace; font-size: 0.9rem;">{confidence}</span>
                    </div>
                    <h2 style="color: #FFFFFF; font-family: 'Doto', monospace; font-size: 2rem; margin: 10px 0; text-align: center;">{ticker}</h2>
                    <div style="text-align: center; margin: 15px 0;">
                        <p style="color: #808080; font-size: 0.8rem; margin: 0;">{price_label}</p>
                        <p style="color: {price_color}; font-family: 'Share Tech Mono', monospace; font-size: 1.5rem; margin: 5px 0;">₹{price:.2f}</p>
                    </div>
                    <hr style="border: 1px solid #333; margin: 15px 0;">
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin: 15px 0;">
                        <div><p style="color: #808080; font-size: 0.75rem; margin: 0;">AI CONFIDENCE</p><p style="color: #FFFFFF; font-family: 'Share Tech Mono', monospace; font-size: 1.1rem; margin: 5px 0;">{prob:.1%}</p></div>
                        <div><p style="color: #808080; font-size: 0.75rem; margin: 0;">RSI</p><p style="color: #FFFFFF; font-family: 'Share Tech Mono', monospace; font-size: 1.1rem; margin: 5px 0;">{rsi:.1f}</p></div>
                        <div><p style="color: #808080; font-size: 0.75rem; margin: 0;">KELLY %</p><p style="color: #FFFFFF; font-family: 'Share Tech Mono', monospace; font-size: 1.1rem; margin: 5px 0;">{kelly:.1%}</p></div>
                        <div><p style="color: #808080; font-size: 0.75rem; margin: 0;">POSITION</p><p style="color: #00FF00; font-family: 'Share Tech Mono', monospace; font-size: 1.1rem; margin: 5px 0;">₹{position:,.0f}</p></div>
                    </div>
                    <div style="text-align: center; margin-top: 15px;">
                        <span style="background: linear-gradient(90deg, #D71921 0%, #FF3344 100%); color: #FFFFFF; font-family: 'Doto', monospace; font-weight: 900; padding: 10px 20px; border-radius: 5px; font-size: 1rem; letter-spacing: 2px;">🎯 BUY SIGNAL</span>
                    </div>
                </div>'''
                st.markdown(card_html, unsafe_allow_html=True)
        
        # Show all BUY recommendations in table
        st.markdown("---")
        st.markdown('<div class="metric-label">ALL BUY RECOMMENDATIONS</div>', unsafe_allow_html=True)
        
        display_df = buy_stocks[['ticker', 'model_probability', 'latest_price', 'kelly_fraction', 'position_size', 'rsi']].copy()
        display_df.columns = ['TICKER', 'AI CONFIDENCE', 'PRICE (₹)', 'KELLY %', 'POSITION (₹)', 'RSI']
        display_df['AI CONFIDENCE'] = display_df['AI CONFIDENCE'].apply(lambda x: f"{x:.1%}")
        display_df['PRICE (₹)'] = display_df['PRICE (₹)'].apply(lambda x: f"₹{x:.2f}")
        display_df['KELLY %'] = display_df['KELLY %'].apply(lambda x: f"{x:.1%}")
        display_df['POSITION (₹)'] = display_df['POSITION (₹)'].apply(lambda x: f"₹{x:,.0f}")
        display_df['RSI'] = display_df['RSI'].apply(lambda x: f"{x:.1f}")
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.info("⚪ NO BUY SIGNALS CURRENTLY • MARKET IN WAIT MODE")

st.markdown("---")

# Stock selector - Top of page (No sidebar)
if not market_df.empty:
    st.markdown('<div class="section-header">SELECT STOCK</div>', unsafe_allow_html=True)
    
    tickers = sorted(market_df['Ticker'].unique())
    cols = st.columns([1, 3])
    
    with cols[0]:
        selected_ticker = st.selectbox("", tickers, label_visibility="collapsed")
    
    # Display ticker badge
    st.markdown(f'<div class="ticker-badge">{selected_ticker}</div>', unsafe_allow_html=True)
    
    # Filter data for selected stock
    stock_data = market_df[market_df['Ticker'] == selected_ticker].copy()
    
    # Latest metrics row
    latest_close = stock_data['Close'].iloc[-1]
    prev_close = stock_data['Close'].iloc[-2]
    change = latest_close - prev_close
    change_pct = (change / prev_close) * 100
    
    metric_cols = st.columns(4)
    
    with metric_cols[0]:
        st.metric("PRICE", f"₹{latest_close:.2f}", f"{change:+.2f} ({change_pct:+.2f}%)")
    
    with metric_cols[1]:
        st.metric("VOLUME", f"{stock_data['Volume'].iloc[-1]:,.0f}")
    
    with metric_cols[2]:
        if 'RSI_14' in stock_data.columns:
            rsi_val = stock_data['RSI_14'].iloc[-1]
            st.metric("RSI", f"{rsi_val:.1f}")
    
    with metric_cols[3]:
        if 'ATRr_14' in stock_data.columns:
            atr_val = stock_data['ATRr_14'].iloc[-1]
            st.metric("VOLATILITY", f"{atr_val:.2f}")
    
    st.markdown("---")
    
    # ===== MAIN CONTENT =====
    
    # Row 1: Price Chart with Indicators
    st.markdown('<div class="section-header">TECHNICAL ANALYSIS</div>', unsafe_allow_html=True)
    
    # Create candlestick chart with Nothing-inspired dark theme
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=('PRICE & BOLLINGER BANDS', 'RSI MOMENTUM', 'MACD DIVERGENCE'),
        row_heights=[0.5, 0.25, 0.25]
    )
    
    # Candlestick with Nothing colors
    fig.add_trace(
        go.Candlestick(
            x=stock_data.index,
            open=stock_data['Open'],
            high=stock_data['High'],
            low=stock_data['Low'],
            close=stock_data['Close'],
            name='PRICE',
            increasing_line_color='#00FF00',
            decreasing_line_color='#D71921'
        ),
        row=1, col=1
    )
    
    # Bollinger Bands
    if 'BBU_20_2.0' in stock_data.columns:
        fig.add_trace(
            go.Scatter(x=stock_data.index, y=stock_data['BBU_20_2.0'], 
                      name='BB UPPER', line=dict(color='#808080', dash='dash', width=1)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=stock_data.index, y=stock_data['BBL_20_2.0'], 
                      name='BB LOWER', line=dict(color='#808080', dash='dash', width=1)),
            row=1, col=1
        )
    
    # RSI
    if 'RSI_14' in stock_data.columns:
        fig.add_trace(
            go.Scatter(x=stock_data.index, y=stock_data['RSI_14'], 
                      name='RSI', line=dict(color='#FFFFFF', width=2)),
            row=2, col=1
        )
        # Overbought/Oversold lines
        fig.add_hline(y=70, line_dash="dash", line_color="#D71921", line_width=1, row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#00FF00", line_width=1, row=2, col=1)
    
    # MACD
    if 'MACD_12_26_9' in stock_data.columns:
        fig.add_trace(
            go.Scatter(x=stock_data.index, y=stock_data['MACD_12_26_9'], 
                      name='MACD', line=dict(color='#D71921', width=2)),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(x=stock_data.index, y=stock_data['MACDs_12_26_9'], 
                      name='SIGNAL', line=dict(color='#FFFFFF', width=2)),
            row=3, col=1
        )
    
    # Nothing-inspired dark theme for chart
    fig.update_layout(
        height=800, 
        showlegend=True,
        xaxis_rangeslider_visible=False,
        plot_bgcolor='#000000',
        paper_bgcolor='#000000',
        font=dict(family='Share Tech Mono, monospace', color='#FFFFFF', size=12),
        legend=dict(bgcolor='#1A1A1A', bordercolor='#333333', borderwidth=1)
    )
    
    fig.update_xaxes(showgrid=True, gridcolor='#1A1A1A', zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor='#1A1A1A', zeroline=False)
    fig.update_yaxes(title_text="PRICE (₹)", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Row 2: Sentiment Analysis
    st.markdown('<div class="section-header">SENTIMENT INTELLIGENCE</div>', unsafe_allow_html=True)
    
    if not sentiment_df.empty:
        col1, col2, col3 = st.columns(3)
        
        latest_sentiment = sentiment_df.iloc[-1]
        
        with col1:
            sentiment_score = latest_sentiment['sentiment_mean']
            sentiment_emoji = "🟢" if sentiment_score > 0 else "🔴"
            st.metric(
                "SENTIMENT SCORE",
                f"{sentiment_score:+.3f}",
                f"{sentiment_emoji} {'BULLISH' if sentiment_score > 0 else 'BEARISH'}"
            )
        
        with col2:
            st.metric(
                "ARTICLE COUNT",
                int(latest_sentiment['article_count'])
            )
        
        with col3:
            st.metric(
                "DOMINANT TONE",
                latest_sentiment['sentiment_label'].upper()
            )
        
        # Sentiment trend with Nothing theme
        fig_sentiment = go.Figure()
        
        fig_sentiment.add_trace(
            go.Scatter(
                x=sentiment_df['date'],
                y=sentiment_df['sentiment_mean'],
                mode='lines+markers',
                name='DAILY SENTIMENT',
                line=dict(color='#D71921', width=2),
                marker=dict(size=6, color='#FFFFFF', line=dict(color='#D71921', width=1))
            )
        )
        
        if 'sentiment_ma_5' in sentiment_df.columns:
            fig_sentiment.add_trace(
                go.Scatter(
                    x=sentiment_df['date'],
                    y=sentiment_df['sentiment_ma_5'],
                    mode='lines',
                    name='5-DAY MOVING AVG',
                    line=dict(color='#FFFFFF', width=3)
                )
            )
        
        fig_sentiment.add_hline(y=0, line_dash="solid", line_color="#333333", line_width=2)
        fig_sentiment.update_layout(
            title=dict(text="SENTIMENT TREND", font=dict(family='Orbitron', size=16)),
            xaxis_title="DATE",
            yaxis_title="SENTIMENT SCORE",
            height=400,
            plot_bgcolor='#000000',
            paper_bgcolor='#000000',
            font=dict(family='Share Tech Mono', color='#FFFFFF', size=12),
            legend=dict(bgcolor='#1A1A1A', bordercolor='#333333', borderwidth=1)
        )
        fig_sentiment.update_xaxes(showgrid=True, gridcolor='#1A1A1A')
        fig_sentiment.update_yaxes(showgrid=True, gridcolor='#1A1A1A')
        
        st.plotly_chart(fig_sentiment, use_container_width=True)
    
    # Row 3: AI Predictions & Kelly Agent
    st.markdown('<div class="section-header">AI SIGNAL • KELLY CRITERION</div>', unsafe_allow_html=True)
    
    if not predictions_df.empty:
        # Filter for selected ticker
        ticker_pred = predictions_df[predictions_df['ticker'] == selected_ticker]
        
        if not ticker_pred.empty:
            pred = ticker_pred.iloc[0]
            
            # Display action prominently
            action = pred['action']
            if action == "BUY":
                st.markdown(f'<p class="signal-buy">● BUY SIGNAL</p>', unsafe_allow_html=True)
            elif action == "SHORT":
                st.markdown(f'<p class="signal-sell">● SHORT SIGNAL</p>', unsafe_allow_html=True)
            else:
                st.markdown(f'<p style="color: #808080; font-family: Orbitron; font-size: 2rem;">○ WAIT</p>', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("MODEL PROBABILITY", f"{pred['model_probability']:.2%}")
            
            with col2:
                st.metric("KELLY FRACTION", f"{pred['kelly_fraction']:.2%}")
            
            with col3:
                st.metric("POSITION SIZE", f"₹{pred['position_size']:,.0f}")
            
            with col4:
                confidence = "HIGH" if pred['model_probability'] > 0.7 else "MEDIUM" if pred['model_probability'] > 0.6 else "LOW"
                st.metric("CONFIDENCE", confidence)
            
            # Alignment indicator
            st.markdown("---")
            if pred.get('is_aligned', False):
                st.success("✓ TECH & SENTIMENT ALIGNED • FULL POSITION")
            else:
                st.warning("⚠ TECH/SENTIMENT CONFLICT • REDUCED POSITION (-50%)")
        else:
            st.info(f"⚪ NO PREDICTIONS FOR {selected_ticker}")
    else:
        st.info("⚪ RUN PREDICTIONS: `python src/predict.py`")
    
    # ===== BACKTEST RESULTS SECTION =====
    st.markdown("---")
    st.markdown('<div class="section-header">BACKTEST PERFORMANCE</div>', unsafe_allow_html=True)
    
    # Add "Run Backtest" button
    col_btn, col_status = st.columns([1, 3])
    with col_btn:
        run_backtest_btn = st.button("🔄 RUN BACKTEST", key="run_backtest", use_container_width=True)
    
    # Run real-time backtest if button clicked
    if run_backtest_btn:
        with st.spinner("⚙ Running backtest..."):
            try:
                # Initialize backtester
                backtester = Backtester(initial_capital=100000, commission=0.001)
                
                # Run backtest
                results = backtester.run_backtest(market_df, predictions_df)
                
                if 'error' not in results:
                    # Save results
                    from datetime import datetime
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    Path('data/results').mkdir(exist_ok=True, parents=True)
                    results['trades_df'].to_csv(f"data/results/backtest_trades_{timestamp}.csv", index=False)
                    
                    # Create equity curve plot
                    import matplotlib.pyplot as plt
                    import matplotlib
                    matplotlib.use('Agg')
                    
                    fig, ax = plt.subplots(figsize=(12, 6), facecolor='#000000')
                    ax.set_facecolor('#000000')
                    
                    portfolio_df = pd.Series(results['portfolio_history'])
                    ax.plot(portfolio_df.values, color='#00FF00', linewidth=2, label='Portfolio Value')
                    ax.axhline(y=100000, color='#808080', linestyle='--', linewidth=1, label='Initial Capital')
                    ax.fill_between(range(len(portfolio_df)), portfolio_df.values, 100000, 
                                   where=(portfolio_df.values >= 100000), alpha=0.3, color='#00FF00')
                    ax.fill_between(range(len(portfolio_df)), portfolio_df.values, 100000,
                                   where=(portfolio_df.values < 100000), alpha=0.3, color='#D71921')
                    
                    ax.set_xlabel('Days', color='#FFFFFF', fontsize=12)
                    ax.set_ylabel('Portfolio Value (₹)', color='#FFFFFF', fontsize=12)
                    ax.set_title('EQUITY CURVE', color='#D71921', fontsize=16, fontweight='bold', pad=20)
                    ax.tick_params(colors='#FFFFFF')
                    ax.spines['bottom'].set_color('#808080')
                    ax.spines['top'].set_color('#808080')
                    ax.spines['left'].set_color('#808080')
                    ax.spines['right'].set_color('#808080')
                    ax.grid(True, alpha=0.2, color='#808080')
                    ax.legend(facecolor='#1A1A1A', edgecolor='#808080', labelcolor='#FFFFFF')
                    
                    plt.tight_layout()
                    plot_path = f"data/results/backtest_{timestamp}.png"
                    plt.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='#000000')
                    plt.close()
                    
                    with col_status:
                        st.success(f"✓ Backtest completed! {results['total_trades']} trades executed")
                else:
                    with col_status:
                        st.error(f"❌ Error: {results['error']}")
            except Exception as e:
                with col_status:
                    st.error(f"❌ Backtest failed: {str(e)}")
    
    # Display backtest results
    backtest_files = list(Path('data/results').glob('backtest_trades_*.csv'))
    backtest_images = list(Path('data/results').glob('backtest_*.png'))
    
    if backtest_files:
        latest_backtest = max(backtest_files, key=lambda x: x.stat().st_ctime)
        backtest_df = pd.read_csv(latest_backtest)
        
        # Calculate metrics
        total_trades = len(backtest_df)
        winning_trades = (backtest_df['pnl'] > 0).sum()
        losing_trades = total_trades - winning_trades
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        avg_win = backtest_df[backtest_df['pnl'] > 0]['pnl_pct'].mean() if (backtest_df['pnl'] > 0).any() else 0
        avg_loss = backtest_df[backtest_df['pnl'] < 0]['pnl_pct'].mean() if (backtest_df['pnl'] < 0).any() else 0
        total_pnl = backtest_df['pnl'].sum()
        total_return = (total_pnl / 100000) * 100  # Assuming 100k initial capital
        
        # Calculate max drawdown
        cumulative_pnl = backtest_df['pnl'].cumsum()
        running_max = cumulative_pnl.expanding().max()
        drawdown = cumulative_pnl - running_max
        max_drawdown = (drawdown.min() / 100000) * 100 if len(drawdown) > 0 else 0
        
        # Display metrics in two rows
        bt_cols1 = st.columns(4)
        with bt_cols1[0]:
            st.metric("TOTAL RETURN", f"{total_return:+.2f}%")
        with bt_cols1[1]:
            st.metric("TOTAL TRADES", total_trades)
        with bt_cols1[2]:
            st.metric("WIN RATE", f"{win_rate:.1f}%")
        with bt_cols1[3]:
            st.metric("TOTAL P&L", f"₹{total_pnl:,.0f}")
        
        bt_cols2 = st.columns(4)
        with bt_cols2[0]:
            st.metric("WINNERS", winning_trades)
        with bt_cols2[1]:
            st.metric("LOSERS", losing_trades)
        with bt_cols2[2]:
            st.metric("AVG WIN", f"{avg_win:+.2f}%")
        with bt_cols2[3]:
            st.metric("MAX DRAWDOWN", f"{max_drawdown:.2f}%")
        
        # Show backtest equity curve chart if available
        if backtest_images:
            st.markdown('<div class="metric-label">EQUITY CURVE</div>', unsafe_allow_html=True)
            latest_image = max(backtest_images, key=lambda x: x.stat().st_ctime)
            st.image(str(latest_image), use_container_width=True)
        
        # Show recent trades
        st.markdown('<div class="metric-label">RECENT TRADES</div>', unsafe_allow_html=True)
        recent_trades = backtest_df.tail(10)[['ticker', 'entry_date', 'exit_date', 'pnl_pct', 'exit_reason']]
        recent_trades['pnl_pct'] = recent_trades['pnl_pct'].apply(lambda x: f"{x:+.2f}%")
        st.dataframe(recent_trades, use_container_width=True, hide_index=True)
    else:
        st.info("⚪ Click 'RUN BACKTEST' button above or run: `python src/backtesting.py`")
    
    st.markdown("---")
    
    # Advanced Quantitative Analytics Section
    st.markdown('<div class="section-header">ADVANCED QUANT ANALYTICS</div>', unsafe_allow_html=True)
    st.markdown('<p style="color: #808080; font-family: Share Tech Mono; font-size: 0.9rem; margin-bottom: 1.5rem;">Time-Series Intuition • Stochastic Calculus • Risk Decomposition</p>', unsafe_allow_html=True)
    
    if not stock_data.empty and len(stock_data) > 30:
        try:
            from src.quant_analytics import QuantAnalytics
            
            qa = QuantAnalytics()
            prices = stock_data['Close'].dropna()
            
            # Run comprehensive analysis
            analysis = qa.comprehensive_analysis(prices)
            
            # Section 1: Return Distribution & Fat Tails
            st.markdown('<div style="background: linear-gradient(135deg, #1A1A1A 0%, #0D0D0D 100%); padding: 1.5rem; border-radius: 8px; border: 1px solid #D71921; margin-bottom: 1rem;">', unsafe_allow_html=True)
            st.markdown('<p style="color: #D71921; font-family: Doto; font-size: 1.1rem; font-weight: bold; margin-bottom: 1rem;">📊 RETURN DISTRIBUTION • FAT TAILS ANALYSIS</p>', unsafe_allow_html=True)
            
            ret_analysis = analysis['returns_analysis']
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f'<div style="text-align: center;"><p style="color: #808080; font-size: 0.8rem; margin-bottom: 0.3rem;">SKEWNESS</p><p style="color: #FFFFFF; font-size: 1.3rem; font-weight: bold;">{ret_analysis["skewness"]:.3f}</p></div>', unsafe_allow_html=True)
            with col2:
                st.markdown(f'<div style="text-align: center;"><p style="color: #808080; font-size: 0.8rem; margin-bottom: 0.3rem;">EXCESS KURTOSIS</p><p style="color: #FFFFFF; font-size: 1.3rem; font-weight: bold;">{ret_analysis["excess_kurtosis"]:.3f}</p></div>', unsafe_allow_html=True)
            with col3:
                tail_color = "#D71921" if "FAT" in ret_analysis["tail_type"] else "#00FF00"
                st.markdown(f'<div style="text-align: center;"><p style="color: #808080; font-size: 0.8rem; margin-bottom: 0.3rem;">TAIL TYPE</p><p style="color: {tail_color}; font-size: 0.9rem; font-weight: bold;">{ret_analysis["tail_type"]}</p></div>', unsafe_allow_html=True)
            with col4:
                st.markdown(f'<div style="text-align: center;"><p style="color: #808080; font-size: 0.8rem; margin-bottom: 0.3rem;">VaR 95%</p><p style="color: #D71921; font-size: 1.3rem; font-weight: bold;">{ret_analysis["var_95"]*100:.2f}%</p></div>', unsafe_allow_html=True)
            
            st.markdown('<p style="color: #FFFFFF; font-size: 0.85rem; margin-top: 1rem;"><strong>Interpretation:</strong> ' + 
                       (f'Fat tails detected (excess kurtosis {ret_analysis["excess_kurtosis"]:.2f} > 2) - expect extreme moves more often than normal distribution. ' if ret_analysis["excess_kurtosis"] > 2 else 'Normal tail behavior - returns follow expected distribution. ') +
                       (f'Negative skew ({ret_analysis["skewness"]:.2f}) means large losses more likely than large gains.' if ret_analysis["skewness"] < -0.5 else f'Positive skew ({ret_analysis["skewness"]:.2f}) means large gains more likely than large losses.' if ret_analysis["skewness"] > 0.5 else 'Symmetric return distribution.') +
                       '</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Section 2: Volatility Clustering
            st.markdown('<div style="background: linear-gradient(135deg, #1A1A1A 0%, #0D0D0D 100%); padding: 1.5rem; border-radius: 8px; border: 1px solid #D71921; margin-bottom: 1rem;">', unsafe_allow_html=True)
            st.markdown('<p style="color: #D71921; font-family: Doto; font-size: 1.1rem; font-weight: bold; margin-bottom: 1rem;">⚡ VOLATILITY CLUSTERING • REGIME DETECTION</p>', unsafe_allow_html=True)
            
            vol_cluster = analysis['volatility_clustering']
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f'<div style="text-align: center;"><p style="color: #808080; font-size: 0.8rem; margin-bottom: 0.3rem;">CURRENT VOL</p><p style="color: #FFFFFF; font-size: 1.3rem; font-weight: bold;">{vol_cluster["current_vol"]*100:.2f}%</p></div>', unsafe_allow_html=True)
            with col2:
                st.markdown(f'<div style="text-align: center;"><p style="color: #808080; font-size: 0.8rem; margin-bottom: 0.3rem;">AVG VOL</p><p style="color: #FFFFFF; font-size: 1.3rem; font-weight: bold;">{vol_cluster["avg_vol"]*100:.2f}%</p></div>', unsafe_allow_html=True)
            with col3:
                regime_color = "#D71921" if "HIGH" in vol_cluster["regime"] else "#00FF00" if "LOW" in vol_cluster["regime"] else "#FFA500"
                st.markdown(f'<div style="text-align: center;"><p style="color: #808080; font-size: 0.8rem; margin-bottom: 0.3rem;">REGIME</p><p style="color: {regime_color}; font-size: 0.9rem; font-weight: bold;">{vol_cluster["regime"]}</p></div>', unsafe_allow_html=True)
            with col4:
                st.markdown(f'<div style="text-align: center;"><p style="color: #808080; font-size: 0.8rem; margin-bottom: 0.3rem;">CLUSTERING</p><p style="color: #FFFFFF; font-size: 1.3rem; font-weight: bold;">{vol_cluster["clustering_coef"]:.3f}</p></div>', unsafe_allow_html=True)
            
            st.markdown('<p style="color: #FFFFFF; font-size: 0.85rem; margin-top: 1rem;"><strong>Interpretation:</strong> ' +
                       ('High volatility clustering detected (coef > 0.3) - volatility shocks persist, adjust position sizing. ' if vol_cluster["clustering_coef"] > 0.3 else 'Low volatility persistence - independent daily moves. ') +
                       f'Current vol is {vol_cluster["vol_ratio"]:.1f}x average. Risk level: {vol_cluster["risk_level"]}.' +
                       '</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Section 3: Mean Reversion
            st.markdown('<div style="background: linear-gradient(135deg, #1A1A1A 0%, #0D0D0D 100%); padding: 1.5rem; border-radius: 8px; border: 1px solid #D71921; margin-bottom: 1rem;">', unsafe_allow_html=True)
            st.markdown('<p style="color: #D71921; font-family: Doto; font-size: 1.1rem; font-weight: bold; margin-bottom: 1rem;">🔄 MEAN REVERSION • STATIONARITY TEST</p>', unsafe_allow_html=True)
            
            mean_rev = analysis['mean_reversion']
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f'<div style="text-align: center;"><p style="color: #808080; font-size: 0.8rem; margin-bottom: 0.3rem;">ADF STATISTIC</p><p style="color: #FFFFFF; font-size: 1.3rem; font-weight: bold;">{mean_rev["adf_stat"]:.3f}</p></div>', unsafe_allow_html=True)
            with col2:
                st.markdown(f'<div style="text-align: center;"><p style="color: #808080; font-size: 0.8rem; margin-bottom: 0.3rem;">P-VALUE</p><p style="color: #FFFFFF; font-size: 1.3rem; font-weight: bold;">{mean_rev["adf_pvalue"]:.4f}</p></div>', unsafe_allow_html=True)
            with col3:
                stat_color = "#00FF00" if mean_rev["is_stationary"] else "#D71921"
                st.markdown(f'<div style="text-align: center;"><p style="color: #808080; font-size: 0.8rem; margin-bottom: 0.3rem;">STATIONARY</p><p style="color: {stat_color}; font-size: 1.1rem; font-weight: bold;">{"YES" if mean_rev["is_stationary"] else "NO"}</p></div>', unsafe_allow_html=True)
            with col4:
                half_life_display = f"{mean_rev['half_life']:.1f}" if mean_rev['half_life'] != np.inf else "∞"
                st.markdown(f'<div style="text-align: center;"><p style="color: #808080; font-size: 0.8rem; margin-bottom: 0.3rem;">HALF-LIFE (days)</p><p style="color: #FFFFFF; font-size: 1.3rem; font-weight: bold;">{half_life_display}</p></div>', unsafe_allow_html=True)
            
            st.markdown('<p style="color: #FFFFFF; font-size: 0.85rem; margin-top: 1rem;"><strong>Trading Signal:</strong> ' +
                       f'{mean_rev["trading_signal"]} ' +
                       ('ADF p-value < 0.05 confirms stationarity - price mean-reverts. ' if mean_rev["is_stationary"] else 'Non-stationary - price trends without reverting to mean. ') +
                       ('Short half-life indicates fast reversion.' if mean_rev["half_life"] < 30 and mean_rev["half_life"] != np.inf else '') +
                       '</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Section 4: Regime Shifts
            st.markdown('<div style="background: linear-gradient(135deg, #1A1A1A 0%, #0D0D0D 100%); padding: 1.5rem; border-radius: 8px; border: 1px solid #D71921; margin-bottom: 1rem;">', unsafe_allow_html=True)
            st.markdown('<p style="color: #D71921; font-family: Doto; font-size: 1.1rem; font-weight: bold; margin-bottom: 1rem;">🎯 MARKET REGIME • TREND DETECTION</p>', unsafe_allow_html=True)
            
            regime = analysis['regime_shifts']
            
            col1, col2 = st.columns(2)
            with col1:
                trend_color = "#00FF00" if "BULL" in regime["trend"] else "#D71921" if "BEAR" in regime["trend"] else "#FFA500"
                st.markdown(f'<div style="text-align: center; background: #000000; padding: 1rem; border-radius: 6px; border: 2px solid {trend_color};">' +
                           f'<p style="color: #808080; font-size: 0.8rem; margin-bottom: 0.3rem;">TREND REGIME</p>' +
                           f'<p style="color: {trend_color}; font-size: 1.5rem; font-weight: bold;">{regime["trend"]}</p>' +
                           f'<p style="color: #FFFFFF; font-size: 0.9rem; margin-top: 0.5rem;">Mean Return: {regime["current_mean"]*100:.2f}%</p>' +
                           '</div>', unsafe_allow_html=True)
            with col2:
                vol_regime_color = "#D71921" if "HIGH" in regime["vol_regime"] else "#00FF00" if "LOW" in regime["vol_regime"] else "#FFA500"
                st.markdown(f'<div style="text-align: center; background: #000000; padding: 1rem; border-radius: 6px; border: 2px solid {vol_regime_color};">' +
                           f'<p style="color: #808080; font-size: 0.8rem; margin-bottom: 0.3rem;">VOLATILITY REGIME</p>' +
                           f'<p style="color: {vol_regime_color}; font-size: 1.1rem; font-weight: bold;">{regime["vol_regime"]}</p>' +
                           f'<p style="color: #FFFFFF; font-size: 0.9rem; margin-top: 0.5rem;">Std Dev: {regime["current_std"]*100:.2f}%</p>' +
                           '</div>', unsafe_allow_html=True)
            
            st.markdown(f'<p style="color: #FFFFFF; font-size: 0.95rem; margin-top: 1rem; text-align: center; background: #1A1A1A; padding: 0.8rem; border-radius: 6px;"><strong>📌 RECOMMENDATION:</strong> {regime["recommendation"]}</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Section 5: GBM Parameters (Stochastic Calculus)
            st.markdown('<div style="background: linear-gradient(135deg, #1A1A1A 0%, #0D0D0D 100%); padding: 1.5rem; border-radius: 8px; border: 1px solid #D71921; margin-bottom: 1rem;">', unsafe_allow_html=True)
            st.markdown('<p style="color: #D71921; font-family: Doto; font-size: 1.1rem; font-weight: bold; margin-bottom: 1rem;">🎲 STOCHASTIC CALCULUS • GBM PARAMETERS</p>', unsafe_allow_html=True)
            st.markdown('<p style="color: #808080; font-size: 0.85rem; margin-bottom: 1rem;">Geometric Brownian Motion: dS = μ·S·dt + σ·S·dW</p>', unsafe_allow_html=True)
            
            gbm = analysis['gbm_parameters']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                drift_color = "#00FF00" if gbm["drift"] > 0 else "#D71921"
                st.markdown(f'<div style="text-align: center; background: #000000; padding: 1rem; border-radius: 6px;">' +
                           f'<p style="color: #808080; font-size: 0.8rem; margin-bottom: 0.3rem;">DRIFT (μ)</p>' +
                           f'<p style="color: {drift_color}; font-size: 1.8rem; font-weight: bold;">{gbm["drift"]*100:.2f}%</p>' +
                           f'<p style="color: #FFFFFF; font-size: 0.8rem; margin-top: 0.5rem;">{gbm["drift_type"]}</p>' +
                           '</div>', unsafe_allow_html=True)
            with col2:
                st.markdown(f'<div style="text-align: center; background: #000000; padding: 1rem; border-radius: 6px;">' +
                           f'<p style="color: #808080; font-size: 0.8rem; margin-bottom: 0.3rem;">DIFFUSION (σ)</p>' +
                           f'<p style="color: #FFA500; font-size: 1.8rem; font-weight: bold;">{gbm["diffusion"]*100:.2f}%</p>' +
                           f'<p style="color: #FFFFFF; font-size: 0.8rem; margin-top: 0.5rem;">{gbm["vol_type"]}</p>' +
                           '</div>', unsafe_allow_html=True)
            with col3:
                sharpe_color = "#00FF00" if gbm["sharpe_ratio"] > 1 else "#FFA500" if gbm["sharpe_ratio"] > 0 else "#D71921"
                st.markdown(f'<div style="text-align: center; background: #000000; padding: 1rem; border-radius: 6px;">' +
                           f'<p style="color: #808080; font-size: 0.8rem; margin-bottom: 0.3rem;">SHARPE RATIO</p>' +
                           f'<p style="color: {sharpe_color}; font-size: 1.8rem; font-weight: bold;">{gbm["sharpe_ratio"]:.2f}</p>' +
                           f'<p style="color: #FFFFFF; font-size: 0.8rem; margin-top: 0.5rem;">{"EXCELLENT" if gbm["sharpe_ratio"] > 2 else "GOOD" if gbm["sharpe_ratio"] > 1 else "FAIR" if gbm["sharpe_ratio"] > 0 else "POOR"}</p>' +
                           '</div>', unsafe_allow_html=True)
            
            st.markdown('<p style="color: #FFFFFF; font-size: 0.85rem; margin-top: 1rem;"><strong>Interpretation:</strong> ' +
                       f'Drift (μ = {gbm["drift"]*100:.2f}%) represents expected annual return direction. ' +
                       f'Diffusion (σ = {gbm["diffusion"]*100:.2f}%) is annualized volatility from Brownian motion randomness. ' +
                       f'Sharpe ratio {gbm["sharpe_ratio"]:.2f} indicates risk-adjusted return quality.' +
                       '</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"⚠ Analytics Error: {str(e)}")
            st.info("Need at least 30 days of data for comprehensive analysis")
    else:
        st.info("⚪ Need minimum 30 days of market data for quantitative analysis")
    
else:
    st.warning("⚠ NO DATA FOUND")
    
    st.markdown('<div class="section-header">QUICK START</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### INSTALLATION
    ```bash
    pip install -r requirements.txt
    ```
    
    ### DATA COLLECTION
    ```bash
    python src/data_collection/market_data.py
    python src/data_collection/news_scraper.py
    python src/sentiment/finbert_engine.py
    ```
    
    ### TRAINING
    ```bash
    python src/train.py
    ```
    
    ### PREDICTION
    ```bash
    python src/predict.py
    ```
    
    ### BACKTEST
    ```bash
    python src/backtesting.py
    ```
    
    ### LAUNCH DASHBOARD
    ```bash
    streamlit run dashboard.py
    ```
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #808080; font-family: 'Share Tech Mono'; font-size: 0.8rem; letter-spacing: 0.1rem; margin-top: 3rem;">
    NIFTY50 ENSEMBLE • BI-LSTM + FINBERT-INDIA • KELLY CRITERION RISK MGMT<br>
    COST: ₹0 • 100% OPEN-SOURCE • Developed by Shamique Khan<br>
    <a href="https://www.linkedin.com/in/shamique-khan/" style="color: #D71921; text-decoration: none;">LinkedIn</a> •
</div>
""", unsafe_allow_html=True)
