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
    # Market data
    market_files = list(Path('data/raw').glob('market_data_*.csv'))
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
