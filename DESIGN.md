# NIFTY50 AI Dashboard - Nothing Design System

## Design Philosophy

Inspired by **Nothing's** minimalist, tech-forward brand identity:

### Color Palette
```
PRIMARY:   #000000 (Pure Black)
SECONDARY: #FFFFFF (Pure White)  
ACCENT:    #FF0000 (Signal Red)
GRID:      #1A1A1A (Dark Gray)
TEXT:      #808080 (Medium Gray)
```

### Typography
- **Headers**: Orbitron (900 weight) - Dot matrix inspired
- **Body**: Share Tech Mono - Monospace technical feel
- **Metrics**: Orbitron (700 weight) with glow effects

### Visual Elements

#### 1. Main Title
```
NIFTY50 AI
```
- 4rem size
- All caps
- Red glow effect (pulsing animation)
- Letter spacing: 0.3rem

#### 2. Ticker Badge
```
┌─────────────┐
│  RELIANCE   │
└─────────────┘
```
- Red background
- White text
- White border (2px)
- Orbitron font

#### 3. Signal Indicators
```
● BUY SIGNAL    (Green with glow)
● SHORT SIGNAL  (Red with glow)
○ WAIT          (Gray, no fill)
```

#### 4. Chart Theme
- Black background
- White/Red line colors
- Gray grid (#1A1A1A)
- Minimal legends
- Clean axes

#### 5. Metrics Display
```
PRICE
₹1,234.56
+12.34 (+1.02%)
```
- Label: Uppercase, gray, small
- Value: Large, white, Orbitron
- Delta: Colored (green/red)

## Layout Structure

```
┌─────────────────────────────────────────────┐
│              NIFTY50 AI                     │
│    NEURO-SYMBOLIC TRADING SYSTEM            │
├─────────────────────────────────────────────┤
│  SELECT STOCK                               │
│  [Dropdown]                                 │
│                                             │
│  ┌─────────┐                                │
│  │RELIANCE │                                │
│  └─────────┘                                │
├─────────────────────────────────────────────┤
│  PRICE    VOLUME    RSI    VOLATILITY       │
│  ₹1234    1.2M      65.3   0.45             │
├─────────────────────────────────────────────┤
│  TECHNICAL ANALYSIS                         │
│  ┌────────────────────────────────────────┐ │
│  │ [Candlestick Chart]                    │ │
│  │ [RSI Chart]                            │ │
│  │ [MACD Chart]                           │ │
│  └────────────────────────────────────────┘ │
├─────────────────────────────────────────────┤
│  SENTIMENT INTELLIGENCE                     │
│  SCORE  ARTICLES  TONE                      │
│  +0.45  12        BULLISH                   │
│  [Sentiment Trend Chart]                    │
├─────────────────────────────────────────────┤
│  AI SIGNAL • KELLY CRITERION                │
│  ● BUY SIGNAL                               │
│  PROB    KELLY    SIZE      CONFIDENCE      │
│  75%     0.38%    ₹38,000   HIGH            │
│  ✓ TECH & SENTIMENT ALIGNED                 │
└─────────────────────────────────────────────┘
```

## Animation Effects

### Glow Animation (Title)
```css
@keyframes glow {
  from { text-shadow: 0 0 10px #FF0000; }
  to   { text-shadow: 0 0 40px #FF0000; }
}
```

### Hover Effects (Buttons)
```css
button:hover {
  background: #FFFFFF;
  color: #000000;
  box-shadow: 0 0 20px #FF0000;
}
```

## Accessibility

- High contrast (Black/White)
- Clear font hierarchy
- Readable at all sizes
- Color-blind friendly signals (uses shapes too)

## Mobile Responsive

- Stacks vertically on small screens
- Maintains font readability
- Preserves Nothing aesthetic

## Customization

Edit `.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#FF0000"      # Accent color
backgroundColor = "#000000"    # Main background
secondaryBackgroundColor = "#1A1A1A"  # Cards
textColor = "#FFFFFF"          # Text
font = "monospace"             # Font family
```

Edit custom CSS in `dashboard.py`:
- Search for `DOT_MATRIX_CSS` variable
- Modify colors, fonts, sizes
- Add new animations

## Deployment

The design is optimized for:
- **Streamlit Cloud** (free hosting)
- **Heroku** (with buildpack)
- **Docker** (containerized)
- **Local** (instant preview)

See [DEPLOYMENT.md](DEPLOYMENT.md) for details.
