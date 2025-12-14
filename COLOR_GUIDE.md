# 🎨 Nothing Design System - Color Palette

## Primary Colors

```
┌─────────────────────────────────────────────────────┐
│                                                     │
│  ███████████████  #000000  PURE BLACK             │
│  ███████████████  Primary Background               │
│  ███████████████  Represents: The void, clarity    │
│                                                     │
├─────────────────────────────────────────────────────┤
│                                                     │
│  □□□□□□□□□□□□□□□  #FFFFFF  PURE WHITE             │
│  □□□□□□□□□□□□□□□  Primary Text                    │
│  □□□□□□□□□□□□□□□  Represents: Truth, data          │
│                                                     │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  #FF0000  SIGNAL RED             │
│  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  Accent/Alerts                   │
│  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  Represents: Action, urgency      │
│                                                     │
└─────────────────────────────────────────────────────┘
```

## Supporting Colors

```
┌─────────────────────────────────────────────────────┐
│  ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒  #1A1A1A  DARK GRAY              │
│  Grid lines, card backgrounds, subtle divisions     │
│                                                     │
│  ░░░░░░░░░░░░░░░  #333333  MID GRAY               │
│  Borders, secondary elements                        │
│                                                     │
│  ················  #808080  LIGHT GRAY             │
│  Labels, secondary text, disabled states           │
└─────────────────────────────────────────────────────┘
```

## Semantic Colors

```
┌─────────────────────────────────────────────────────┐
│  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  #00FF00  SUCCESS GREEN          │
│  Buy signals, positive sentiment, gains            │
│                                                     │
│  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  #FF0000  DANGER RED             │
│  Sell signals, negative sentiment, losses          │
│                                                     │
│  ░░░░░░░░░░░░░░░  #808080  NEUTRAL GRAY           │
│  Wait signals, neutral sentiment                   │
└─────────────────────────────────────────────────────┘
```

## Usage Examples

### Main Title
```css
Color: #FFFFFF
Background: #000000
Glow: 0 0 20px #FF0000
Font: Orbitron 900
```

### Section Headers
```css
Color: #FF0000
Background: #000000
Border-bottom: 2px solid #FF0000
Font: Orbitron 700
```

### Metrics
```css
Label: #808080 (Share Tech Mono)
Value: #FFFFFF (Orbitron 700)
Glow: 0 0 10px #FF0000
Delta-positive: #00FF00
Delta-negative: #FF0000
```

### Charts
```css
Background: #000000
Grid: #1A1A1A
Line (primary): #FF0000
Line (secondary): #FFFFFF
Candle-up: #00FF00
Candle-down: #FF0000
```

### Buttons
```css
Default:
  Background: #FF0000
  Text: #FFFFFF
  Border: none

Hover:
  Background: #FFFFFF
  Text: #000000
  Shadow: 0 0 20px #FF0000
```

### Cards/Containers
```css
Background: #1A1A1A
Border: 1px solid #333333
Text: #FFFFFF
```

## Color Psychology

### Black (#000000)
- **Meaning**: Sophistication, power, elegance
- **Use**: Background, primary surface
- **Effect**: Creates focus, reduces eye strain

### White (#FFFFFF)
- **Meaning**: Purity, clarity, simplicity
- **Use**: Text, data, important elements
- **Effect**: Maximum readability, draws attention

### Red (#FF0000)
- **Meaning**: Urgency, action, importance
- **Use**: Accents, signals, calls-to-action
- **Effect**: Demands attention, creates energy

### Gray (#808080)
- **Meaning**: Neutrality, balance, sophistication
- **Use**: Labels, secondary text, dividers
- **Effect**: Hierarchy, subtle guidance

## Contrast Ratios

```
White on Black:  21:1  ✅ AAA (Perfect)
Red on Black:    5.3:1 ✅ AA (Good)
Gray on Black:   3.9:1 ✅ AA (Acceptable)
Green on Black:  7.2:1 ✅ AAA (Excellent)
```

All combinations meet WCAG 2.1 accessibility standards!

## Color Variations

### Hover States
```
Red Hover:     #FF0000 → #FFFFFF (inverse)
White Hover:   #FFFFFF → #FF0000 (accent)
Gray Hover:    #808080 → #FFFFFF (brighten)
```

### Active States
```
Red Active:    #CC0000 (darker)
White Active:  #EEEEEE (slightly dimmed)
```

### Disabled States
```
All colors → #333333 (mid gray)
```

## Gradients (Sparingly Used)

### Title Glow
```css
from: 0 0 10px #FF0000
to:   0 0 40px #FF0000
```

### Chart Highlights
```css
from: rgba(255, 0, 0, 0.2)
to:   rgba(255, 0, 0, 0.0)
```

## Dark Mode Only

This design system is **exclusively dark mode**.

Why?
- Reduces eye strain for data-heavy interfaces
- Better for extended viewing sessions
- Nothing brand aesthetic
- Professional trading terminal feel
- Battery saving on OLED screens

## Export Formats

### CSS Variables
```css
:root {
  --color-black: #000000;
  --color-white: #FFFFFF;
  --color-red: #FF0000;
  --color-gray-dark: #1A1A1A;
  --color-gray-mid: #333333;
  --color-gray-light: #808080;
  --color-success: #00FF00;
  --color-danger: #FF0000;
}
```

### Python Dictionary
```python
NOTHING_COLORS = {
    'black': '#000000',
    'white': '#FFFFFF',
    'red': '#FF0000',
    'gray_dark': '#1A1A1A',
    'gray_mid': '#333333',
    'gray_light': '#808080',
    'success': '#00FF00',
    'danger': '#FF0000',
}
```

### Streamlit Theme (TOML)
```toml
[theme]
primaryColor = "#FF0000"
backgroundColor = "#000000"
secondaryBackgroundColor = "#1A1A1A"
textColor = "#FFFFFF"
```

## Accessibility Notes

1. **High Contrast**: 21:1 ratio ensures readability
2. **Color Blindness**: Shapes and labels supplement colors
3. **Screen Readers**: Semantic HTML maintained
4. **Focus Indicators**: Red outlines on interactive elements
5. **Text Size**: Minimum 12px, scales with zoom

## Brand Alignment

This palette directly reflects Nothing's design philosophy:

- **Minimalism**: Only essential colors (3 primary)
- **Transparency**: Pure, unprocessed colors
- **Tech-Forward**: High contrast, digital aesthetic
- **Functional**: Each color serves a purpose
- **Distinctive**: Immediately recognizable

---

**Use these colors consistently to maintain the Nothing aesthetic!**
