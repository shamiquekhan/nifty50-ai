# NIFTY50 AI Dashboard Launcher (Windows PowerShell)
# Quick launch script with dependency checking

Write-Host "============================================================" -ForegroundColor Red
Write-Host "   NIFTY50 AI • NOTHING DESIGN SYSTEM" -ForegroundColor White
Write-Host "============================================================" -ForegroundColor Red
Write-Host ""

# Check Python
Write-Host "Checking Python installation..." -ForegroundColor Cyan
$pythonVersion = python --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Python not found!" -ForegroundColor Red
    Write-Host "Please install Python 3.8+ from https://www.python.org/" -ForegroundColor Yellow
    exit 1
}
Write-Host "  Found: $pythonVersion" -ForegroundColor Green

# Check if in correct directory
if (-not (Test-Path "dashboard.py")) {
    Write-Host "ERROR: dashboard.py not found!" -ForegroundColor Red
    Write-Host "Please run this script from the Nifty50Qualtml directory" -ForegroundColor Yellow
    exit 1
}

# Check virtual environment
Write-Host "`nChecking virtual environment..." -ForegroundColor Cyan
if (-not (Test-Path "venv")) {
    Write-Host "  Creating virtual environment..." -ForegroundColor Yellow
    python -m venv venv
    Write-Host "  Created venv/" -ForegroundColor Green
}

# Activate virtual environment
Write-Host "  Activating virtual environment..." -ForegroundColor Yellow
& .\venv\Scripts\Activate.ps1

# Install dependencies
Write-Host "`nInstalling dependencies..." -ForegroundColor Cyan
pip install -q -r requirements.txt
if ($LASTEXITCODE -eq 0) {
    Write-Host "  All packages installed" -ForegroundColor Green
} else {
    Write-Host "  Warning: Some packages may have failed" -ForegroundColor Yellow
}

# Check for data
Write-Host "`nChecking data files..." -ForegroundColor Cyan
if (Test-Path "data\raw\*.csv") {
    Write-Host "  Market data found" -ForegroundColor Green
} else {
    Write-Host "  WARNING: No market data found!" -ForegroundColor Yellow
    Write-Host "  Run: python src/data_collection/market_data.py" -ForegroundColor Yellow
    Write-Host "  Continuing in demo mode..." -ForegroundColor Gray
}

# Launch dashboard
Write-Host "`n============================================================" -ForegroundColor Red
Write-Host " LAUNCHING DASHBOARD" -ForegroundColor White
Write-Host "============================================================" -ForegroundColor Red
Write-Host ""
Write-Host "  Design: Nothing Brand Identity" -ForegroundColor White
Write-Host "  Theme: Black/White/Red + Dot Matrix" -ForegroundColor White
Write-Host "  Features: LSTM + FinBERT + Kelly Criterion" -ForegroundColor White
Write-Host ""
Write-Host "  Opening: http://localhost:8501" -ForegroundColor Green
Write-Host "  Press Ctrl+C to stop the server" -ForegroundColor Gray
Write-Host ""
Write-Host "============================================================" -ForegroundColor Red
Write-Host ""

# Run Streamlit
streamlit run dashboard.py

# Cleanup
Write-Host "`nShutting down..." -ForegroundColor Yellow
deactivate
