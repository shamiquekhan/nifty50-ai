#!/bin/bash
# Start Auto-Update System (Linux/Mac)
# Runs every 4 hours: Data Collection → Sentiment → Training → Predictions

echo "========================================"
echo "  NIFTY50 AI - Auto-Update System"
echo "========================================"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python not found!"
    exit 1
fi

# Create logs directory
mkdir -p logs

echo "Starting Auto-Update System..."
echo "  Update Interval: Every 4 hours"
echo "  Press Ctrl+C to stop"
echo ""

# Run the auto-update system
python3 src/auto_update.py

echo ""
echo "Auto-Update System stopped."
