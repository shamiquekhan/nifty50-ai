"""
Quick Launch Script for NIFTY50 AI Dashboard
Automatically checks dependencies and launches the dashboard.
"""

import subprocess
import sys
import os
from pathlib import Path

def check_requirements():
    """Check if required packages are installed."""
    required = ['streamlit', 'pandas', 'plotly']
    missing = []
    
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    return missing

def install_requirements():
    """Install missing requirements."""
    print("📦 Installing required packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    print("✅ Installation complete!")

def check_data():
    """Check if data files exist."""
    data_path = Path('data/raw')
    
    if not data_path.exists() or not list(data_path.glob('*.csv')):
        print("\n⚠️  WARNING: No market data found!")
        print("📥 Run data collection first:")
        print("   python src/data_collection/market_data.py")
        print("\n🔄 Continuing anyway (demo mode)...")
        return False
    
    return True

def launch_dashboard():
    """Launch the Streamlit dashboard."""
    print("\n" + "="*60)
    print("🚀 LAUNCHING NIFTY50 AI DASHBOARD")
    print("="*60)
    print("\n📱 Design: Nothing Brand Identity")
    print("🎨 Theme: Black/White/Red • Dot Matrix")
    print("📊 Features: LSTM + FinBERT + Kelly Criterion")
    print("\n🌐 Opening browser at: http://localhost:8501")
    print("⚡ Press Ctrl+C to stop the server")
    print("="*60 + "\n")
    
    subprocess.run([sys.executable, "-m", "streamlit", "run", "dashboard.py"])

def main():
    """Main execution."""
    banner = """
    ╔════════════════════════════════════════════════════════════╗
    ║                                                            ║
    ║   ███╗   ██╗██╗███████╗████████╗██╗   ██╗███████╗ ██████╗ ║
    ║   ████╗  ██║██║██╔════╝╚══██╔══╝╚██╗ ██╔╝██╔════╝██╔═████╗║
    ║   ██╔██╗ ██║██║█████╗     ██║    ╚████╔╝ ███████╗██║██╔██║║
    ║   ██║╚██╗██║██║██╔══╝     ██║     ╚██╔╝  ╚════██║████╔╝██║║
    ║   ██║ ╚████║██║██║        ██║      ██║   ███████║╚██████╔╝║
    ║   ╚═╝  ╚═══╝╚═╝╚═╝        ╚═╝      ╚═╝   ╚══════╝ ╚═════╝ ║
    ║                                                            ║
    ║              AI TRADING SYSTEM • NOTHING DESIGN            ║
    ║                                                            ║
    ╚════════════════════════════════════════════════════════════╝
    """
    print(banner)
    
    # Check dependencies
    print("\n🔍 Checking dependencies...")
    missing = check_requirements()
    
    if missing:
        print(f"❌ Missing packages: {', '.join(missing)}")
        response = input("\n📦 Install missing packages? (y/n): ")
        if response.lower() == 'y':
            install_requirements()
        else:
            print("⚠️  Exiting. Please install requirements manually.")
            return
    else:
        print("✅ All dependencies installed")
    
    # Check data
    check_data()
    
    # Launch
    launch_dashboard()

if __name__ == "__main__":
    main()
