"""
Automated Data Update & Model Fine-tuning System
Runs every 4 hours to keep system up-to-date with latest market data
"""

import schedule
import time
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import logging
import yaml

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/auto_update.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class AutoUpdateSystem:
    """Automated system for data updates and model fine-tuning."""
    
    def __init__(self):
        """Initialize auto-update system."""
        self.update_interval = 4  # hours
        self.is_running = False
        
        # Create logs directory
        Path('logs').mkdir(exist_ok=True)
        
        logger.info("=" * 70)
        logger.info("AUTO-UPDATE SYSTEM INITIALIZED")
        logger.info(f"Update Interval: Every {self.update_interval} hours")
        logger.info("=" * 70)
    
    def run_script(self, script_path: str, description: str) -> bool:
        """
        Execute a Python script and log results.
        
        Args:
            script_path: Path to the script to run
            description: Description of what the script does
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"\n{'>'*3} {description}...")
            logger.info(f"    Executing: {script_path}")
            
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=600  # 10 minutes timeout
            )
            
            if result.returncode == 0:
                logger.info(f"    SUCCESS: {description}")
                if result.stdout:
                    for line in result.stdout.split('\n')[:5]:  # Show first 5 lines
                        if line.strip():
                            logger.info(f"    | {line}")
                return True
            else:
                logger.error(f"    FAILED: {description}")
                if result.stderr:
                    for line in result.stderr.split('\n')[:10]:
                        if line.strip():
                            logger.error(f"    | {line}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"    TIMEOUT: {description} took longer than 10 minutes")
            return False
        except Exception as e:
            logger.error(f"    ERROR: {description} - {str(e)}")
            return False
    
    def update_all(self):
        """Execute complete update cycle: data collection, model fine-tuning, predictions."""
        logger.info("\n" + "=" * 70)
        logger.info(f"UPDATE CYCLE STARTED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 70)
        
        success_count = 0
        total_steps = 6
        
        # Step 1: Collect market data
        if self.run_script(
            'src/data_collection/market_data.py',
            'STEP 1/6: Collecting latest market data'
        ):
            success_count += 1
        
        # Step 2: Scrape news articles
        if self.run_script(
            'src/data_collection/news_scraper.py',
            'STEP 2/6: Scraping latest news articles'
        ):
            success_count += 1
        
        # Step 3: Analyze sentiment with FinBERT
        if self.run_script(
            'src/sentiment/finbert_engine.py',
            'STEP 3/6: Analyzing sentiment with FinBERT-India'
        ):
            success_count += 1
        
        # Step 4: Fine-tune LSTM model with latest data (incremental)
        if self.run_script(
            'src/incremental_training.py',
            'STEP 4/6: Incremental fine-tuning of LSTM model'
        ):
            success_count += 1
        
        # Step 5: Generate fresh predictions
        if self.run_script(
            'src/predict.py',
            'STEP 5/6: Generating AI predictions'
        ):
            success_count += 1
        
        # Step 6: Run backtest with updated model
        if self.run_script(
            'src/backtesting.py',
            'STEP 6/6: Running backtest validation'
        ):
            success_count += 1
        
        # Summary
        logger.info("\n" + "=" * 70)
        logger.info(f"UPDATE CYCLE COMPLETED: {success_count}/{total_steps} steps successful")
        logger.info(f"Success Rate: {(success_count/total_steps)*100:.1f}%")
        logger.info(f"Next update in {self.update_interval} hours")
        logger.info("=" * 70 + "\n")
        
        return success_count == total_steps
    
    def start(self):
        """Start the automated update system."""
        logger.info("\n" + "=" * 70)
        logger.info("STARTING AUTO-UPDATE SYSTEM")
        logger.info("=" * 70)
        
        # Run immediately on start
        logger.info("\n>>> Running initial update...")
        self.update_all()
        
        # Schedule updates every 4 hours
        schedule.every(self.update_interval).hours.do(self.update_all)
        
        self.is_running = True
        logger.info(f"\n[OK] Auto-update scheduled every {self.update_interval} hours")
        logger.info("[OK] System is now running in background...")
        logger.info("[OK] Press Ctrl+C to stop\n")
        
        # Keep running
        try:
            while self.is_running:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            logger.info("\n\n" + "!" * 70)
            logger.info("AUTO-UPDATE SYSTEM STOPPED BY USER")
            logger.info("!" * 70)
            self.is_running = False


def main():
    """Main entry point."""
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║        NIFTY50 AUTO-UPDATE & FINE-TUNING SYSTEM              ║
    ║                                                               ║
    ║  → Updates market data every 4 hours                         ║
    ║  → Scrapes latest news & sentiment                           ║
    ║  → Fine-tunes LSTM model with fresh data                     ║
    ║  → Generates real-time predictions                           ║
    ║  → Validates with backtesting                                ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    system = AutoUpdateSystem()
    system.start()


if __name__ == "__main__":
    main()
