"""
RSS News Harvester - Unlimited Free News Data
Sources: MoneyControl, Economic Times, LiveMint
Updates every 15 minutes - builds proprietary news dataset
"""

import feedparser
import pandas as pd
from datetime import datetime
import time
from pathlib import Path
import logging
from typing import List, Dict
import hashlib

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/news_harvester.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Unlimited RSS Sources (Industry Secret)
RSS_URLS = {
    'moneycontrol_market': 'https://www.moneycontrol.com/rss/marketreports.xml',
    'moneycontrol_business': 'https://www.moneycontrol.com/rss/business.xml',
    'economic_times_stocks': 'https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms',
    'economic_times_news': 'https://economictimes.indiatimes.com/news/economy/rssfeeds/1373380680.cms',
    'livemint_markets': 'https://www.livemint.com/rss/markets',
    'livemint_companies': 'https://www.livemint.com/rss/companies'
}

# NIFTY50 Tickers for filtering
NIFTY50_KEYWORDS = [
    'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK', 'HINDUNILVR', 'ITC', 'SBIN',
    'BHARTIARTL', 'BAJFINANCE', 'KOTAKBANK', 'LT', 'AXISBANK', 'ASIANPAINT', 'MARUTI',
    'TITAN', 'SUNPHARMA', 'ULTRACEMCO', 'NESTLEIND', 'WIPRO', 'POWERGRID', 'NTPC',
    'TECHM', 'HCLTECH', 'ONGC', 'M&M', 'TATAMOTORS', 'TATASTEEL', 'BAJAJFINSV',
    'ADANIPORTS', 'COALINDIA', 'JSWSTEEL', 'GRASIM', 'HINDALCO', 'INDUSINDBK',
    'CIPLA', 'DRREDDY', 'DIVISLAB', 'EICHERMOT', 'BRITANNIA', 'HEROMOTOCO',
    'BAJAJ-AUTO', 'SHREECEM', 'APOLLOHOSP', 'BPCL', 'UPL', 'TATACONSUM',
    'ADANIENT', 'SBILIFE', 'LTIM'
]


class NewsHarvester:
    """Harvest unlimited news from RSS feeds."""
    
    def __init__(self, output_dir: str = 'data/news'):
        """Initialize news harvester."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.db_file = self.output_dir / 'news_database.csv'
        self.seen_hashes = self._load_seen_hashes()
        
    def _load_seen_hashes(self) -> set:
        """Load previously seen article hashes to avoid duplicates."""
        if self.db_file.exists():
            try:
                df = pd.read_csv(self.db_file)
                if 'hash' in df.columns:
                    return set(df['hash'].tolist())
            except Exception as e:
                logger.warning(f"Could not load seen hashes: {e}")
        return set()
    
    def _hash_article(self, title: str, link: str) -> str:
        """Create unique hash for article."""
        content = f"{title}{link}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _extract_tickers(self, text: str) -> List[str]:
        """Extract NIFTY50 tickers mentioned in text."""
        text_upper = text.upper()
        mentioned = []
        for ticker in NIFTY50_KEYWORDS:
            # Check for ticker name or common variations
            if ticker in text_upper:
                mentioned.append(ticker)
            # Also check for company name variations
            if ticker == 'HDFCBANK' and 'HDFC BANK' in text_upper:
                mentioned.append(ticker)
            elif ticker == 'ICICIBANK' and 'ICICI BANK' in text_upper:
                mentioned.append(ticker)
        return list(set(mentioned))
    
    def harvest_single_feed(self, source_name: str, url: str) -> List[Dict]:
        """Harvest news from a single RSS feed."""
        articles = []
        
        try:
            logger.info(f"Fetching from {source_name}...")
            feed = feedparser.parse(url)
            
            for entry in feed.entries:
                # Extract article data
                title = entry.get('title', '')
                link = entry.get('link', '')
                summary = entry.get('summary', entry.get('description', ''))
                published = entry.get('published', datetime.now().isoformat())
                
                # Create hash to check for duplicates
                article_hash = self._hash_article(title, link)
                
                # Skip if already seen
                if article_hash in self.seen_hashes:
                    continue
                
                # Extract mentioned tickers
                full_text = f"{title} {summary}"
                mentioned_tickers = self._extract_tickers(full_text)
                
                articles.append({
                    'hash': article_hash,
                    'source': source_name,
                    'title': title,
                    'summary': summary[:500],  # Limit summary length
                    'link': link,
                    'published': published,
                    'mentioned_tickers': ','.join(mentioned_tickers) if mentioned_tickers else 'GENERAL',
                    'fetched_at': datetime.now().isoformat()
                })
                
                # Mark as seen
                self.seen_hashes.add(article_hash)
            
            logger.info(f"✓ {source_name}: {len(articles)} new articles")
            
        except Exception as e:
            logger.error(f"✗ Error fetching {source_name}: {e}")
        
        return articles
    
    def harvest_all(self) -> pd.DataFrame:
        """Harvest news from all RSS sources."""
        all_articles = []
        
        logger.info("=" * 60)
        logger.info("STARTING NEWS HARVEST CYCLE")
        logger.info("=" * 60)
        
        for source_name, url in RSS_URLS.items():
            articles = self.harvest_single_feed(source_name, url)
            all_articles.extend(articles)
            time.sleep(1)  # Be polite to servers
        
        # Convert to DataFrame
        if all_articles:
            df = pd.DataFrame(all_articles)
            
            # Append to database (or create if doesn't exist)
            if self.db_file.exists():
                df.to_csv(self.db_file, mode='a', header=False, index=False)
            else:
                df.to_csv(self.db_file, index=False)
            
            logger.info("=" * 60)
            logger.info(f"✓ HARVEST COMPLETE: {len(df)} NEW ARTICLES SAVED")
            logger.info(f"✓ Total database size: {len(self.seen_hashes)} unique articles")
            logger.info("=" * 60)
            
            return df
        else:
            logger.info("No new articles found in this cycle")
            return pd.DataFrame()
    
    def get_ticker_news(self, ticker: str, days: int = 7) -> pd.DataFrame:
        """Get news for a specific ticker from last N days."""
        if not self.db_file.exists():
            return pd.DataFrame()
        
        df = pd.read_csv(self.db_file)
        df['fetched_at'] = pd.to_datetime(df['fetched_at'])
        
        # Filter by date
        cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=days)
        df = df[df['fetched_at'] >= cutoff_date]
        
        # Filter by ticker
        df = df[df['mentioned_tickers'].str.contains(ticker, na=False)]
        
        return df.sort_values('fetched_at', ascending=False)
    
    def run_continuous(self, interval_minutes: int = 15):
        """Run harvester continuously (for deployment)."""
        logger.info(f"Starting continuous harvester (interval: {interval_minutes} mins)")
        
        while True:
            try:
                self.harvest_all()
                logger.info(f"Next harvest in {interval_minutes} minutes...")
                time.sleep(interval_minutes * 60)
            except KeyboardInterrupt:
                logger.info("Harvester stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in harvest cycle: {e}")
                logger.info("Retrying in 5 minutes...")
                time.sleep(300)


def main():
    """Main entry point for news harvester."""
    import argparse
    
    parser = argparse.ArgumentParser(description='RSS News Harvester')
    parser.add_argument('--continuous', action='store_true', help='Run continuously')
    parser.add_argument('--interval', type=int, default=15, help='Harvest interval in minutes')
    parser.add_argument('--ticker', type=str, help='Get news for specific ticker')
    args = parser.parse_args()
    
    harvester = NewsHarvester()
    
    if args.ticker:
        # Get news for specific ticker
        news_df = harvester.get_ticker_news(args.ticker)
        print(f"\n{len(news_df)} articles found for {args.ticker}:\n")
        print(news_df[['source', 'title', 'fetched_at']].to_string())
    elif args.continuous:
        # Run continuously
        harvester.run_continuous(interval_minutes=args.interval)
    else:
        # Single harvest
        harvester.harvest_all()


if __name__ == '__main__':
    main()
