"""
News Scraper Module
Collects financial news from RSS feeds (MoneyControl, Economic Times, LiveMint).
Enhanced with unlimited free sources - Industry secret for proprietary datasets.
"""

import feedparser
import pandas as pd
import yaml
from datetime import datetime
from pathlib import Path
import time
from typing import List, Dict
import hashlib
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NewsScraper:
    """Scrapes financial news from RSS feeds - Unlimited & Free."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Enhanced RSS sources (Industry secret - unlimited access)
        self.rss_urls = {
            'moneycontrol_market': 'https://www.moneycontrol.com/rss/marketreports.xml',
            'moneycontrol_business': 'https://www.moneycontrol.com/rss/business.xml',
            'economic_times_stocks': 'https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms',
            'economic_times_economy': 'https://economictimes.indiatimes.com/news/economy/rssfeeds/1373380680.cms',
            'livemint_markets': 'https://www.livemint.com/rss/markets',
            'livemint_companies': 'https://www.livemint.com/rss/companies'
        }
        
        self.raw_data_path = Path(self.config['paths']['raw_data'])
        
        # Ensure data directory exists
        self.raw_data_path.mkdir(parents=True, exist_ok=True)
        
        # Track seen articles to avoid duplicates
        self.seen_hashes = set()
    
    def parse_date(self, date_string: str) -> datetime:
        """
        Parse date from RSS feed entry.
        
        Args:
            date_string: Date string from RSS feed
            
        Returns:
            datetime object or None if parsing fails
        """
        try:
            # Try common RSS date formats
            formats = [
                '%a, %d %b %Y %H:%M:%S %z',  # RFC 2822
                '%Y-%m-%dT%H:%M:%S%z',       # ISO 8601
                '%a, %d %b %Y %H:%M:%S',     # Without timezone
            ]
            
            for fmt in formats:
                try:
                    return datetime.strptime(date_string, fmt)
                except ValueError:
                    continue
            
            # If all formats fail, return None
            return None
            
        except Exception as e:
            print(f"⚠️  Date parsing error: {str(e)}")
            return None
    
    def fetch_feed(self, url: str) -> List[Dict]:
        """
        Fetch news from a single RSS feed.
        
        Args:
            url: RSS feed URL
            
        Returns:
            List of news items
        """
        news_items = []
        
        try:
            print(f"📡 Fetching feed: {url}")
            feed = feedparser.parse(url)
            
            if feed.bozo:  # Check for parsing errors
                print(f"⚠️  Feed parsing warning for {url}")
            
            for entry in feed.entries:
                # Extract date
                date_str = entry.get('published', entry.get('updated', ''))
                parsed_date = self.parse_date(date_str) if date_str else datetime.now()
                
                # Extract text content
                title = entry.get('title', '')
                summary = entry.get('summary', entry.get('description', ''))
                
                # Combine title and summary for more context
                full_text = f"{title}. {summary}"
                
                news_items.append({
                    'date': parsed_date,
                    'date_only': parsed_date.date() if parsed_date else None,
                    'title': title,
                    'summary': summary,
                    'full_text': full_text,
                    'source': url.split('/')[2],  # Extract domain
                    'link': entry.get('link', '')
                })
            
            print(f"✅ Fetched {len(news_items)} articles from {url.split('/')[2]}")
            
        except Exception as e:
            print(f"❌ Error fetching {url}: {str(e)}")
        
        return news_items
    
    def fetch_all_news(self, delay: float = 1.0) -> pd.DataFrame:
        """
        Fetch news from all configured RSS feeds.
        
        Args:
            delay: Delay between requests (in seconds) to be polite
            
        Returns:
            DataFrame with all news articles
        """
        all_news = []
        
        print("\n🗞️  Starting news collection...")
        print("=" * 60)
        
        for url in self.rss_urls:
            items = self.fetch_feed(url)
            all_news.extend(items)
            
            # Be polite - add delay between requests
            time.sleep(delay)
        
        if not all_news:
            print("⚠️  No news articles collected!")
            return pd.DataFrame()
        
        # Convert to DataFrame
        news_df = pd.DataFrame(all_news)
        
        # Remove duplicates (same title)
        news_df = news_df.drop_duplicates(subset=['title'], keep='first')
        
        # Sort by date
        news_df = news_df.sort_values('date', ascending=False)
        
        # Save to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.raw_data_path / f"news_raw_{timestamp}.csv"
        news_df.to_csv(filename, index=False)
        
        print("\n" + "=" * 60)
        print(f"💾 Saved {len(news_df)} unique articles to {filename}")
        print(f"📅 Date range: {news_df['date_only'].min()} to {news_df['date_only'].max()}")
        print(f"📰 Sources: {', '.join(news_df['source'].unique())}")
        
        return news_df
    
    def get_latest_news(self) -> pd.DataFrame:
        """
        Load the most recent news data file.
        
        Returns:
            DataFrame with news data
        """
        files = list(self.raw_data_path.glob("news_raw_*.csv"))
        
        if not files:
            print("⚠️  No news data files found. Run fetch_all_news() first.")
            return pd.DataFrame()
        
        # Get most recent file
        import os
        latest_file = max(files, key=os.path.getctime)
        print(f"📂 Loading {latest_file.name}...")
        
        df = pd.read_csv(latest_file, parse_dates=['date'])
        return df
    
    def filter_by_keywords(self, df: pd.DataFrame, keywords: List[str]) -> pd.DataFrame:
        """
        Filter news articles by keywords (useful for stock-specific news).
        
        Args:
            df: News DataFrame
            keywords: List of keywords to search for
            
        Returns:
            Filtered DataFrame
        """
        if df.empty:
            return df
        
        # Create regex pattern for keywords (case-insensitive)
        pattern = '|'.join(keywords)
        
        # Filter by title or summary
        mask = (df['title'].str.contains(pattern, case=False, na=False) |
                df['summary'].str.contains(pattern, case=False, na=False))
        
        filtered_df = df[mask].copy()
        
        print(f"🔍 Filtered to {len(filtered_df)} articles containing: {', '.join(keywords)}")
        
        return filtered_df
    
    def get_daily_article_count(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get count of articles per day.
        
        Args:
            df: News DataFrame
            
        Returns:
            DataFrame with daily counts
        """
        if df.empty or 'date_only' not in df.columns:
            return pd.DataFrame()
        
        daily_counts = df.groupby('date_only').size().reset_index(name='article_count')
        return daily_counts


def main():
    """Main execution function."""
    scraper = NewsScraper()
    
    # Fetch all news
    news_df = scraper.fetch_all_news()
    
    if not news_df.empty:
        print("\n" + "=" * 60)
        print("📋 News Summary:")
        print("=" * 60)
        print(news_df.info())
        
        print("\n📰 Sample Headlines:")
        print(news_df[['date_only', 'title', 'source']].head(10))
        
        # Show daily distribution
        daily_counts = scraper.get_daily_article_count(news_df)
        print("\n📊 Daily Article Distribution:")
        print(daily_counts.tail(10))
        
        print("\n✅ News collection complete!")


if __name__ == "__main__":
    main()
