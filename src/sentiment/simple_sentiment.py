"""
Simple Sentiment Analyzer (Fallback)
Uses lexicon-based approach for quick sentiment analysis
"""

import pandas as pd
import yaml
from pathlib import Path
from datetime import datetime

# Simple sentiment lexicon for financial news
POSITIVE_WORDS = {
    'gain', 'gains', 'surge', 'surges', 'rally', 'rallies', 'profit', 'profits',
    'growth', 'rise', 'rises', 'jump', 'jumps', 'soar', 'soars', 'up', 'high',
    'positive', 'strong', 'bullish', 'recovery', 'advance', 'advances', 'boost',
    'outperform', 'upgrade', 'buy', 'record', 'best', 'top', 'leading', 'winning'
}

NEGATIVE_WORDS = {
    'loss', 'losses', 'fall', 'falls', 'drop', 'drops', 'decline', 'declines',
    'weak', 'bearish', 'down', 'low', 'negative', 'poor', 'worst', 'crash',
    'plunge', 'plunges', 'tumble', 'tumbles', 'sell', 'selling', 'downgrade',
    'underperform', 'concern', 'concerns', 'risk', 'risks', 'fear', 'fears'
}


def analyze_text(text):
    """Simple lexicon-based sentiment analysis."""
    text_lower = text.lower()
    words = set(text_lower.split())
    
    pos_count = len(words & POSITIVE_WORDS)
    neg_count = len(words & NEGATIVE_WORDS)
    
    total = pos_count + neg_count
    if total == 0:
        return 0.0  # Neutral
    
    # Calculate sentiment score (-1 to 1)
    score = (pos_count - neg_count) / max(total, 1)
    return score


def process_news_file(news_file):
    """Process news file and generate sentiment scores."""
    print(f"\n🚀 Processing news file: {news_file}")
    print("="*60)
    
    # Load news
    news_df = pd.read_csv(news_file, parse_dates=['date'])
    print(f"📰 Loaded {len(news_df)} news articles")
    
    # Analyze sentiment for each article
    print("🤖 Analyzing sentiment...")
    news_df['sentiment_score'] = news_df['full_text'].apply(analyze_text)
    
    # Aggregate by date
    daily_sentiment = news_df.groupby('date_only').agg({
        'sentiment_score': ['mean', 'std', 'count']
    }).reset_index()
    
    daily_sentiment.columns = ['date', 'sentiment_mean', 'sentiment_std', 'article_count']
    
    # Fill NaN std with 0
    daily_sentiment['sentiment_std'] = daily_sentiment['sentiment_std'].fillna(0)
    
    # Add sentiment label
    daily_sentiment['sentiment_label'] = daily_sentiment['sentiment_mean'].apply(
        lambda x: 'positive' if x > 0.1 else ('negative' if x < -0.1 else 'neutral')
    )
    
    print(f"✅ Analyzed {len(news_df)} articles")
    print(f"📅 Generated sentiment for {len(daily_sentiment)} days")
    
    return daily_sentiment


def main():
    """Main function."""
    # Load config
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Find latest news file
    news_files = list(Path('data/raw').glob('news_raw_*.csv'))
    
    if not news_files:
        print("❌ No news files found. Run news_scraper.py first.")
        return
    
    latest_news = max(news_files, key=lambda x: x.stat().st_ctime)
    
    # Process news
    daily_sentiment = process_news_file(latest_news)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"data/processed/daily_sentiment_{timestamp}.csv"
    daily_sentiment.to_csv(output_file, index=False)
    
    print("\n" + "="*60)
    print(f"💾 Saved daily sentiment to {output_file}")
    print(f"📅 Date range: {daily_sentiment['date'].min()} to {daily_sentiment['date'].max()}")
    
    print("\n" + "="*60)
    print("📋 Sentiment Summary:")
    print("="*60)
    print(daily_sentiment.info())
    
    print("\n📊 Sample Data:")
    print(daily_sentiment.head(10).to_string(index=False))
    
    print("\n✅ Sentiment analysis complete!")


if __name__ == "__main__":
    main()
