import pandas as pd
from pathlib import Path
from datetime import datetime

# Load news
news_df = pd.read_csv('data/raw/news_raw_20251214_204918.csv', parse_dates=['date'])
print(f"Loaded {len(news_df)} articles")

# Simple sentiment
POSITIVE = {'gain', 'gains', 'surge', 'rally', 'profit', 'growth', 'rise', 'up', 'high', 'positive', 'strong', 'bullish', 'boost', 'buy', 'record', 'best', 'top'}
NEGATIVE = {'loss', 'losses', 'fall', 'drop', 'decline', 'weak', 'bearish', 'down', 'low', 'negative', 'poor', 'worst', 'crash', 'plunge', 'sell', 'downgrade', 'concern', 'risk', 'fear'}

def score(text):
    words = set(text.lower().split())
    pos = len(words & POSITIVE)
    neg = len(words & NEGATIVE)
    total = pos + neg
    return (pos - neg) / max(total, 1) if total > 0 else 0.0

news_df['sentiment_score'] = news_df['full_text'].apply(score)

# Aggregate
daily = news_df.groupby('date_only').agg({'sentiment_score': ['mean', 'std', 'count']}).reset_index()
daily.columns = ['date', 'sentiment_mean', 'sentiment_std', 'article_count']
daily['sentiment_std'] = daily['sentiment_std'].fillna(0)
daily['sentiment_label'] = daily['sentiment_mean'].apply(lambda x: 'positive' if x > 0.1 else ('negative' if x < -0.1 else 'neutral'))

# Save
output = f"data/processed/daily_sentiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
daily.to_csv(output, index=False)
print(f"Saved to {output}")
print(daily)
