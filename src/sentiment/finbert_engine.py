"""
FinBERT-India Sentiment Analysis Engine
Uses Hugging Face's FinBERT-India model for Indian market-specific sentiment analysis.
"""

import pandas as pd
import yaml
from pathlib import Path
from typing import Union, List
import warnings
warnings.filterwarnings('ignore')


class FinBERTSentimentEngine:
    """
    Sentiment analysis using FinBERT-India model.
    Specifically trained on Indian financial news for better accuracy.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_name = self.config['sentiment']['model']
        self.max_tokens = self.config['sentiment']['max_tokens']
        self.ma_days = self.config['sentiment']['moving_average_days']
        
        self.processed_data_path = Path(self.config['paths']['processed_data'])
        self.processed_data_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize model (lazy loading)
        self.pipeline = None
    
    def load_model(self):
        """Load the FinBERT-India model from Hugging Face."""
        if self.pipeline is None:
            print(f"🤖 Loading {self.model_name}...")
            print("⏳ First-time load may take a few minutes to download model...")
            
            try:
                from transformers import pipeline
                import torch
                
                # Use PyTorch explicitly
                device = 0 if torch.cuda.is_available() else -1
                
                # Load sentiment analysis pipeline with FinBERT-India
                self.pipeline = pipeline(
                    "text-classification",
                    model=self.model_name,
                    device=device,  # Use GPU if available
                    framework="pt"  # Force PyTorch
                )
                
                print("✅ Model loaded successfully!")
                
            except Exception as e:
                print(f"❌ Error loading model: {str(e)}")
                print("💡 Tip: Make sure transformers and torch are installed:")
                print("   pip install transformers torch")
                raise
    
    def get_sentiment_score(self, text: str) -> float:
        """
        Get sentiment score for a single text.
        
        Args:
            text: News headline or article text
            
        Returns:
            Sentiment score: +1 (positive) to -1 (negative)
        """
        if not text or not isinstance(text, str):
            return 0.0
        
        try:
            # Truncate text to max tokens to fit BERT
            truncated_text = text[:self.max_tokens]
            
            # Get prediction
            result = self.pipeline(truncated_text)[0]
            label = result['label'].lower()
            score = result['score']
            
            # Map labels to numerical values
            # FinBERT-India typically uses: positive, negative, neutral
            if 'positive' in label:
                return score
            elif 'negative' in label:
                return -score
            else:  # neutral
                return 0.0
                
        except Exception as e:
            print(f"⚠️  Error analyzing text: {str(e)[:100]}")
            return 0.0
    
    def analyze_dataframe(self, df: pd.DataFrame, text_column: str = 'full_text') -> pd.DataFrame:
        """
        Analyze sentiment for all rows in a DataFrame.
        
        Args:
            df: DataFrame with text data
            text_column: Name of column containing text to analyze
            
        Returns:
            DataFrame with added sentiment_score column
        """
        # Ensure model is loaded
        self.load_model()
        
        if df.empty or text_column not in df.columns:
            print(f"⚠️  Column '{text_column}' not found in DataFrame!")
            return df
        
        print(f"\n🔬 Analyzing sentiment for {len(df)} articles...")
        print("=" * 60)
        
        # Apply sentiment analysis
        df['sentiment_score'] = df[text_column].apply(
            lambda x: self.get_sentiment_score(x) if pd.notna(x) else 0.0
        )
        
        # Add sentiment label for interpretation
        df['sentiment_label'] = df['sentiment_score'].apply(
            lambda x: 'Positive' if x > 0.1 else ('Negative' if x < -0.1 else 'Neutral')
        )
        
        # Statistics
        positive = (df['sentiment_score'] > 0.1).sum()
        negative = (df['sentiment_score'] < -0.1).sum()
        neutral = len(df) - positive - negative
        
        print(f"\n📊 Sentiment Distribution:")
        print(f"   ✅ Positive: {positive} ({positive/len(df)*100:.1f}%)")
        print(f"   ⚠️  Neutral:  {neutral} ({neutral/len(df)*100:.1f}%)")
        print(f"   ❌ Negative: {negative} ({negative/len(df)*100:.1f}%)")
        print(f"   📈 Average Score: {df['sentiment_score'].mean():.3f}")
        
        return df
    
    def aggregate_daily_sentiment(self, df: pd.DataFrame, date_column: str = 'date_only') -> pd.DataFrame:
        """
        Aggregate sentiment scores by day.
        
        Args:
            df: DataFrame with sentiment scores
            date_column: Name of date column
            
        Returns:
            DataFrame with daily aggregated sentiment
        """
        if df.empty or date_column not in df.columns:
            return pd.DataFrame()
        
        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
            df[date_column] = pd.to_datetime(df[date_column])
        
        # Aggregate by date
        daily_sentiment = df.groupby(date_column).agg({
            'sentiment_score': ['mean', 'std', 'count'],
            'sentiment_label': lambda x: x.mode()[0] if len(x) > 0 else 'Neutral'
        }).reset_index()
        
        # Flatten column names
        daily_sentiment.columns = ['date', 'sentiment_mean', 'sentiment_std', 'article_count', 'dominant_sentiment']
        
        # Add moving average
        daily_sentiment[f'sentiment_ma_{self.ma_days}'] = (
            daily_sentiment['sentiment_mean']
            .rolling(window=self.ma_days, min_periods=1)
            .mean()
        )
        
        print(f"\n📅 Aggregated sentiment for {len(daily_sentiment)} days")
        print(f"   Rolling average window: {self.ma_days} days")
        
        return daily_sentiment
    
    def process_news_file(self, news_file: str) -> pd.DataFrame:
        """
        Complete pipeline: Load news, analyze sentiment, aggregate daily.
        
        Args:
            news_file: Path to news CSV file
            
        Returns:
            DataFrame with daily sentiment scores
        """
        print(f"\n🚀 Processing news file: {news_file}")
        print("=" * 60)
        
        # Load news data
        news_df = pd.read_csv(news_file, parse_dates=['date'])
        print(f"📰 Loaded {len(news_df)} news articles")
        
        # Analyze sentiment
        news_df = self.analyze_dataframe(news_df)
        
        # Aggregate daily
        daily_sentiment = self.aggregate_daily_sentiment(news_df)
        
        # Save processed data
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed news with sentiment
        detailed_file = self.processed_data_path / f"news_sentiment_{timestamp}.csv"
        news_df.to_csv(detailed_file, index=False)
        print(f"\n💾 Saved detailed sentiment to: {detailed_file}")
        
        # Save daily aggregated sentiment
        daily_file = self.processed_data_path / f"daily_sentiment_{timestamp}.csv"
        daily_sentiment.to_csv(daily_file, index=False)
        print(f"💾 Saved daily sentiment to: {daily_file}")
        
        return daily_sentiment
    
    def get_latest_daily_sentiment(self) -> pd.DataFrame:
        """
        Load the most recent daily sentiment file.
        
        Returns:
            DataFrame with daily sentiment
        """
        files = list(self.processed_data_path.glob("daily_sentiment_*.csv"))
        
        if not files:
            print("⚠️  No daily sentiment files found. Run process_news_file() first.")
            return pd.DataFrame()
        
        import os
        latest_file = max(files, key=os.path.getctime)
        print(f"📂 Loading {latest_file.name}...")
        
        df = pd.read_csv(latest_file, parse_dates=['date'])
        return df


def main():
    """Main execution function."""
    import sys
    
    engine = FinBERTSentimentEngine()
    
    # Check if news file exists
    from pathlib import Path
    news_files = list(Path('data/raw').glob('news_raw_*.csv'))
    
    if not news_files:
        print("❌ No news data found!")
        print("💡 Run news_scraper.py first to collect news data.")
        sys.exit(1)
    
    # Use most recent news file
    import os
    latest_news = max(news_files, key=os.path.getctime)
    
    # Process news and generate sentiment
    daily_sentiment = engine.process_news_file(str(latest_news))
    
    print("\n" + "=" * 60)
    print("📋 Daily Sentiment Summary:")
    print("=" * 60)
    print(daily_sentiment.head(10))
    
    print("\n✅ Sentiment analysis complete!")


if __name__ == "__main__":
    main()
