import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def get_ollama_insights(symbol, start_date, end_date):
    """
    For demonstration, returns random external factors.
    """
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    factor_values = np.random.normal(loc=0.0, scale=1.0, size=len(dates))
    return pd.DataFrame({'ds': dates, 'external_factor': factor_values})

def get_news_sentiment(symbol: str) -> float:
    """
    Simulate fetching news headlines and computing a sentiment score.
    In a real application, integrate with a news API and use NLP sentiment analysis.
    """
    analyzer = SentimentIntensityAnalyzer()
    dummy_headlines = [
        f"{symbol} reports record profits in latest earnings.",
        f"{symbol} faces regulatory scrutiny amid market expansion.",
        f"Investors are optimistic about {symbol}'s new product launch.",
        f"{symbol} experiences a slight dip after competitor announcement.",
        f"Market reacts positively to {symbol}'s strategic partnership."
    ]
    scores = [analyzer.polarity_scores(headline)['compound'] for headline in dummy_headlines]
    avg_score = sum(scores) / len(scores)
    return round(avg_score, 2)