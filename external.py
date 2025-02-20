import pandas as pd
import numpy as np

def get_ollama_insights(symbol, start_date, end_date):
    """
    For demonstration, returns random external factors.
    Adapt if your LLM can provide numeric daily signals.
    """
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    factor_values = np.random.normal(loc=0.0, scale=1.0, size=len(dates))
    return pd.DataFrame({'ds': dates, 'external_factor': factor_values})