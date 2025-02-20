import re
import numpy as np
import pandas as pd

def remove_think_tags(text: str) -> str:
    """
    Strips out any <think>...</think> sections in the LLM output.
    """
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

def optimize_portfolio(returns_df):
    """
    Placeholder for portfolio optimization using mean-variance optimization.
    Returns equal weights for simplicity.
    """
    num_assets = returns_df.shape[1]
    weights = np.full(num_assets, 1/num_assets)
    return weights

def tune_prophet_parameters(df_train):
    """
    Placeholder for automated hyperparameter tuning using techniques like Optuna.
    Returns a dictionary of best parameters (for demonstration purposes).
    """
    best_params = {
        "changepoint_prior_scale": 0.05,
        "seasonality_prior_scale": 10.0
    }
    return best_params

def compute_rsi(series, period: int = 14):
    """
    Compute the Relative Strength Index (RSI) for a pandas Series.
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)  # default neutral value if insufficient data

def compute_buy_sell_signals(rsi_series):
    """
    Generate buy/sell signals based on RSI:
      - RSI < 30: Buy
      - RSI > 70: Sell
      - Otherwise: Hold
    """
    signals = rsi_series.apply(lambda x: "Buy" if x < 30 else ("Sell" if x > 70 else "Hold"))
    return signals