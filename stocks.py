import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from prophet import Prophet
import plotly.graph_objs as go
import ollama

#######################################
# Utility: Remove <think>...</think>
#######################################
import re

def remove_think_tags(text: str) -> str:
    """
    Strips out any <think>...</think> sections in the LLM output.
    """
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


#######################################
# 1) Chat-based approach with Ollama
#######################################
def init_deepseek_interaction():
    """Initialize a conversation history for deepseek-r1 in st.session_state if not already present."""
    if "deepseek_interaction" not in st.session_state:
        st.session_state["deepseek_interaction"] = [
            {
                "role": "system",
                "content": (
                    "You are a helpful stock-analysis assistant. "
                    "Only produce a concise, final answer. Do not show your chain-of-thought."
                )
            }
        ]

def append_user_message(content: str):
    st.session_state["deepseek_interaction"].append({"role": "user", "content": content})

def append_assistant_message(content: str):
    st.session_state["deepseek_interaction"].append({"role": "assistant", "content": content})

def call_deepseek_r1_via_ollama() -> str:
    """
    Calls the Ollama-based 'deepseek-r1' model with the conversation in st.session_state["deepseek_interaction"].
    Returns the assistant's final message content.
    """
    try:
        response = ollama.chat(
            model="deepseek-r1",  # Adjust model name/tag if needed
            messages=st.session_state["deepseek_interaction"]
        )
        return response.message.content
    except Exception as e:
        return f"Error calling deepseek-r1: {str(e)}"

def get_deepseek_analysis(symbol: str, start_date, end_date) -> str:
    """
    Creates a short prompt about the stock performance, calls deepseek-r1, and returns its message (w/o chain-of-thought).
    """
    init_deepseek_interaction()

    user_prompt = (
        f"Analyze {symbol} stock performance between {start_date:%Y-%m-%d} and {end_date:%Y-%m-%d}. "
        "Explain key reasons for any price movements, referencing relevant corporate announcements, "
        "market sentiment, or other external factors. Provide a concise, fact-based summary. "
        "Do not reveal your chain-of-thought; just give the final answer."
    )
    append_user_message(user_prompt)

    # Actually call the model
    raw_reply = call_deepseek_r1_via_ollama()
    append_assistant_message(raw_reply)

    # Remove any <think> tags before displaying
    cleaned_reply = remove_think_tags(raw_reply)
    return cleaned_reply


#######################################
# 2) External Factor Placeholder
#######################################
def get_ollama_insights(symbol, start_date, end_date):
    """
    For demonstration, returns random external factors. Adapt if your LLM can provide numeric daily signals.
    """
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    factor_values = np.random.normal(loc=0.0, scale=1.0, size=len(dates))
    return pd.DataFrame({'ds': dates, 'external_factor': factor_values})


#######################################
# 3) Plotly Forecast Chart
#######################################
def plot_interactive(df, symbol, forecast):
    fig = go.Figure()

    # Historical data
    fig.add_trace(
        go.Scatter(
            x=df['ds'], y=df['y'],
            mode='lines', name='Historical',
            line=dict(color='blue'),
            hovertemplate='Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
        )
    )

    # Forecast line
    fig.add_trace(
        go.Scatter(
            x=forecast['ds'], y=forecast['yhat'],
            mode='lines', name='Forecast',
            line=dict(color='orange', dash='dash'),
            hovertemplate='Date: %{x}<br>Forecast: $%{y:.2f}<extra></extra>'
        )
    )

    # Confidence interval
    fig.add_trace(
        go.Scatter(
            x=pd.concat([forecast['ds'], forecast['ds'][::-1]]),
            y=pd.concat([forecast['yhat_upper'], forecast['yhat_lower'][::-1]]),
            fill='toself',
            fillcolor='rgba(255, 165, 0, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=True,
            name='Confidence Interval'
        )
    )

    fig.update_layout(
        title=f"{symbol} Stock Price Prediction",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        hovermode='x unified',
        template='plotly_white'
    )
    return fig


#######################################
# 4) Main Streamlit App
#######################################
st.title("Stock Predictor with deepseek-r1 (Using ollama) - No <think> Exposed")

symbols = st.text_input("Enter stock symbols (comma-separated):", "NVDA, TSLA")
days_back = st.number_input("Historical data range (days):", min_value=1, value=365)
days_to_predict = st.number_input("Forecast horizon (days):", min_value=1, value=30)

if st.button("Load Data and Predict"):
    try:
        symbols_list = [s.strip().upper() for s in symbols.split(",")]

        end_date = datetime.today()
        start_date = end_date - timedelta(days=days_back)

        # 1) Download data
        raw_data = yf.download(symbols_list, start=start_date, end=end_date)
        if raw_data.empty:
            st.warning("No data returned. Check symbols or date range.")
            st.stop()

        # Identify 'Adj Close' or 'Close'
        if isinstance(raw_data.columns, pd.MultiIndex):
            top_cols = raw_data.columns.levels[0]
            if 'Adj Close' in top_cols:
                stock_prices = raw_data['Adj Close']
            elif 'Close' in top_cols:
                stock_prices = raw_data['Close']
            else:
                st.error("Neither 'Adj Close' nor 'Close' found in data.")
                st.stop()
        else:
            if 'Adj Close' in raw_data.columns:
                stock_prices = raw_data['Adj Close']
            elif 'Close' in raw_data.columns:
                stock_prices = raw_data['Close']
            else:
                st.error("Neither 'Adj Close' nor 'Close' found in data.")
                st.stop()

        st.subheader("Latest Downloaded Data (Last 5 Rows)")
        st.dataframe(stock_prices.tail())

        # 2) Forecast per symbol
        for symbol in symbols_list:
            if symbol not in stock_prices.columns:
                st.warning(f"No data found for {symbol}.")
                continue

            st.markdown(f"## {symbol} Forecast")

            # Prepare prophet data
            df_sym = stock_prices[[symbol]].reset_index()
            df_sym.columns = ['ds', 'y']
            df_sym.dropna(inplace=True)
            df_sym['ds'] = pd.to_datetime(df_sym['ds'])

            if df_sym.empty:
                st.warning(f"No valid data for {symbol}. Skipping.")
                continue

            # External factor
            df_factors = get_ollama_insights(symbol, df_sym['ds'].min(), df_sym['ds'].max())
            df_merged = pd.merge(df_sym, df_factors, on='ds', how='left')

            model = Prophet(daily_seasonality=True, weekly_seasonality=True)
            model.add_regressor('external_factor')

            df_train = df_merged.dropna(subset=['external_factor', 'y'])
            if df_train.empty:
                st.warning(f"No overlapping data for {symbol}. Cannot forecast.")
                continue

            model.fit(df_train[['ds', 'y', 'external_factor']])

            future = model.make_future_dataframe(periods=days_to_predict, freq='D')
            last_factor = df_train['external_factor'].iloc[-1]
            future_factors = []
            for fdate in future['ds']:
                row = df_factors.loc[df_factors['ds'] == fdate]
                if not row.empty:
                    future_factors.append(row['external_factor'].values[0])
                else:
                    future_factors.append(last_factor)

            future['external_factor'] = future_factors
            forecast = model.predict(future)

            # Plot
            fig = plot_interactive(df_sym, symbol, forecast)
            st.plotly_chart(fig, use_container_width=True)

            st.write("Forecasted Data (Last 5 rows):")
            st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

            # 3) Show Deepseek note with a loading spinner
            with st.spinner("Loading deepseek analysis..."):
                analysis_note = get_deepseek_analysis(symbol, df_sym['ds'].min(), df_sym['ds'].max())

            st.info(f"**Deepseek's Note for {symbol}:**\n{analysis_note}")

    except Exception as e:
        st.error(f"An error occurred: {e}")