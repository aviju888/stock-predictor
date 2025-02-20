import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA

from deepseek import get_deepseek_analysis
from external import get_ollama_insights, get_news_sentiment
from plot import plot_interactive
from utils import optimize_portfolio, tune_prophet_parameters, compute_rsi, compute_buy_sell_signals

# --- Advanced: Real-Time Data Refresh ---
if st.sidebar.checkbox("Enable Real-Time Refresh", value=False):
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=60000, key="datarefresh")  # refresh every 60 seconds

st.title("Stock Predictor 2.0 – Buy/Sell & Short-Term Predictions")

# --- Sidebar: Advanced Settings & Customization ---
st.sidebar.header("Advanced Settings")
economic_shock = st.sidebar.slider("Economic Shock Factor (%)", min_value=-50, max_value=50, value=0, step=1,
                                   help="Simulate an economic event: positive values exaggerate the external factor, negative values dampen it.")
enable_hyperopt = st.sidebar.checkbox("Enable Hyperparameter Tuning", value=False,
                                      help="Automatically optimize model parameters to improve forecasting accuracy.")
selected_models = st.sidebar.multiselect("Select Forecasting Models", 
                                           options=["Prophet", "ARIMA", "LSTM"], 
                                           default=["Prophet", "ARIMA"],
                                           help="Choose which forecasting models to include in the ensemble.")
st.sidebar.markdown("Use the sidebar to adjust the economic shock factor, toggle hyperparameter tuning, and select forecasting models.")

# --- Tooltip / Help Section ---
with st.expander("What do these variables mean?"):
    st.markdown("""
    **Economic Shock Factor:**  
    Adjusts the external factors used in forecasting to simulate an economic event.  
    *Positive values* exaggerate the influence, *negative values* reduce it.

    **Hyperparameter Tuning:**  
    An automated process (stubbed in this demo) that optimizes model parameters for improved forecasting accuracy.

    **Forecasting Models:**  
    An ensemble of models (Prophet, ARIMA, and a placeholder LSTM) is used to capture both long-term trends and short-term movements.

    **Buy/Sell Signals:**  
    Technical indicators such as the Relative Strength Index (RSI) are computed.  
    - **RSI < 30:** Indicates an oversold condition (Buy signal).  
    - **RSI > 70:** Indicates an overbought condition (Sell signal).  
    - Otherwise, hold.
    """)

# --- User Inputs ---
symbols = st.text_input("Enter stock symbols (comma-separated):", "NVDA, TSLA")
days_back = st.number_input("Historical data range (days):", min_value=30, value=365)
days_to_predict = st.number_input("Forecast horizon (days):", min_value=1, value=30)

if st.button("Load Data and Predict"):
    try:
        symbols_list = [s.strip().upper() for s in symbols.split(",")]
        end_date = datetime.today()
        start_date = end_date - timedelta(days=days_back)
        raw_data = yf.download(symbols_list, start=start_date, end=end_date)
        if raw_data.empty:
            st.warning("No data returned. Check symbols or date range.")
            st.stop()

        # --- Process Downloaded Data ---
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

        ensemble_forecasts = {}

        # --- Forecast & Technical Analysis for Each Symbol ---
        for symbol in symbols_list:
            if symbol not in stock_prices.columns:
                st.warning(f"No data found for {symbol}.")
                continue

            st.markdown(f"## {symbol} Analysis")
            # Prepare the time series data
            df_sym = stock_prices[[symbol]].reset_index()
            df_sym.columns = ['ds', 'y']
            df_sym.dropna(inplace=True)
            df_sym['ds'] = pd.to_datetime(df_sym['ds'])
            if df_sym.empty:
                st.warning(f"No valid data for {symbol}. Skipping.")
                continue

            # --- Technical Indicators for Buy/Sell Signals ---
            df_sym['RSI'] = compute_rsi(df_sym['y'])
            df_sym['Signal'] = compute_buy_sell_signals(df_sym['RSI'])

            st.subheader("Technical Indicators (RSI & Buy/Sell Signals)")
            st.dataframe(df_sym[['ds', 'y', 'RSI', 'Signal']].tail())

            # --- Incorporate External Factors for Forecasting ---
            df_factors = get_ollama_insights(symbol, df_sym['ds'].min(), df_sym['ds'].max())
            df_merged = pd.merge(df_sym, df_factors, on='ds', how='left')
            shock_multiplier = 1 + (economic_shock / 100.0)

            # --- Forecast with Prophet ---
            forecast_prophet = None
            if "Prophet" in selected_models:
                model_prophet = Prophet(daily_seasonality=True, weekly_seasonality=True)
                model_prophet.add_regressor('external_factor')
                df_train = df_merged.dropna(subset=['external_factor', 'y'])
                if df_train.empty:
                    st.warning(f"No overlapping data for {symbol} with external factors. Skipping Prophet forecast.")
                else:
                    if enable_hyperopt:
                        best_params = tune_prophet_parameters(df_train)
                        st.write("Tuned parameters:", best_params)
                    model_prophet.fit(df_train[['ds', 'y', 'external_factor']])
                    future = model_prophet.make_future_dataframe(periods=days_to_predict, freq='D')
                    # Prepare external factor for future dates
                    last_factor = df_train['external_factor'].iloc[-1]
                    future_factors = []
                    for fdate in future['ds']:
                        row = df_factors.loc[df_factors['ds'] == fdate]
                        if not row.empty:
                            future_factors.append(row['external_factor'].values[0])
                        else:
                            future_factors.append(last_factor)
                    future['external_factor'] = np.array(future_factors) * shock_multiplier
                    forecast_prophet = model_prophet.predict(future)

            # --- Forecast with ARIMA ---
            forecast_arima = None
            if "ARIMA" in selected_models:
                df_arima = df_sym.set_index('ds')
                try:
                    model_arima = ARIMA(df_arima['y'], order=(5, 1, 0))
                    model_arima_fit = model_arima.fit()
                    arima_forecast = model_arima_fit.forecast(steps=days_to_predict)
                    forecast_arima = pd.DataFrame({
                        'ds': pd.date_range(start=df_sym['ds'].max() + timedelta(days=1), periods=days_to_predict, freq='D'),
                        'yhat': arima_forecast.values * shock_multiplier
                    })
                except Exception as e:
                    st.warning(f"ARIMA forecast failed for {symbol}: {e}")

            # --- Forecast with LSTM (Placeholder) ---
            forecast_lstm = None
            if "LSTM" in selected_models:
                last_value = df_sym['y'].iloc[-1]
                dates_future = pd.date_range(start=df_sym['ds'].max() + timedelta(days=1), periods=days_to_predict, freq='D')
                forecast_lstm = pd.DataFrame({
                    'ds': dates_future,
                    'yhat': np.full(shape=days_to_predict, fill_value=last_value) * shock_multiplier
                })

            # --- Ensemble Forecasting ---
            forecasts = []




            if forecast_prophet is not None:
                st.markdown("### Model Explainability")
                try:
                    import shap
                    # Remove the 'ds' column for SHAP to avoid datetime issues.
                    df_train_shap = df_train.drop(columns=['ds'])
                    
                    # Create a wrapper that adds back the 'ds' column (using the original values) before prediction.
                    def model_predict_wrapper(X):
                        # X: DataFrame with columns ['y', 'external_factor'] from SHAP.
                        # We add back the original 'ds' values (assumed fixed for explanation).
                        X_with_ds = pd.concat([df_train[['ds']].reset_index(drop=True), X.reset_index(drop=True)], axis=1)
                        return model_prophet.predict(X_with_ds)['yhat']
                    
                    explainer = shap.Explainer(model_predict_wrapper, df_train_shap)
                    shap_values = explainer(df_train_shap)
                    st.pyplot(shap.summary_plot(shap_values, df_train_shap, show=False))
                except Exception as e:
                    st.warning(f"SHAP explainability not available for {symbol}: {e}")

            if forecast_arima is not None:
                forecasts.append(forecast_arima.set_index('ds'))
            if forecast_lstm is not None:
                forecasts.append(forecast_lstm.set_index('ds'))
            
            if forecasts:
                ensemble_df = pd.concat(forecasts, axis=1)
                ensemble_df.columns = [f"model_{i}" for i in range(len(forecasts))]
                ensemble_df['yhat'] = ensemble_df.mean(axis=1)
                ensemble_df = ensemble_df.reset_index()
                ensemble_forecasts[symbol] = ensemble_df

                fig = plot_interactive(df_sym, symbol, ensemble_df, signals=df_sym[['ds', 'Signal']])
                st.plotly_chart(fig, use_container_width=True)

                st.write("Ensemble Forecasted Data (Last 5 rows):")
                st.dataframe(ensemble_df.tail())
            else:
                st.warning(f"No forecasts available for {symbol}.")

            # --- Explainability, Deepseek Analysis & Sentiment ---
            if forecast_prophet is not None:
                st.markdown("### Model Explainability")
                try:
                    import shap
                    explainer = shap.Explainer(model_prophet.predict, df_train[['ds', 'y', 'external_factor']])
                    shap_values = explainer(df_train[['ds', 'y', 'external_factor']])
                    st.pyplot(shap.summary_plot(shap_values, df_train[['ds', 'y', 'external_factor']], show=False))
                except Exception as e:
                    st.warning(f"SHAP explainability not available for {symbol}: {e}")

            with st.spinner("Loading deepseek analysis and sentiment..."):
                analysis_note = get_deepseek_analysis(symbol, df_sym['ds'].min(), df_sym['ds'].max())
                sentiment_score = get_news_sentiment(symbol)
            st.info(f"**Deepseek's Note for {symbol}:**\n{analysis_note}\n**News Sentiment Score:** {sentiment_score}")

        # --- Portfolio Analysis for Multiple Assets ---
        if len(symbols_list) > 1:
            st.markdown("## Portfolio Analysis")
            portfolio_df = stock_prices[symbols_list].pct_change().dropna()
            weights = optimize_portfolio(portfolio_df)
            st.write("Optimized Portfolio Weights:")
            st.dataframe(pd.DataFrame({"Symbol": symbols_list, "Weight": weights}))
            portfolio_return = (portfolio_df.mean() * 252).dot(weights)
            portfolio_volatility = np.sqrt(np.dot(weights, np.dot(portfolio_df.cov() * 252, weights)))
            st.write(f"Expected Annual Return: {portfolio_return:.2%}")
            st.write(f"Annual Volatility: {portfolio_volatility:.2%}")

    except Exception as e:
        st.error(f"An error occurred: {e}")