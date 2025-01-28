This Streamlit-based application predicts stock prices for a given set of symbols using historical data and Prophet’s forecasting model. It also leverages Deepseek-r1, an AI model, to analyze the reasons behind stock price movements over a given period. The AI model’s insights are displayed along with the forecasted stock price chart.

Features:
- Stock Prediction: Predict future stock prices based on historical data using the Prophet model.
- External Factors: Optionally incorporates external factors (e.g., market sentiment, economic data) to improve forecasts.
- Interactive Forecast Chart: Displays stock prices, forecasted values, and confidence intervals using Plotly.
- AI Insights: Uses Deepseek-r1 via Ollama for AI-generated analysis of stock performance.
- Loading Spinner: Displays a loading spinner while waiting for the AI model’s response.
