import pandas as pd
import plotly.graph_objs as go

def plot_interactive(df, symbol, forecast):
    fig = go.Figure()

    # Historical data trace
    fig.add_trace(
        go.Scatter(
            x=df['ds'], y=df['y'],
            mode='lines', name='Historical',
            line=dict(color='blue'),
            hovertemplate='Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
        )
    )

    # Forecast trace
    fig.add_trace(
        go.Scatter(
            x=forecast['ds'], y=forecast['yhat'],
            mode='lines', name='Forecast',
            line=dict(color='orange', dash='dash'),
            hovertemplate='Date: %{x}<br>Forecast: $%{y:.2f}<extra></extra>'
        )
    )

    # Confidence interval trace
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