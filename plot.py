import pandas as pd
import plotly.graph_objs as go

def plot_interactive(df, symbol, forecast, signals=None):
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

    # Ensemble forecast trace
    fig.add_trace(
        go.Scatter(
            x=forecast['ds'], y=forecast['yhat'],
            mode='lines', name='Ensemble Forecast',
            line=dict(color='orange', dash='dash'),
            hovertemplate='Date: %{x}<br>Forecast: $%{y:.2f}<extra></extra>'
        )
    )
    
    # If signals provided, overlay buy/sell markers
    if signals is not None:
        # Merge signals with forecast dates for alignment (if applicable)
        for idx, row in signals.iterrows():
            if row['Signal'] == "Buy":
                marker_color = "green"
            elif row['Signal'] == "Sell":
                marker_color = "red"
            else:
                marker_color = "gray"
            fig.add_trace(
                go.Scatter(
                    x=[row['ds']], y=[df.loc[df['ds'] == row['ds'], 'y'].values[0]],
                    mode='markers',
                    marker=dict(color=marker_color, size=10, symbol='circle'),
                    name=f"{row['Signal']} Signal",
                    hovertemplate=f"Date: {row['ds']}<br>Signal: {row['Signal']}<extra></extra>"
                )
            )

    fig.update_layout(
        title=f"{symbol} Stock Price Prediction (Ensemble Forecast)",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        hovermode='x unified',
        template='plotly_white'
    )
    return fig