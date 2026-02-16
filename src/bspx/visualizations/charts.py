import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from bspx.indicators import add_bollinger_bands, add_moving_averages, add_rsi


def _add_candlestick_trace(
    fig: go.Figure, data: pd.DataFrame, row: int = 1, col: int = 1
) -> None:
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"],
            name="Price",
        ),
        row=row,
        col=col,
    )


def _add_moving_average_traces(
    fig: go.Figure, data: pd.DataFrame, row: int = 1, col: int = 1
) -> None:
    if "Short_MA" not in data.columns or "Long_MA" not in data.columns:
        return

    fig.add_trace(
        go.Scatter(x=data.index, y=data["Short_MA"], mode="lines", name="Short MA"),
        row=row,
        col=row,
    )

    fig.add_trace(
        go.Scatter(x=data.index, y=data["Long_MA"], mode="lines", name="Long MA"),
        row=row,
        col=row,
    )


def _add_bollinger_signals(
    fig: go.Figure, data: pd.DataFrame, row: int = 1, col: int = 1
) -> None:
    buy_signals = data[data["Signal"] == 1]
    sell_signals = data[data["Signal"] == -1]

    if not buy_signals.empty:
        fig.add_trace(
            go.Scatter(
                x=buy_signals.index,
                y=buy_signals["Close"],
                mode="markers",
                marker=dict(color="green", size=10),
                name="Buy",
            ),
            row=row,
            col=col,
        )

    if not sell_signals.empty:
        fig.add_trace(
            go.Scatter(
                x=sell_signals.index,
                y=sell_signals["Close"],
                mode="markers",
                marker=dict(color="red", size=10),
                name="Sell",
            ),
            row=row,
            col=col,
        )


def _add_rsi_subplot(fig: go.Figure, data: pd.DataFrame, row: int = 2) -> None:
    if "RSI" not in data.columns:
        return

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["RSI"],
            mode="lines",
            name="RSI",
            line=dict(color="purple"),
        ),
        row=row,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=[70] * len(data),
            mode="lines",
            line=dict(dash="dash", color="red", width=1),
            name="Overbought (70)",
            showlegend=False,
        ),
        row=row,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=[30] * len(data),
            mode="lines",
            line=dict(dash="dash", color="green", width=1),
            name="Oversold (30)",
            showlegend=False,
        ),
        row=row,
        col=1,
    )


def candlestick_chart(
    data: pd.DataFrame,
    ticker: str,
    plot_moving_average: bool = True,
    plot_bollinger_bands: bool = True,
    plot_rsi: bool = False,
    ma_short_window: int = 20,
    ma_long_window: int = 50,
    bollinger_window: int = 20,
    rsi_window: int = 14,
) -> go.Figure:
    fig: go.Figure
    df = data.copy()

    if plot_moving_average:
        df = add_moving_averages(df, ma_short_window, ma_long_window)

    if plot_bollinger_bands:
        df = add_bollinger_bands(df, bollinger_window)

    if plot_rsi:
        df = add_rsi(df, rsi_window)

    if plot_rsi:
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.7, 0.3],
            subplot_titles=(f"{ticker} Price", "RSI"),
        )
    else:
        fig = go.Figure()

    _add_candlestick_trace(fig, df, row=1, col=1)

    if plot_moving_average:
        _add_moving_average_traces(fig, df, row=1, col=1)

    if plot_bollinger_bands:
        _add_bollinger_signals(fig, df, row=1, col=1)

    if plot_rsi:
        _add_rsi_subplot(fig, df, row=2)

    fig.update_layout(
        title=f"{ticker} Stock Analysis",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        height=800 if plot_rsi else 600,
        showlegend=True,
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
    )

    if plot_rsi:
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="RSI", row=2, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=1)

    return fig


def candlestick_rsi_chart(data: pd.DataFrame, ticker: str) -> None:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.7, 0.3],
        subplot_titles=("Candlestick Chart", "RSI"),
    )

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"],
            name="Price",
        ),
        row=1,
        col=1,
    )

    # Moving averages
    fig.add_trace(
        go.Scatter(x=data.index, y=data["Short_MA"], mode="lines", name="Short MA"),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(x=data.index, y=data["Long_MA"], mode="lines", name="Long MA"),
        row=1,
        col=1,
    )

    # Strategies
    buy_signals = data[data["Signal"] == 1]
    sell_signals = data[data["Signal"] == -1]

    fig.add_trace(
        go.Scatter(
            x=buy_signals.index,
            y=buy_signals["Close"],
            mode="markers",
            marker=dict(color="green", size=10),
            name="Buy",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=sell_signals.index,
            y=sell_signals["Close"],
            mode="markers",
            marker=dict(color="red", size=10),
            name="Sell",
        ),
        row=1,
        col=1,
    )

    # RSI
    fig.add_trace(
        go.Scatter(x=data.index, y=data["RSI"], mode="lines", name="RSI"), row=2, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=[70] * len(data),
            mode="lines",
            line=dict(dash="dash", color="red"),
            name="Overbought",
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=[30] * len(data),
            mode="lines",
            line=dict(dash="dash", color="green"),
            name="Oversold",
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        title=f"{ticker} Stock Analysis",
        xaxis=dict(rangeslider=dict(visible=False)),
        xaxis2_title="Date",
        yaxis1_title="Price",
        yaxis2_title="RSI",
        height=800,
        showlegend=True,
    )

    fig.show()
