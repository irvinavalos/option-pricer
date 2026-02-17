import pandas as pd

DEFAULT_RSI_WINDOW = 14


def calculate_rsi(
    data: pd.DataFrame, window: int = DEFAULT_RSI_WINDOW, price_col: str = "Close"
) -> pd.Series:
    delta = data[price_col].diff()
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    loss = loss.mask(loss == 0)

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def add_rsi(data: pd.DataFrame, window: int = DEFAULT_RSI_WINDOW) -> pd.DataFrame:
    df = data.copy()
    df["RSI"] = calculate_rsi(df, window)

    return df.dropna()
