import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def calculate_log_returns(prices: pd.Series) -> pd.Series:
    return (prices / prices.shift(1)).apply(func=np.log)


def calculate_rolling_vol(
    prices: pd.Series, window: int = 20, annualize: bool = True
) -> pd.Series:
    log_returns = calculate_log_returns(prices)
    vol = log_returns.rolling(window=window).std()

    if annualize:
        vol *= np.sqrt(TRADING_DAYS_PER_YEAR)

    return vol


def calculate_ewma_vol(
    prices: pd.Series, span: float = 20, annualize: bool = True
) -> pd.Series:
    log_returns = calculate_log_returns(prices)
    vol = log_returns.ewm(span=span).std()

    if annualize:
        vol *= np.sqrt(TRADING_DAYS_PER_YEAR)

    return vol


def add_volatility_columns(
    data: pd.DataFrame,
    price_col: str = "Close",
    window: int = 20,
    ewma_span: float = 20,
) -> pd.DataFrame:
    if data is None or data.empty:
        raise ValueError("Input data cannot be empty")

    df = data.copy()

    df["Log_returns"] = calculate_log_returns(df[price_col])
    df["Rolling_vol"] = calculate_rolling_vol(df[price_col], window=window)
    df["Ewma_vol"] = calculate_ewma_vol(df[price_col], span=ewma_span)

    return df.dropna()


def realized_volatility(
    prices: pd.Series, period: int = 30, annualize: bool = True
) -> float:
    log_returns = calculate_log_returns(prices).dropna()
    recent_returns = log_returns.tail(period)
    vol = recent_returns.std()

    if annualize:
        vol *= np.sqrt(TRADING_DAYS_PER_YEAR)

    return vol
