from pathlib import Path
from typing import NewType

import pandas as pd
import yfinance as yf

from bspx.paths import RAW_DIR

Ticker = NewType("Ticker", str)


def _clean_tickers(tickers: list[Ticker] | list[str]) -> list[Ticker]:
    return [Ticker(t.strip().upper()) for t in tickers]


def _returns_path(ticker: Ticker) -> Path:
    return RAW_DIR / Path(f"{ticker}_returns.csv")


def _get_ticker_returns(ticker_returns_path: Path) -> pd.DataFrame:
    return pd.read_csv(
        ticker_returns_path, index_col=0, parse_dates=True, date_format="%Y-%m-%d"
    )


def _download_ticker_returns(
    ticker: Ticker, start: str, end: str
) -> pd.DataFrame | None:
    df = yf.download(ticker, start=start, end=end)

    if df is None or df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df


def _validate_and_clean_data(df: pd.DataFrame, ticker: Ticker) -> pd.DataFrame | None:
    required_cols = ["Open", "High", "Low", "Close"]

    df.columns = df.columns.str.title()

    available_cols = [c for c in required_cols if c in df.columns]

    if len(available_cols) < 4:
        missing = list(set(required_cols) - set(available_cols))
        print(f"Warning: {ticker} missing required columns: {missing}")

        return None

    keep_cols = list(available_cols)

    if "Volume" in df.columns:
        keep_cols.append("Volume")

    return df[keep_cols].copy()


def get_stock_data(
    tickers: list[Ticker] | list[str], start_date: str, end_date: str
) -> dict[Ticker, pd.DataFrame]:
    data: dict[Ticker, pd.DataFrame] = {}
    tickers = _clean_tickers(tickers)

    for ticker in tickers:
        df: pd.DataFrame | None = None
        ticker_path = _returns_path(ticker)

        if ticker_path.exists():
            try:
                df = _get_ticker_returns(ticker_path)
            except Exception as e:
                print(f"Warning: Couldn't load cached data for {ticker}: {e}")
                df = None

        if df is None:
            df = _download_ticker_returns(ticker, start_date, end_date)

            if df is None:
                print(f"Warning: Couldn't download data for {ticker}")
                continue

        clean_df = _validate_and_clean_data(df, ticker)

        if clean_df is None:
            print(f"Warning: Invalid data structure for {ticker}, skipping...")
            continue

        if not ticker_path.exists():
            clean_df.to_csv(ticker_path, date_format="%Y-%m-%d")

        data[ticker] = clean_df

    return data
