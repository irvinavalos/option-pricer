import pandas as pd

DEFAULT_SHORT_WINDOW = 20
DEFAULT_LONG_WINDOW = 50


def calculate_moving_averages(
    data: pd.DataFrame,
    short_window: int = DEFAULT_SHORT_WINDOW,
    long_window: int = DEFAULT_LONG_WINDOW,
    price_col: str = "Close",
) -> tuple[pd.Series, pd.Series]:
    short_ma = data[price_col].rolling(window=short_window).mean()
    long_ma = data[price_col].rolling(window=long_window).mean()

    return short_ma, long_ma  # pyright: ignore[reportReturnType]


def generate_ma_crossover_signals(short_ma: pd.Series, long_ma: pd.Series) -> pd.Series:
    signals = pd.Series(0, index=short_ma.index)

    golden_cross = (short_ma > long_ma) & (short_ma.shift(1) <= long_ma.shift(1))
    death_cross = (short_ma < long_ma) & (short_ma.shift(1) >= long_ma.shift(1))

    signals[golden_cross] = 1
    signals[death_cross] = -1

    return signals


def add_moving_averages(
    data: pd.DataFrame,
    short_window: int = DEFAULT_SHORT_WINDOW,
    long_window: int = DEFAULT_LONG_WINDOW,
    generate_signals: bool = True,
) -> pd.DataFrame:
    if short_window >= long_window:
        raise ValueError("short_window must be less than long_window")

    df = data.copy()

    short_ma, long_ma = calculate_moving_averages(df, short_window, long_window)

    df["Short_MA"] = short_ma
    df["Long_MA"] = long_ma

    if generate_signals:
        df["Signal"] = generate_ma_crossover_signals(short_ma, long_ma)

    return df.dropna()
