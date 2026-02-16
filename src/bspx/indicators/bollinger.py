import pandas as pd

DEFAULT_BOLLINGER_WINDOW = 20
DEFAULT_BOLLINGER_STD = 2.0


def calculate_bollinger_bands(
    data: pd.DataFrame,
    window: int = DEFAULT_BOLLINGER_WINDOW,
    num_std: float = DEFAULT_BOLLINGER_STD,
    price_col: str = "Close",
) -> tuple[pd.Series, pd.Series, pd.Series]:
    rolling = data[price_col].rolling(window=window)
    std = num_std * rolling.std()
    middle_band = rolling.mean()

    upper_band = middle_band + std
    lower_band = middle_band - std

    return lower_band, middle_band, upper_band  # pyright: ignore[reportReturnType]


def generate_bollinger_signals(
    data: pd.DataFrame, price_col: str = "Close"
) -> pd.Series:
    signals = pd.Series(0, index=data.index)

    buy_condition = (data[price_col] <= data["Lower_Band"]) & (
        data[price_col].shift(1) > data["Lower_Band"].shift(1)
    )

    sell_condition = (data[price_col] >= data["Lower_Band"]) & (
        data[price_col].shift(1) < data["Lower_Band"].shift(1)
    )

    signals[buy_condition] = 1
    signals[sell_condition] = -1

    return signals


def add_bollinger_bands(
    data: pd.DataFrame,
    window: int = DEFAULT_BOLLINGER_WINDOW,
    num_std: float = DEFAULT_BOLLINGER_STD,
    generate_signals: bool = True,
) -> pd.DataFrame:
    df = data.copy()

    lower_band, middle_band, upper_band = calculate_bollinger_bands(df, window, num_std)

    df["MA_20"] = middle_band
    df["Lower_Band"] = lower_band
    df["Upper_Band"] = upper_band

    if generate_signals:
        df["Signal"] = generate_bollinger_signals(df)

    return df.dropna()
