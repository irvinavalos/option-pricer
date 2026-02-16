from bspx.market_data import get_stock_data
from bspx.paths import RAW_DIR
from bspx.visualizations import candlestick_chart


def ensure_data_dirs() -> None:
    for dir in [RAW_DIR]:
        dir.mkdir(parents=True, exist_ok=True)


def main():
    ensure_data_dirs()

    tickers = ["AAPL", "GOOG", "MSFT"]
    start = "2023-01-01"
    end = "2024-11-01"

    stock_data = get_stock_data(tickers=tickers, start_date=start, end_date=end)

    for ticker, data in stock_data.items():
        fig = candlestick_chart(
            data=data,
            ticker=ticker,
            plot_rsi=True,
            plot_moving_average=True,
            plot_bollinger_bands=True,
        )
        fig.show()


if __name__ == "__main__":
    main()
