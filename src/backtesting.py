import numpy as np
import pandas as pd

from src.models import Asset
import yfinance as yf
import datetime as dt


def fetch_monthly_returns(
    ticker: str, start: dt.datetime | None, end: dt.datetime | None
) -> pd.Series:
    ticker_data: pd.DataFrame = yf.download(
        ticker, start=start, end=end, interval="1mo"
    )
    ticker_data.columns = ticker_data.columns.droplevel(1)
    if "Dividends" not in ticker_data.columns:
        ticker_data["Dividends"] = 0.0

    # Adjust for dividends by adding the 'Dividends' column to the 'Close' prices
    adjusted_close = ticker_data["Close"] + ticker_data["Dividends"]
    returns = adjusted_close.pct_change()
    dividend_yield = ticker_data["Dividends"].divide(
        ticker_data["Close"].shift(1), axis=0
    )
    return returns + dividend_yield


def fetch_historic_returns(
    assets: dict[str, Asset], allocation: pd.DataFrame
) -> pd.DataFrame:
    total_growth = pd.Series(index=allocation.index, data=0.0)
    for name, asset in assets.items():
        assert asset.ticker, f"Missing ticker for asset {name}"
        monthly_returns = fetch_monthly_returns(
            asset.ticker,
            start=allocation.index[0] - dt.timedelta(days=31),
            end=allocation.index[-1],
        )
        monthly_returns = monthly_returns.reindex(allocation.index, method="ffill")
        total_growth += (1 + monthly_returns) * allocation[name].values
    return total_growth.to_frame()


def estimate_returns_and_volatility(
    ticker: str, start: dt.datetime|None, end: dt.datetime|None
) -> tuple[pd.Series, pd.Series]:
    returns = fetch_monthly_returns(ticker, start, end)
    growth = returns.mean() * 12
    volatility = returns.std() * np.sqrt(12)
    return growth, volatility
