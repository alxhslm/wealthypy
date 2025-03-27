import numpy as np
import pandas as pd

from src.models import Asset
import yfinance as yf
import datetime as dt


def fetch_monthly_returns(
    ticker: str, start: dt.datetime | None, end: dt.datetime | None
) -> pd.Series:
    ticker_data = yf.download(ticker, start=start, end=end, interval="1mo")
    return ticker_data[("Close", ticker)].pct_change()


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
