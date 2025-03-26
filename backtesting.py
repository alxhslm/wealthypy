import pandas as pd

from models import Asset
import yfinance as yf
import datetime as dt


def fetch_historic_growth(
    assets: dict[str, Asset], allocation: pd.DataFrame
) -> pd.DataFrame:
    total_growth = pd.Series(index=allocation.index, data=0.0)
    for name, asset in assets.items():
        assert asset.ticker, f"Missing ticker for asset {name}"
        ticker_data = yf.download(
            asset.ticker,
            start=allocation.index[0] - dt.timedelta(days=31),
            end=allocation.index[-1],
            interval="1mo",
        )
        monthly_returns = (
            ticker_data[("Close", asset.ticker)]
            .pct_change()
            .reindex(allocation.index, method="ffill")
        )
        total_growth += (1 + monthly_returns) * allocation[name].values
    return total_growth.to_frame()
