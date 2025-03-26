import numpy as np
import pandas as pd

from models import Asset


def sample_growth(
    assets: dict[str, Asset],
    allocation: pd.DataFrame,
    num_simulations: int,
    seed: int = 42,
) -> pd.DataFrame:
    np.random.seed(seed)
    total_growth = pd.DataFrame(
        index=allocation.index, columns=range(num_simulations), data=0.0
    )
    for name, asset in assets.items():
        monthly_returns = np.random.normal(
            asset.returns / 12,
            asset.volatility / np.sqrt(12),
            (allocation.shape[0], num_simulations),
        )
        growth = (1 + monthly_returns) * allocation[name].values[:, np.newaxis]
        total_growth += growth
    return total_growth


def compute_quantiles(value: pd.Series) -> pd.Series:
    quantiles = value.quantile([0.1, 0.5, 0.9])
    quantiles.index = (
        quantiles.index.to_series()
        .apply(lambda x: f"{x * 100:.0f}%")
        .rename("Quantile")
    )
    quantiles.loc["Mean"] = value.mean()
    return quantiles[["10%", "50%", "Mean", "90%"]]
