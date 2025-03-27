import numpy as np
import pandas as pd


def run_simulation(
    starting_amount: float,
    monthly_contributions: pd.Series,
    growth: pd.DataFrame,
    inflation: float,
) -> tuple[pd.DataFrame, pd.Series]:
    # Compute portfolio values
    portfolio = pd.DataFrame(index=growth.index, columns=growth.columns)
    portfolio.iloc[0, :] = starting_amount
    for i in range(1, growth.shape[0]):
        portfolio.iloc[i, :] = (
            (portfolio.iloc[i - 1, :]) * growth.iloc[i, :]
            + monthly_contributions.iloc[i - 1]
        )

    # Adjust for inflation
    inflation_adjustment = (1 + inflation / 12) ** np.arange(growth.shape[0])
    years = len(growth) / 12
    aer = growth.divide(1 + inflation / 12).prod().pow(1 / years) - 1
    return portfolio.divide(inflation_adjustment, axis=0), aer
