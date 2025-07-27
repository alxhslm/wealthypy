from dataclasses import asdict
import streamlit as st
import pandas as pd
import datetime as dt
import numpy as np
from src.backtesting import (
    estimate_returns_and_volatility,
    fetch_historic_returns,
    fetch_monthly_returns,
)
from src.models import Asset
from src.monte_carlo import sample_returns, compute_quantiles
from src.plotting import plot_hist_returns, plot_returns, plot_scatter
from src.simulate import run_simulation

pd.options.plotting.backend = "plotly"


@st.cache_data
def monte_carlo(
    starting_amount: float,
    monthly_contributions: pd.Series,
    assets: dict[str, Asset],
    allocation: pd.DataFrame,
    inflation: float,
    fees: float,
    start_date: dt.date,
    end_date: dt.date,
    cov_matrix: pd.DataFrame | None = None,
    num_simulations: int = 1,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    index = pd.date_range(start_date, end_date, freq="ME").date
    return run_simulation(
        starting_amount=starting_amount,
        monthly_contributions=monthly_contributions.reindex(index, method="ffill").fillna(0),
        growth=sample_returns(
            assets=assets,
            allocation=allocation.reindex(index, method="ffill"),
            num_simulations=num_simulations,
            seed=seed,
            cov_matrix=cov_matrix,  # Pass the covariance matrix
        )
        - fees / 12,
        inflation=inflation,
    )


@st.cache_data
def backtest(
    starting_amount: float,
    monthly_contributions: pd.Series,
    assets: dict[str, Asset],
    allocation: pd.DataFrame,
    inflation: float,
    fees: float,
    start_date: dt.date,
    end_date: dt.date,
) -> tuple[pd.DataFrame, pd.Series]:
    index = pd.date_range(start_date, end_date, freq="ME").date

    return run_simulation(
        starting_amount=starting_amount,
        monthly_contributions=monthly_contributions.reindex(index, method="ffill").fillna(0),
        growth=fetch_historic_returns(
            assets=assets, allocation=allocation.reindex(index, method="ffill").bfill()
        )
        - fees / 12,
        inflation=inflation,
    )


# Streamlit UI
st.set_page_config(layout="wide", page_icon=":chart:", page_title="WealthyPy")
with st.sidebar:
    st.title(":chart: WealthyPy")
    st.caption("Portfolio Growth Simulator")

if "assets" not in st.session_state:
    st.session_state["assets"] = {
        "Equities": Asset(returns=5.0 / 100, volatility=25 / 100, ticker="VWRP.L"),
        "Bonds": Asset(returns=2.0 / 100, volatility=0.0 / 100, ticker="VAGS.L"),
    }

if "allocation" not in st.session_state:
    allocation = pd.DataFrame(
        {
            k: [1.0 / len(st.session_state["assets"])]
            for k in st.session_state["assets"]
        },
        index=[dt.date.today() - dt.timedelta(days=3 * 365.25)],
    )
    allocation.index.name = "Date"
    st.session_state["allocation"] = allocation


with st.sidebar:
    container = st.container()
    end_date = st.date_input(
        "End date", max_value=dt.date.today(), value=dt.date.today()
    )
    with container:
        start_date = st.date_input(
            "Start date",
            value=dt.date.today() - dt.timedelta(days=3 * 365.25),
            max_value=end_date,
        )
    page = st.radio(
        "Select mode",
        options=["Configure funds", "Configure portfolio", "Run simulation"],
        index=0,
        horizontal=True,
    )

if "starting_amount" not in st.session_state:
    st.session_state["starting_amount"] = 10000.0
if "contributions" not in st.session_state:
    st.session_state["contributions"] = pd.DataFrame(
        {
            "Date": [start_date],
            "Monthly Contribution": [1000.0],
        }
    ).set_index("Date")

if "simulation_dfs" not in st.session_state:
    st.session_state["simulation_dfs"] = pd.DataFrame(dtype=float)
if "cagr" not in st.session_state:
    st.session_state["cagr"] = pd.Series(dtype=float)

if page == "Configure funds":
    st.header("Funds")
    starting_amount = st.number_input(
        "Starting amount (£)",
        value=st.session_state["starting_amount"],
        step=1000.0,
        format="%.1f",
    )
    contributions = st.data_editor(
        st.session_state["contributions"].reset_index(),
        num_rows="dynamic",
        hide_index=True,
        column_config={
            "Monthly Contribution": st.column_config.NumberColumn(format="£%.2f")
        },
    ).set_index("Date")
    st.session_state["starting_amount"] = starting_amount
    st.session_state["contributions"] = contributions
elif page == "Configure portfolio":
    st.header("Portfolio")
    with st.form("portfolio_form"):
        cols = st.columns(2)
        with cols[0]:
            start_est = st.date_input(
                "Start Date", value=dt.date(1900, 1, 1), max_value=dt.date.today()
            )
        with cols[1]:
            end_est = st.date_input(
                "End Date", value=dt.date.today(), max_value=dt.date.today()
            )
        estimate_returns = st.form_submit_button(
            "Estimate returns",
            help="Estimate the returns and volatility of the assets based on historical data from Yahoo Finance.",
            icon=":material/cloud_download:",
        )
    assets = st.session_state["assets"]
    st.subheader("Assets")
    for name, asset in assets.items():
        if estimate_returns and asset.ticker:
            growth, volatility = estimate_returns_and_volatility(
                asset.ticker, start_est, end_est
            )
            asset.returns = growth
            asset.volatility = volatility

    asset_df = pd.DataFrame.from_records(
        [{"name": name} | asdict(asset) for name, asset in assets.items()]
    ).set_index("name")
    asset_df["returns"] *= 100  # Convert to percentage
    asset_df["volatility"] *= 100  # Convert to percentage
    modified_asset_df = st.data_editor(
        asset_df,
        num_rows="dynamic",
        column_config={
            "name": st.column_config.TextColumn(
                "Asset Name",
                help="The name of the asset. This will be used in the allocation table.",
                max_chars=50,
            ),
            "returns": st.column_config.NumberColumn(
                "Annual Returns",
                help="The expected annual returns of the asset.",
                format="%.1f%%",
                min_value=0.0,
                max_value=100.0,
            ),
            "volatility": st.column_config.NumberColumn(
                "Annual Volatility",
                help="The expected annual volatility of the asset.",
                format="%.1f%%",
                min_value=0.0,
                max_value=100.0,
            ),
            "ticker": st.column_config.TextColumn(
                "Ticker",
                help="The ticker symbol of the asset. This is used to fetch historical data.",
                max_chars=10,
            ),
        },
    ).replace(np.nan, None)
    modified_asset_df["returns"] /= 100  # Convert back to decimal
    modified_asset_df["volatility"] /= 100  # Convert back to decimal
    assets = {name: Asset(**asset) for name, asset in modified_asset_df.iterrows()}

    st.subheader("Allocation")
    last_asset = list(assets.keys())[-1]
    allocation = st.session_state["allocation"]
    allocation = allocation.reindex(columns=assets.keys())
    allocation.index = allocation.index + (start_date - allocation.index[0])
    if not allocation.equals(st.session_state["allocation"]):
        for i in range(1, allocation.shape[1] - 1):
            max_allocation = 1 - allocation.iloc[:, :i].sum(axis=1)
            allocation.iloc[:, i] = allocation.iloc[:, i].clip(
                lower=0, upper=max_allocation
            )
        allocation.iloc[:, -1] = (1 - allocation.iloc[:, :-1].sum(axis=1)).clip(lower=0)
    allocation: pd.DataFrame = st.data_editor(
        allocation.reset_index(),
        hide_index=True,
        num_rows="dynamic",
        column_config={
            k: st.column_config.NumberColumn(min_value=0, max_value=1, format="percent")
            for k in assets
        }
        | {"Date": st.column_config.DateColumn()},
        key="allocation_editor",
    ).set_index("Date")

    st.session_state["assets"] = assets
    st.session_state["allocation"] = allocation
elif page == "Run simulation":
    st.header("Simulation")
    inflation = st.number_input("Inflation Rate (%)", value=0.0, step=0.1) / 100
    fees = st.number_input("Annual fees (%)", value=0.4, step=1.0) / 100
    starting_amount = st.session_state["starting_amount"]
    contributions = st.session_state["contributions"]
    assets = st.session_state["assets"]
    allocation = st.session_state["allocation"]

    mode = st.radio(
        "Select simulation method",
        ["Monte-carlo", "Backtesting"],
        index=0,
        horizontal=True,
    )

    columns = st.columns([0.15, 0.85])
    with columns[0]:
        simulate = st.button("Run Simulation")
    with columns[1]:
        if mode == "Monte-carlo":
            with st.popover("Settings", icon=":material/settings:"):
                num_simulations = st.number_input(
                    "Number of Simulations", value=10000, step=1
                )
                if st.toggle(
                    "Account for correlation between assets",
                    value=True,
                    help="If unchecked, the correlation between assets will be ignored.",
                ):
                    # Estimate the covariance matrix from historical returns
                    cov_matrix = pd.DataFrame(
                        {
                            name: fetch_monthly_returns(
                                asset.ticker, start=start_date, end=end_date
                            )
                            for name, asset in assets.items()
                        }
                    ).cov()
                else:
                    cov_matrix = None

    st.divider()

    if simulate:
        if mode == "Monte-carlo":
            simulation_dfs, cagr = monte_carlo(
                starting_amount=starting_amount,
                monthly_contributions=contributions["Monthly Contribution"],
                allocation=allocation,
                assets=assets,
                inflation=inflation,
                fees=fees,
                start_date=start_date,
                end_date=end_date,
                cov_matrix=cov_matrix,
                num_simulations=num_simulations,
                seed=42,
            )
        elif mode == "Backtesting":
            simulation_dfs, cagr = backtest(
                starting_amount=starting_amount,
                monthly_contributions=contributions["Monthly Contribution"],
                allocation=allocation,
                assets=assets,
                inflation=inflation,
                fees=fees,
                start_date=start_date,
                end_date=end_date,
            )
        st.session_state["simulation_dfs"] = simulation_dfs
        st.session_state["cagr"] = cagr
    else:
        simulation_dfs = st.session_state["simulation_dfs"]
        cagr = st.session_state["cagr"]

    if mode == "Monte-carlo":
        st.plotly_chart(
            plot_returns(
                simulation_dfs,
                confidence=st.radio(
                    "Select confidence interval",
                    [0.9, 0.95, 0.99],
                    format_func=lambda x: f"{x * 100:.0f}%",
                    index=1,
                    horizontal=True,
                ),
            )
        )
        metric = st.radio("Select metric", ["Returns", "CAGR"], horizontal=True)
        show_cumulative = st.toggle(
            "Show cumulative distribution",
            value=False,
            help="If checked, the histogram will show the cumulative distribution of the final portfolio value.",
            disabled=metric == "CAGR",
        )
        if metric == "Returns":
            st.plotly_chart(
                plot_hist_returns(
                    simulation_dfs.iloc[-1],
                    cumulative=show_cumulative,
                    xlabel="Portfolio Value (£)",
                    title="Portfolio Value Distribution",
                )
            )
        elif metric == "CAGR":
            st.plotly_chart(
                plot_hist_returns(
                    cagr * 100, xlabel="CAGR [%]", title="CAGR Distribution"
                )
            )
    else:
        st.plotly_chart(plot_returns(simulation_dfs))

        st.metric(
            "Portfolio Value (£)",
            f"£{simulation_dfs.iloc[-1, 0]:.2f}",
            delta=f"{cagr.mean() * 100:.2f}%",
        )


else:
    raise ValueError(f"Unknown page {page}")
