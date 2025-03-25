from dataclasses import dataclass
import streamlit as st
import numpy as np
import pandas as pd
import datetime as dt
import plotly.graph_objects as go
import yfinance as yf

pd.options.plotting.backend = "plotly"


@dataclass
class Asset:
    returns: float
    volatility: float
    ticker: str | None = None


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
    years = (growth.index[-1] - growth.index[0]).total_seconds() / (365.25 * 24 * 3600)
    aer = growth.divide(inflation_adjustment, axis=0).prod().pow(1 / years) - 1
    return portfolio.divide(inflation_adjustment, axis=0), aer


def _sample_volatile_growth(
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


def _fetch_historic_growth(
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
    num_simulations: int = 1,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    index = pd.date_range(start_date, end_date, freq="ME").date
    return run_simulation(
        starting_amount=starting_amount,
        monthly_contributions=monthly_contributions.reindex(index, method="ffill"),
        growth=_sample_volatile_growth(
            assets=assets,
            allocation=allocation.reindex(index, method="ffill"),
            num_simulations=num_simulations,
            seed=seed,
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
        monthly_contributions=monthly_contributions.reindex(index, method="ffill"),
        growth=_fetch_historic_growth(
            assets=assets, allocation=allocation.reindex(index, method="ffill")
        )
        - fees / 12,
        inflation=inflation,
    )


# Streamlit UI
st.set_page_config(
    layout="wide", page_icon=":chart", page_title="Portfolio Growth Simulator"
)
st.title("Portfolio Growth Simulator")

if "default_assets" not in st.session_state:
    st.session_state["default_assets"] = {
        "Equities": Asset(returns=5.0 / 100, volatility=25 / 100, ticker="VWRP.L"),
        "Bonds": Asset(returns=2.0 / 100, volatility=0.0 / 100, ticker="VAGS.L"),
    }

if "default_allocation" not in st.session_state:
    default_allocation = pd.DataFrame(
        {
            k: [1.0 / len(st.session_state["default_assets"])]
            for k in st.session_state["default_assets"]
        },
        index=[dt.date.today() - dt.timedelta(days=3 * 365.25)],
    )
    default_allocation.index.name = "Date"
    st.session_state["default_allocation"] = default_allocation


def _get_name(assets: dict[str, Asset]) -> str:
    new_name = "New asset"
    i = 1
    while new_name in assets:
        new_name = f"New asset ({i})"
        i += 1
    return new_name


with st.sidebar:
    st.header("Simulation")
    container = st.container()
    end_date = st.date_input(
        "End Date", max_value=dt.date.today(), value=dt.date.today()
    )
    with container:
        start_date = st.date_input(
            "Start Date",
            value=dt.date.today() - dt.timedelta(days=3 * 365.25),
            max_value=end_date,
        )
    st.header("Funds")
    starting_amount = st.number_input("Starting amount (£)", value=10000, step=1000)
    contributions = st.data_editor(
        pd.DataFrame(
            {
                "Date": [start_date],
                "Monthly Contribution": [1000.0],
            }
        ),
        num_rows="dynamic",
        hide_index=True,
        column_config={
            "Monthly Contribution": st.column_config.NumberColumn(format="£%.2f")
        },
    ).set_index("Date")

    st.header("Portfolio")
    tabs = st.tabs(["Assets", "Allocation"])
    with tabs[0]:
        container = st.container()

        if st.button(":material/add:"):
            new_name = _get_name(st.session_state["default_assets"])
            st.session_state["default_assets"][new_name] = Asset(
                returns=0.0, volatility=0.0
            )
            st.session_state["default_allocation"][new_name] = 0.0

        with container:
            selected_assets = {}
            renamed_assets = {}
            to_delete = []
            for name, asset in st.session_state["default_assets"].items():
                with st.expander(f"{name}"):
                    new_name = st.text_input("Name", name, key=f"{name}_name")
                    if new_name != name:
                        renamed_assets[name] = new_name
                    selected_assets[name] = Asset(
                        returns=st.number_input(
                            "Annual Returns (%)",
                            value=asset.returns * 100,
                            step=0.1,
                            key=f"{name}_return",
                        )
                        / 100,
                        volatility=(
                            st.number_input(
                                "Annual Volatility (%)",
                                value=asset.volatility * 100,
                                step=0.1,
                                key=f"{name}_volatility",
                            )
                            / 100
                        ),
                        ticker=st.text_input(
                            "Ticker", asset.ticker, key=f"{name}_ticker"
                        ),
                    )
                    if len(st.session_state["default_assets"]) > 1:
                        if st.button(":material/delete:", key=f"{name}_delete"):
                            to_delete.append(name)

    with tabs[1]:
        last_asset = list(selected_assets.keys())[-1]
        default_allocation = st.session_state["default_allocation"]
        default_allocation.index = default_allocation.index + (start_date -  default_allocation.index[0])
        allocation: pd.DataFrame = st.data_editor(
            default_allocation.reset_index(),
            num_rows="dynamic",
            hide_index=True,
            column_config={
                k: st.column_config.NumberColumn(
                    min_value=0, max_value=1, format="percent"
                )
                for k in selected_assets
            },
            key=id(st.session_state["default_allocation"]),
        ).set_index("Date")
        if not allocation.equals(st.session_state["default_allocation"]):
            for i in range(1, allocation.shape[1] - 1):
                max_allocation = 1 - allocation.iloc[:, :i].sum(axis=1)
                allocation.iloc[:, i] = allocation.iloc[:, i].clip(
                    lower=0, upper=max_allocation
                )
            allocation.iloc[:, -1] = (1 - allocation.iloc[:, :-1].sum(axis=1)).clip(
                lower=0
            )
            st.session_state["default_allocation"] = allocation
            st.rerun()

    if to_delete:
        st.session_state["default_assets"] = {
            k: v for k, v in selected_assets.items() if k not in to_delete
        }
        st.session_state["default_allocation"] = allocation.drop(columns=to_delete)
        st.rerun()

    if renamed_assets:
        st.session_state["default_assets"] = selected_assets
        for old_name, new_name in renamed_assets.items():
            st.session_state["default_assets"][new_name] = st.session_state[
                "default_assets"
            ].pop(old_name)
        st.session_state["default_allocation"] = allocation.rename(
            columns=renamed_assets
        )
        st.rerun()

    st.header("Settings")
    inflation = st.number_input("Inflation Rate (%)", value=0.0, step=0.1) / 100
    fees = st.number_input("Annual fees (%)", value=0.4, step=1.0) / 100

mode = st.radio("Select mode", ["Monte-carlo", "Backtesting"], index=0, horizontal=True)

columns = st.columns([0.15, 0.85])
with columns[0]:
    simulate = st.button("Run Simulation")
with columns[1]:
    if mode == "Monte-carlo":
        with st.popover(":material/settings:"):
            num_simulations = st.number_input("Number of Simulations", value=10000, step=1)

st.divider()

if simulate:
    confidence = st.radio(
        "Select confidence interval",
        [0.9, 0.95, 0.99],
        format_func=lambda x: f"{x * 100:.0f}%",
        index=1,
        horizontal=True,
    )

    if mode == "Monte-carlo":
        simulation_dfs, aer = monte_carlo(
            starting_amount=starting_amount,
            monthly_contributions=contributions["Monthly Contribution"],
            allocation=allocation,
            assets=selected_assets,
            inflation=inflation,
            fees=fees,
            start_date=start_date,
            end_date=end_date,
            num_simulations=num_simulations,
            seed=42,
        )
    elif mode == "Backtesting":
        simulation_dfs, aer = backtest(
            starting_amount=starting_amount,
            monthly_contributions=contributions["Monthly Contribution"],
            allocation=allocation,
            assets=selected_assets,
            inflation=inflation,
            fees=fees,
            start_date=start_date,
            end_date=end_date,
        )
    else:
        raise ValueError(f"Unknown mode {mode}")

    aggregate_df = pd.DataFrame(
        {
            "median": simulation_dfs.mean(axis=1),
            "upper": simulation_dfs.quantile(0.5 + confidence / 2, axis=1),
            "lower": simulation_dfs.quantile(0.5 - confidence / 2, axis=1),
        }
    )

    # Plotting with Plotly
    fig = go.Figure()

    # Add mean line
    fig.add_trace(
        go.Scatter(
            x=aggregate_df.index,
            y=aggregate_df["median"],
            mode="lines",
            name="Mean",
            showlegend=False,
        )
    )

    # Add standard deviation shaded area
    fig.add_trace(
        go.Scatter(
            x=aggregate_df.index,
            y=aggregate_df["upper"],
            mode="lines",
            name="95% Confidence Interval (upper)",
            line=dict(width=0),
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=aggregate_df.index,
            y=aggregate_df["lower"],
            mode="lines",
            name="95% Confidence Interval (lower)",
            line=dict(width=0),
            fill="tonexty",
            showlegend=False,
        )
    )

    fig.update_layout(
        title="Portfolio growth with selected confidence interval",
        xaxis_title="Date",
        yaxis_title="Portfolio Value (£)",
    )

    st.plotly_chart(fig)

    st.plotly_chart(
        simulation_dfs.iloc[-1]
        .hist(
            nbins=int(max(simulation_dfs.shape[1] / 50, 10)),
            histnorm="percent",
            cumulative=st.toggle("Show cumulative distribution", False),
            title="Final portfolio value",
            labels={"value": "Portfolio Value (£)"},
        )
        .update_layout(showlegend=False, yaxis_title="Percentage (%)")
    )

    quantiles = pd.DataFrame(
        {
            "Portfolio Value": simulation_dfs.iloc[-1].quantile([0.1, 0.5, 0.9]),
            "AER": aer.quantile([0.1, 0.5, 0.9]),
        }
    )
    quantiles.index = (
        quantiles.index.to_series()
        .apply(lambda x: f"{x * 100:.0f}%")
        .rename("Quantile")
    )
    quantiles.loc["Mean", :] = [simulation_dfs.iloc[-1].mean(), aer.mean()]
    st.dataframe(
        quantiles,
        column_config={
            "Portfolio Value": st.column_config.NumberColumn(format="£%.2f"),
            "AER": st.column_config.NumberColumn(format="percent"),
        },
    )

    st.plotly_chart(
        pd.DataFrame({"value": simulation_dfs.iloc[-1], "aer": 100 * aer}).plot.scatter(
            x="value",
            y="aer",
            title="Portfolio Value vs AER",
            labels={"value": "Portfolio Value (£)", "aer": "AER [%]"},
        )
    )
