import streamlit as st
import pandas as pd
import datetime as dt

from src.backtesting import estimate_returns_and_volatility, fetch_historic_returns
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
    num_simulations: int = 1,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    index = pd.date_range(start_date, end_date, freq="ME").date
    return run_simulation(
        starting_amount=starting_amount,
        monthly_contributions=monthly_contributions.reindex(index, method="ffill"),
        growth=sample_returns(
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
        growth=fetch_historic_returns(
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
    with st.expander("Estimate returns and volatility", icon=":material/cloud_download:"):
        start_est = st.date_input(
            "Start Date", value=dt.date(1900, 1, 1), max_value=dt.date.today()
        )
        end_est = st.date_input(
            "Start Date", value=dt.date.today(), max_value=dt.date.today()
        )
        estimate = st.button("Estimate")
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
                    ticker = st.text_input("Ticker", asset.ticker, key=f"{name}_ticker")
                    if estimate and ticker is not None:
                        asset.returns, asset.volatility = (
                            estimate_returns_and_volatility(
                                ticker, start=start_est, end=end_est
                            )
                        )
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
                        ticker=ticker,
                    )
                    if len(st.session_state["default_assets"]) > 1:
                        if st.button(":material/delete:", key=f"{name}_delete"):
                            to_delete.append(name)

    with tabs[1]:
        last_asset = list(selected_assets.keys())[-1]
        default_allocation = st.session_state["default_allocation"]
        default_allocation.index = default_allocation.index + (
            start_date - default_allocation.index[0]
        )
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
            num_simulations = st.number_input(
                "Number of Simulations", value=10000, step=1
            )

st.divider()


if simulate:
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

        st.plotly_chart(
            plot_hist_returns(
                simulation_dfs.iloc[-1],
                cumulative=st.toggle("Show cumulative distribution", False),
            )
        )

        quantiles = pd.DataFrame(
            {
                "Portfolio Value": compute_quantiles(simulation_dfs.iloc[-1]),
                "AER": compute_quantiles(aer),
            }
        )
        st.dataframe(
            quantiles,
            column_config={
                "Portfolio Value": st.column_config.NumberColumn(format="£%.2f"),
                "AER": st.column_config.NumberColumn(format="percent"),
            },
        )

        st.plotly_chart(
            plot_scatter(
                pd.DataFrame({"value": simulation_dfs.iloc[-1], "aer": 100 * aer}),
                x="value",
                y="aer",
                title="Portfolio Value vs AER",
                x_title="Portfolio Value (£)",
                y_title="AER (%)",
            )
        )
    elif mode == "Backtesting":
        simulation_df, aer = backtest(
            starting_amount=starting_amount,
            monthly_contributions=contributions["Monthly Contribution"],
            allocation=allocation,
            assets=selected_assets,
            inflation=inflation,
            fees=fees,
            start_date=start_date,
            end_date=end_date,
        )
        st.plotly_chart(plot_returns(simulation_df))
        st.metric(
            "Portfolio Value (£)",
            f"£{simulation_df.iloc[-1, 0]:.2f}",
            delta=f"{aer.mean() * 100:.2f}%",
        )
    else:
        raise ValueError(f"Unknown mode {mode}")
