from dataclasses import dataclass
import streamlit as st
import numpy as np
import pandas as pd
import datetime as dt
import plotly.graph_objects as go

pd.options.plotting.backend = "plotly"


@dataclass
class Asset:
    returns: float
    volatility: float


@st.cache_data
def simulator(
    starting_amount: float,
    monthly_contributions: pd.Series,
    assets: dict[str, Asset],
    allocation: pd.DataFrame,
    inflation: float,
    fees: float,
    years: int,
    num_simulations: int = 1,
    seed: int = 42,
) -> pd.DataFrame:
    months = years * 12
    index = pd.date_range(dt.date.today(), periods=months, freq="ME").date

    monthly_contributions = monthly_contributions.reindex(index, method="ffill")

    total_growth = np.zeros((months, num_simulations))
    np.random.seed(seed)
    allocation = allocation.reindex(index, method="ffill")
    for name, asset in assets.items():
        monthly_returns = np.random.normal(
            asset.returns / 12,
            asset.volatility / np.sqrt(12),
            (months, num_simulations),
        )
        growth = (1 + monthly_returns) * allocation[name].values[:, np.newaxis]
        total_growth += growth

    total_growth -= fees / 12

    # Compute portfolio values
    portfolio = pd.DataFrame(index=index, columns=range(num_simulations))
    portfolio.iloc[0, :] = starting_amount
    for i in range(1, months):
        portfolio.iloc[i, :] = (
            portfolio.iloc[i - 1, :] + monthly_contributions.iloc[i - 1]
        ) * total_growth[i, :]

    # Adjust for inflation
    inflation_adjustment = (1 + inflation / 12) ** np.arange(months)
    return portfolio.divide(inflation_adjustment, axis=0)


# Streamlit UI
st.set_page_config(
    layout="wide", page_icon=":chart", page_title="Portfolio Growth Simulator"
)
st.title("Portfolio Growth Simulator")

if "default_assets" not in st.session_state:
    st.session_state["default_assets"] = {
        "Equities": Asset(returns=5.0 / 100, volatility=25 / 100),
        "Bonds": Asset(returns=2.0 / 100, volatility=0.0 / 100),
    }

if "default_allocation" not in st.session_state:
    st.session_state["default_allocation"] = pd.DataFrame(
        {"Date": [dt.date.today()]}
        | {
            k: [1.0 / len(st.session_state["default_assets"])]
            for k in st.session_state["default_assets"]
        },
    ).set_index("Date")


def _get_name(assets: dict[str, Asset]) -> str:
    new_name = "New asset"
    i = 1
    while new_name in assets:
        new_name = f"New asset ({i})"
        i += 1
    return new_name


with st.sidebar:
    st.header("Funds")
    starting_amount = st.number_input("Starting amount (£)", value=10000, step=1000)
    contributions = st.data_editor(
        pd.DataFrame(
            {
                "Date": [dt.date.today()],
                "Monthly Contribution": [1000.0],
            }
        ),
        num_rows="dynamic",
        hide_index=True,
        column_config={"Monthly Contribution":st.column_config.NumberColumn(format="£%.2f")},
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

        with container:
            selected_assets = {}
            asset_names = {}
            to_delete = []
            for name, asset in st.session_state["default_assets"].items():
                with st.expander(f"{name}"):
                    new_name = st.text_input("Name", name, key=f"{name}_name")
                    if new_name != name:
                        asset_names[name] = new_name
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
                    )
                    if st.button(":material/delete:", key=f"{name}_delete"):
                        to_delete.append(name)

    with tabs[1]:
        allocation = (
            st.data_editor(
                st.session_state["default_allocation"].multiply(100).reset_index(),
                num_rows="dynamic",
                hide_index=True,
            ).set_index("Date")
            / 100
        )

    if to_delete:
        st.session_state["default_assets"] = {
            k: v for k, v in selected_assets.items() if k not in to_delete
        }
        st.session_state["default_allocation"] = allocation.drop(columns=to_delete)
        st.rerun()
    if asset_names:
        st.session_state["default_assets"] = selected_assets
        for old_name, new_name in asset_names.items():
            st.session_state["default_assets"][new_name] = st.session_state[
                "default_assets"
            ].pop(old_name)
        st.session_state["default_allocation"] = allocation.rename(columns=asset_names)
        st.rerun()
    st.header("Settings")
    inflation = st.number_input("Inflation Rate (%)", value=0.0, step=0.1) / 100
    fees = st.number_input("Annual fees (%)", value=0.4, step=1.0) / 100


cols = st.columns(2)
with cols[0]:
    years = st.number_input("Number of Years", value=38, step=1)
with cols[1]:
    num_simulations = st.number_input("Number of Simulations", value=10000, step=1)
run_simulation = st.button("Run Simulation")
st.divider()


if run_simulation:
    confidence = st.radio(
        "Select confidence interval",
        [0.9, 0.95, 0.99],
        format_func=lambda x: f"{x * 100:.0f}%",
        index=1,
        horizontal=True,
    )

    simulation_dfs = simulator(
        starting_amount=starting_amount,
        monthly_contributions=contributions["Monthly Contribution"],
        allocation=allocation,
        assets=selected_assets,
        inflation=inflation,
        fees=fees,
        years=years,
        num_simulations=num_simulations,
        seed=42,
    )

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
            nbins=int(max(num_simulations / 50, 10)),
            histnorm="percent",
            cumulative=st.toggle("Show cumulative distribution", False),
            title="Final portfolio value",
            labels={"value": "Portfolio Value (£)"},
        )
        .update_layout(showlegend=False, yaxis_title="Percentage (%)")
    )

    quantiles = (
        simulation_dfs.iloc[-1].quantile([0.1, 0.5, 0.9]).to_frame("Portfolio Value")
    )
    quantiles.index = (
        quantiles.index.to_series()
        .apply(lambda x: f"{x * 100:.0f}%")
        .rename("Quantile")
    )
    st.dataframe(quantiles.style.format("£{:.2f}"))
