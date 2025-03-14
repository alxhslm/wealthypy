import streamlit as st
import numpy as np
import pandas as pd
import datetime as dt
import plotly.graph_objects as go

pd.options.plotting.backend = "plotly"


@st.cache_data
def simulator(
    starting_amount: float,
    monthly_contributions: pd.Series,
    equity_return: float,
    equity_volatility: float,
    bond_return: float,
    inflation: float,
    fees: float,
    equity_weight: pd.Series,
    years: int,
    num_simulations: int = 1,
    seed: int = 42,
) -> pd.DataFrame:
    months = years * 12
    index = pd.date_range(dt.date.today(), periods=months, freq="ME").date

    monthly_contributions = monthly_contributions.reindex(
        index, method="ffill"
    )

    equity_weight = equity_weight.reindex(
        index, method="ffill"
    )
    bond_weight = 1 - equity_weight

    # Generate returns
    np.random.seed(seed)
    equity_monthly_returns = np.random.normal(
        equity_return / 12, equity_volatility / np.sqrt(12), (months, num_simulations)
    )
    bond_monthly_returns = np.full_like(equity_monthly_returns, bond_return / 12)

    # Calculate growth
    equity_growth = (1 + equity_monthly_returns) * equity_weight.values[:, np.newaxis]
    bond_growth = (1 + bond_monthly_returns) * bond_weight.values[:, np.newaxis]
    total_growth = equity_growth + bond_growth - fees/12

    # Compute portfolio values
    portfolio = pd.DataFrame(
        index=index,
        columns=range(num_simulations)
    )

    portfolio.iloc[0, :] = starting_amount
    for i in range(1, months):
        portfolio.iloc[i, :] = (
            portfolio.iloc[i - 1, :] + monthly_contributions.iloc[i - 1]
        ) * total_growth[i, :]

    # Adjust for inflation
    inflation_adjustment = (1 + inflation / 12) ** np.arange(months)
    return portfolio.divide(inflation_adjustment, axis=0)


# Streamlit UI
st.title("Portfolio Growth Simulator")

with st.sidebar:
    starting_amount = st.number_input("Starting Amount (£)", value=10000, step=1000)
    years = st.number_input("Number of Years", value=38, step=1)
    inflation = st.number_input("Inflation Rate (%)", value=0.0, step=0.1) / 100
    fees = st.number_input("Annual fees (%)", value=0.4, step=1.0) / 100

    tabs = st.tabs(["Equities", "Bonds"])
    with tabs[0]:
        equity_return = (
            st.number_input("Equity Annual Return (%)", value=5.0, step=0.1) / 100
        )
        equity_volatility = (
            st.number_input("Equity Annual Volatility (%)", value=25.0, step=0.1) / 100
        )
    with tabs[1]:
        bond_return = (
            st.number_input("Bond Annual Return (%)", value=0.5, step=0.1) / 100
        )
    num_simulations = st.number_input("Number of Simulations", value=10000, step=1)

# Collect variable monthly contributions
st.subheader("Schedule", help="Enter the monthly contribution and equity weight")
schedule = st.data_editor(
    pd.DataFrame(
        {
            "Date": [dt.date.today()],
            "Monthly Contribution": [1000.0],
            "Equity Weight": [50.0],
        }
    ),
    num_rows="dynamic",
    hide_index=True,
).set_index("Date")



# Run Simulation
dfs = simulator(
    starting_amount=starting_amount,
    monthly_contributions=schedule["Monthly Contribution"],
    equity_return=equity_return,
    equity_volatility=equity_volatility,
    equity_weight=schedule["Equity Weight"]/ 100,
    bond_return=bond_return,
    inflation=inflation,
    fees=fees,
    years=years,
    num_simulations=num_simulations,
    seed=42,
)

confidence = st.radio("Select confidence interval", [0.9, 0.95, 0.99], format_func=lambda x:f"{x*100:.0f}%", index=1, horizontal=True)
df = pd.DataFrame({"median": dfs.mean(axis=1), "upper": dfs.quantile(0.5+confidence/2, axis=1), "lower": dfs.quantile(0.5-confidence/2, axis=1)})


# Plotting with Plotly
fig = go.Figure()

# Add mean line
fig.add_trace(
    go.Scatter(x=df.index, y=df["median"], mode="lines", name="Mean", showlegend=False)
)

# Add standard deviation shaded area
fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["upper"],
        mode="lines",
        name="95% Confidence Interval (upper)",
        line=dict(width=0),
        showlegend=False,
    )
)

fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["lower"],
        mode="lines",
        name="95% Confidence Interval (lower)" ,
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
    dfs.iloc[-1]
    .hist(
        nbins=int(max(num_simulations / 50, 10)),
        histnorm="percent",
        cumulative=st.toggle("Show cumulative distribution", False),
        title="Final portfolio value",
        labels={"value": "Portfolio Value (£)"},
    )
    .update_layout(showlegend=False, yaxis_title="Percentage (%)")
)
