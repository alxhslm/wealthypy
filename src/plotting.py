import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

pd.options.plotting.backend = "plotly"


def plot_returns(
    simulation_dfs: dict[str, pd.DataFrame],
    confidence: float | None = None,
) -> go.Figure:
    fig = go.Figure()
    colors = dict(zip(simulation_dfs, px.colors.qualitative.Plotly))
    for key, df in simulation_dfs.items():
        if confidence:
            aggregate_df = pd.DataFrame(
                {
                    "median": df.median(axis=1),
                    "upper": df.quantile(0.5 + confidence / 2, axis=1),
                    "lower": df.quantile(0.5 - confidence / 2, axis=1),
                }
            )
        else:
            aggregate_df = df.median(axis=1).to_frame("median")
        # Add median line for each portfolio
        fig.add_trace(
            go.Scatter(
                x=aggregate_df.index,
                y=aggregate_df["median"],
                mode="lines",
                name=key,
                legendgroup=key,
                line_color=colors[key],
                showlegend=True,
            )
        )
        if confidence:
            fig.add_trace(
                go.Scatter(
                    x=aggregate_df.index,
                    y=aggregate_df["lower"],
                    mode="lines",
                    legendgroup=key,
                    line=dict(width=0),
                    showlegend=False,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=aggregate_df.index,
                    y=aggregate_df["upper"],
                    mode="lines",
                    legendgroup=key,
                    line=dict(width=0),
                    marker=dict(color=colors[key]),
                    fill="tonexty",
                    showlegend=False,
                )
            )
    fig.update_layout(
        title="Portfolio growth with selected confidence interval",
        xaxis_title="Date",
        yaxis_title="Portfolio Value (£)",
    )
    return fig


def plot_hist_returns(
    returns: pd.DataFrame, xlabel: str, title: str, cumulative: bool = False
) -> go.Figure:
    fig = px.histogram(
        returns if not isinstance(returns, pd.DataFrame) else returns.squeeze(),
        nbins=int(max(len(returns) / 50, 10)),
        histnorm="percent",
        cumulative="sum" if cumulative else None,
        title=title,
        labels={"value": xlabel, "variable": "Configuration"},
        marginal="box",
        color_discrete_sequence=px.colors.qualitative.Plotly,
        barmode="overlay",
    )
    fig.update_layout( yaxis_title="Percentage (%)")
    return fig


def plot_scatter(
    df: pd.DataFrame, x: str, y: str, x_title: str, y_title: str, title: str
) -> go.Figure:
    return df.plot.scatter(
        x=x, y=y, title=title, labels={"value": x_title, "aer": y_title}
    )
