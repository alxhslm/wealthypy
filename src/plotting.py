import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

pd.options.plotting.backend = "plotly"


def plot_returns(simulation_dfs: pd.DataFrame, confidence:float|None=None) -> go.Figure:
    if confidence:
        aggregate_df = pd.DataFrame(
            {
                "median": simulation_dfs.median(axis=1),
                "upper": simulation_dfs.quantile(0.5 + confidence / 2, axis=1),
                "lower": simulation_dfs.quantile(0.5 - confidence / 2, axis=1),
            }
        )
    else:
        aggregate_df = simulation_dfs.median(axis=1).to_frame("median")
    fig = go.Figure()

    # Add median line
    fig.add_trace(
        go.Scatter(
            x=aggregate_df.index,
            y=aggregate_df["median"],
            mode="lines",
            name="median",
            showlegend=False,
        )
    )

    if confidence:
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
    return fig


def plot_hist_returns(returns: pd.Series, xlabel: str, title: str, cumulative: bool=False) -> go.Figure:
    fig = px.histogram(
        returns,
        nbins=int(max(len(returns) / 50, 10)),
        histnorm="percent",
        cumulative="sum" if cumulative else None,
        title=title,
        labels={"value": xlabel},
        marginal="box"
    )
    fig.update_layout(showlegend=False, yaxis_title="Percentage (%)")
    return fig


def plot_scatter(
    df: pd.DataFrame, x: str, y: str, x_title: str, y_title: str, title: str
) -> go.Figure:
    return df.plot.scatter(
        x=x, y=y, title=title, labels={"value": x_title, "aer": y_title}
    )
