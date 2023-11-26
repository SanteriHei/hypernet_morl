# Some visualization utilities

from typing import List, Mapping

import plotly.graph_objects as go


def plot_pareto_front(
        performance_data: List[Mapping[str, List[str]]],
        current_step: int
) -> go.Figure:
    """Plots the current pareto front, and displays the standard deviation of 
        the expressed point on hover.

    Parameters
    ----------
    performance_data : List[Mapping[str, List[str]]]
        The performance of the agent from multiple repeated evaluations.

    Returns
    -------
    go.Figure
        The generated plotly figure.
    """
    current_front = [
            elem["avg_discounted_returns"] for elem in performance_data
    ]

    if len(current_front[0]) != 2:
        raise RuntimeError(("Visualization of pareto front is currently "
                            "supported for only 2 objectives"))

    x_data, y_data = map(list, zip(*current_front))
    text_data = [
            elem["std_discounted_returns"] for elem in performance_data
    ]
    text_data = [
            f"std for obj 1 {elem[0]:.3f}, obj 2 {elem[1]:.3f}" for elem in text_data
    ]
    fig = go.Figure(
        data=go.Scatter(
            x=x_data,
            y=y_data,
            text=text_data,
            mode="markers"
        ),
        layout_title_text=f"Pareto front at step {current_step}",
        layout_xaxis_title="Objective 1",
        layout_yaxis_title="Objective 2"
    )
    return fig

    
