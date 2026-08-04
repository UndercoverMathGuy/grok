"""Thin, vanilla-plotly plotting helpers (replaces neel-plotly's `plot.py`).

Every function returns a plotly Figure — call .show() yourself, or keep
composing. All of them accept mlx arrays, numpy arrays, or lists via
to_numpy. Nothing here is clever: each helper is ~10 lines of plotly
express, so copy one and edit it when you need a variant.
"""

import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def to_numpy(x):
    """Convert mlx array / list / numpy to numpy."""
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, (list, tuple)):
        return np.stack([to_numpy(v) for v in x])
    return np.array(x)  # mx.array and scalars both convert cleanly


def _maybe_square(arr, p):
    """Unflatten a leading p^2 input dim to (p, p) for heatmaps."""
    if p is not None and arr.shape[0] == p * p:
        return arr.reshape(p, p, *arr.shape[1:])
    return arr


def imshow(tensor, p=None, xaxis="", yaxis="", facet_labels=None, **kwargs):
    """Diverging heatmap centered at 0 (RdBu). Pass p to auto-unflatten p^2 rows.

    Extra kwargs go to px.imshow (e.g. facet_col, zmin/zmax, x, y,
    color_continuous_scale='Blues' for one-sided data).
    """
    arr = _maybe_square(to_numpy(tensor), p)
    kwargs.setdefault("color_continuous_scale", "RdBu")
    if kwargs["color_continuous_scale"] == "RdBu":
        kwargs.setdefault("color_continuous_midpoint", 0.0)
    fig = px.imshow(arr, aspect="auto", labels={"x": xaxis, "y": yaxis}, **kwargs)
    if facet_labels:
        for annotation, label in zip(fig.layout.annotations, facet_labels):
            annotation.text = label
    return fig


def imshow_fourier(tensor, fourier, title="", **kwargs):
    """Heatmap of a (p^2,) / (p, p) tensor already in the 2D Fourier basis,
    with axes labelled by component names."""
    arr = np.squeeze(_maybe_square(to_numpy(tensor), fourier.p))
    fig = px.imshow(
        arr,
        x=fourier.names,
        y=fourier.names,
        labels={"x": "x component", "y": "y component"},
        title=title,
        color_continuous_scale="RdBu",
        color_continuous_midpoint=0.0,
        aspect="auto",
        **kwargs,
    )
    fig.update_traces(hovertemplate="%{x}(x) * %{y}(y)<br>value: %{z:.4f}")
    return fig


def lines(ys, x=None, labels=None, title="", xaxis="", yaxis="", log_y=False, hover=None):
    """Overlay line plots. ys: single 1D array or list/2D array (one line per row)."""
    ys = to_numpy(ys)
    if ys.ndim == 1:
        ys = ys[None, :]
    x = np.arange(ys.shape[1]) if x is None else to_numpy(x)
    fig = go.Figure()
    for i, y in enumerate(ys):
        name = labels[i] if labels is not None else str(i)
        fig.add_scatter(x=x, y=y, name=name, hovertext=hover)
    fig.update_layout(
        title=title,
        xaxis_title=xaxis,
        yaxis_title=yaxis,
        yaxis_type="log" if log_y else "linear",
        showlegend=labels is not None,
    )
    return fig


def histogram(x, title="", xaxis="", nbins=20, **kwargs):
    fig = go.Figure(go.Histogram(x=to_numpy(x).flatten(), nbinsx=nbins, **kwargs))
    fig.update_layout(title=title, xaxis_title=xaxis, yaxis_title="count", showlegend=False)
    return fig


def fourier_embed_bars(fourier_embed, fourier, title=""):
    """Grouped bars of cos/sin norms per frequency (paper Figures 3a/3b).

    fourier_embed: (p,) norms of each 1D Fourier component (Const, cos1, sin1, ...).
    """
    vals = to_numpy(fourier_embed)
    freqs = np.arange(1, fourier.p // 2 + 1)
    fig = go.Figure()
    fig.add_bar(x=freqs, y=vals[1::2], name="cos")
    fig.add_bar(x=freqs, y=vals[2::2], name="sin")
    fig.update_layout(barmode="group", title=title, xaxis_title="frequency k", yaxis_title="norm")
    return fig


def add_axis_toggle(fig, axis="y"):
    """Add a linear/log toggle button for an axis (replaces neel-plotly's)."""
    key = f"{axis}axis.type"
    menu = dict(
        type="buttons",
        direction="left",
        x=1.0,
        xanchor="right",
        y=1.15 if axis == "y" else 1.25,
        yanchor="top",
        buttons=[
            dict(label=f"{axis} linear", method="relayout", args=[{key: "linear"}]),
            dict(label=f"{axis} log", method="relayout", args=[{key: "log"}]),
        ],
    )
    fig.update_layout(updatemenus=list(fig.layout.updatemenus or ()) + [menu])
    return fig


def add_phase_lines(fig, positions, **kwargs):
    """Dashed vertical lines marking training phase boundaries."""
    for pos in positions:
        fig.add_vline(pos, line_dash="dash", opacity=0.7, **kwargs)
    return fig
