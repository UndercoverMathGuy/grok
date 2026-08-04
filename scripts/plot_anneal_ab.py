"""Plot the linear vs two-stage anneal A/B: losses over epochs with the
target-mixture schedule marked.

Run:  uv run python scripts/plot_anneal_ab.py [--runs 42_into_37_mse_linear 42_into_37_mse_2stage]
Writes runs/<name>.html per run and a combined runs/anneal_ab.html.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

EVERY = 10  # plot every Nth epoch; full curves make a sluggish html


def run_traces(run_dir: Path):
    m = json.loads((run_dir / "metrics.json").read_text())
    spec = json.loads((run_dir / "anneal.json").read_text())
    epochs = np.arange(0, len(m["test_losses"]), EVERY)
    curves = {k: np.array(m[k])[::EVERY] for k in ["test_losses", "train_losses", "kd_losses"]}
    w_teacher = np.array(m["w_teacher"])[::EVERY]
    w_student = np.array(m["w_student"])[::EVERY]
    return epochs, curves, w_teacher, w_student, spec["transitions"]


def build_figure(runs):
    titles = [r.name for r in runs]
    fig = make_subplots(
        rows=2, cols=len(runs), shared_xaxes=True, subplot_titles=titles,
        row_heights=[0.75, 0.25], vertical_spacing=0.06, horizontal_spacing=0.06,
    )
    colors = {"test_losses": "#4269d0", "train_losses": "#3ca951", "kd_losses": "#9498a0"}
    names = {"test_losses": "test CE", "train_losses": "train CE", "kd_losses": "kd loss"}

    for col, run_dir in enumerate(runs, start=1):
        epochs, curves, w_t, w_s, transitions = run_traces(run_dir)
        for key, y in curves.items():
            fig.add_scatter(
                x=epochs, y=y, name=names[key], legendgroup=key,
                showlegend=col == 1, line=dict(color=colors[key], width=2),
                row=1, col=col,
            )
        fig.add_scatter(
            x=epochs, y=w_t, name="w_teacher", legendgroup="wt", showlegend=col == 1,
            line=dict(color="#ff725c", width=2), row=2, col=col,
        )
        fig.add_scatter(
            x=epochs, y=w_s, name="w_student", legendgroup="ws", showlegend=col == 1,
            line=dict(color="#4269d0", width=2, dash="dot"), row=2, col=col,
        )
        for t in transitions[1:]:
            fig.add_vline(t["epoch"], line_dash="dash", opacity=0.15, row=1, col=col)

    fig.update_yaxes(type="log", title_text="loss (log)", row=1, col=1)
    fig.update_yaxes(type="log", row=1, col=2)
    fig.update_yaxes(title_text="target weight", row=2, col=1)
    fig.update_xaxes(title_text="epoch", row=2)
    fig.update_layout(
        title="Annealed distillation 42 → 37 (mse): linear vs two-stage",
        height=650, legend=dict(orientation="h", y=1.09),
        margin=dict(t=110),
    )
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+",
                        default=["42_into_37_mse_linear", "42_into_37_mse_2stage"])
    parser.add_argument("--out", default="runs/anneal_ab.html")
    args = parser.parse_args()

    fig = build_figure([Path("runs") / r for r in args.runs])
    fig.write_html(args.out, include_plotlyjs="cdn")
    print(f"wrote {args.out}")
