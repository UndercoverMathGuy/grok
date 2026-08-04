"""Plot the CE-from-one-hot run's train/test CE from metrics.json.

Run:  uv run python scripts/plot_ce_from_onehot.py [--run ce_from_onehot_t80_e390]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="ce_from_onehot_t80_e390")
    args = parser.parse_args()
    run_dir = Path("runs") / args.run

    m = json.loads((run_dir / "metrics.json").read_text())
    epochs = np.arange(len(m["test_losses"]))

    fig = go.Figure()
    fig.add_scatter(x=epochs, y=m["test_losses"], name="test CE",
                    line=dict(color="#4269d0", width=1.5))
    fig.add_scatter(x=epochs, y=m["train_losses"], name="train CE",
                    line=dict(color="#3ca951", width=1.5))
    fig.update_yaxes(type="log", title_text="CE (log)")
    fig.update_xaxes(title_text="epoch")
    fig.update_layout(title=f"CE + weight decay from the one-hot implant ({args.run})",
                      height=550, legend=dict(orientation="h", y=1.08))
    out = run_dir.parent / f"{args.run}.html"
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"wrote {out}")
