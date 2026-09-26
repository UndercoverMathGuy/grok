"""ladder_trend.py -- Part 2 tables and figures: what free training does near the exact ladder.

TRENDS ONLY.  No pass/fail, no pre-registration: the theorem is about the global minimiser of the
regularised loss (Wei et al. 2019 Thm 4.1), so where gradient descent happens to land is outside its
scope.  These tables say "we trained the nets and this is roughly what we saw".

Reads FINAL_RUNS/LADDER/train/scored.json (ladder_abl.py) and json/ceilings.json, writes
FINAL_RUNS/LADDER/trends.md and figs/trend_*.html.

Usage: uv run python heracles/ladder_trend.py
"""
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from ladder_core import OUT  # noqa: E402


def rng(xs, d=2):
    xs = [x for x in xs if x is not None]
    if not xs:
        return "—"
    if len(xs) == 1:
        return f"{xs[0]:.{d}f}"
    return f"{st.median(xs):.{d}f} [{min(xs):.{d}f}–{max(xs):.{d}f}]"


def irng(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return "—"
    return f"{int(st.median(xs))} [{min(xs)}–{max(xs)}]"


def kstar_plot(rs, n, p, q):
    """At q = nu the programme is linear and its optimum is the FULL spectrum, so the K <= 12 ladder's
    "12" is a truncation; plot nf = (p-1)/2 there."""
    return (p - 1) // 2 if abs(q - (n + 1)) < 1e-6 else rs[0]["K_star"]


def main():
    rows = []
    for sp in sorted((OUT / "train").glob("scored*.json")):
        if sp.name == "validate.json":
            continue
        d = json.loads(sp.read_text())
        if isinstance(d, list) and d and isinstance(d[0], dict) and "abl" in d[0]:
            rows += d
    if not rows:
        print("no scored*.json yet")
        return
    # merge the LOSS-ablation / Morwani scores (ladder_loss.py) onto the same runs
    lossrows = []
    for sp in sorted((OUT / "train").glob("loss_*.json")):
        if "validate" in sp.name:
            continue
        lossrows += json.loads(sp.read_text())
    lmap = {(r["run"], r["seed"]): r for r in lossrows}
    for r in rows:
        r["loss"] = lmap.get((r["run"], r["seed"]))
    cells = {}
    for r in rows:
        cells.setdefault((r["n"], r["p"], round(r["q"], 4), r["lam"]), []).append(r)

    lines = ["# Part 2 — trends from free training (gated chain, `--act prod`)\n",
             "Recipe: penalty off until train CE < 0.8 ln p, then the stated lambda; constant lr; grad-norm "
             "clip 1; init 4 (n = 2) / 14 (n >= 3); width m = 500 (the constructed K*-net needs "
             "`(n+1)K` <= 30 neurons, so every run has ~20x the width it needs).\n",
             "`quality` = the run's normalized margin as a fraction of the exact ceiling "
             "`mu N^-(nu-q)/q B Gamma_s(p)`; `junk` = 1 - clean fraction (the share of the logit-table "
             "energy that is not a function of D = sum a - c); `K_abl` = interface-ablation count; "
             "`K_d25` = number of frequencies whose removal costs >= 25% of the min-margin (this is the "
             "instrument that is EXACT on constructed nets, 4/4); `K_5pct` = Fourier-energy head count; "
             "`N_.05` = number of frequencies with >= 5% of the top amplitude — a TAIL statistic proportional "
             "to the number of frequencies present, not a count.\n",
             "| n | nu | p | q | s=nu/q | lam | seeds | acc | margin | quality | junk | **K_loss25** | K_loss10 | **K_5pct** | Morwani n5pct | Morwani eff | N_.05 | K_abl | K_d25 | K* (exact) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for (n, p, q, lam), rs in sorted(cells.items()):
        nu = n + 1
        lines.append(
            f"| {n} | {nu} | {p} | {q:g} | {nu / q:.3f} | {lam:g} | {len(rs)} | "
            f"{rng([r['acc'] for r in rs], 4)} | {rng([r['margin'] for r in rs], 2)} | "
            f"{rng([r['quality'] for r in rs], 3)} | {rng([1 - r['clean_frac'] for r in rs], 2)} | "
            f"**{irng([(r['loss'] or {}).get('normalised', {}).get('K_loss_25') for r in rs])}** | "
            f"{irng([(r['loss'] or {}).get('normalised', {}).get('K_loss_10') for r in rs])} | "
            f"**{irng([r['K_5pct'] for r in rs])}** | "
            f"{irng([((r['loss'] or {}).get('neuron') or {}).get('n_freq_5pct_mass') for r in rs])} | "
            f"{rng([((r['loss'] or {}).get('neuron') or {}).get('uniformity') for r in rs], 1)} | "
            f"{irng([r['N_05'] for r in rs])} | {irng([r['abl']['K_abl'] for r in rs])} | "
            f"{irng([r['abl'].get('K_d25') for r in rs])} | "
            f"{kstar_plot(rs, n, p, q)}{' (dense)' if abs(q - (n + 1)) < 1e-6 else ''} |")

    # the three designed comparisons
    lines += ["\n## The matched-s triple (same programme exponent s = 3/2, three different degrees)\n",
              "| (nu, q) | p | K* (exact) | **K_loss25** | **K_5pct** | Morwani n5pct | K_abl (appendix) | quality |",
              "|---|---|---|---|---|---|---|---|"]
    for n, q in [(2, 2.0), (3, 8 / 3), (4, 10 / 3)]:
        for (nn, p, qq, lam), rs in sorted(cells.items()):
            if nn == n and abs(qq - round(q, 4)) < 5e-3:
                lines.append(f"| ({n + 1}, {q:.4g}) | {p} | {kstar_plot(rs, n, p, q)} | "
                             f"**{irng([(r['loss'] or {}).get('normalised', {}).get('K_loss_25') for r in rs])}** | "
                             f"**{irng([r['K_5pct'] for r in rs])}** | "
                             f"{irng([((r['loss'] or {}).get('neuron') or {}).get('n_freq_5pct_mass') for r in rs])} | "
                             f"{irng([r['abl']['K_abl'] for r in rs])} | {rng([r['quality'] for r in rs], 3)} |")
    lines += ["\n## The dense endpoint q = nu, and the degree dial at fixed q = 3\n",
              "| n | nu | q | p | s | K* (exact) | **K_loss25** | **K_5pct** | Morwani n5pct | Morwani eff | N_.05 | quality |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for (n, p, q, lam), rs in sorted(cells.items()):
        if abs(q - (n + 1)) < 1e-6 or abs(q - 3.0) < 1e-6:
            lines.append(f"| {n} | {n + 1} | {q:g} | {p} | {(n + 1) / q:.3f} | {kstar_plot(rs, n, p, q)}"
                         f"{' (dense)' if abs(q - (n + 1)) < 1e-6 else ''} | "
                         f"**{irng([(r['loss'] or {}).get('normalised', {}).get('K_loss_25') for r in rs])}** | "
                         f"**{irng([r['K_5pct'] for r in rs])}** | "
                         f"{irng([((r['loss'] or {}).get('neuron') or {}).get('n_freq_5pct_mass') for r in rs])} | "
                         f"{rng([((r['loss'] or {}).get('neuron') or {}).get('uniformity') for r in rs], 1)} | "
                         f"{irng([r['N_05'] for r in rs])} | {rng([r['quality'] for r in rs], 3)} |")

    (OUT / "trends.md").write_text("\n".join(lines) + "\n")
    print("wrote", OUT / "trends.md")

    try:
        import plotly.graph_objects as go
    except ImportError:
        return
    fig = go.Figure()
    marks = {"K_loss25": ("circle", "#4C6EF5"), "K_5pct": ("diamond", "#2F9E44"),
             "N_.05": ("x", "#868E96"), "K_abl": ("triangle-up", "#ADB5BD"), "K_d25": ("square", "#DEE2E6")}
    seen = set()
    for (n, p, q, lam), rs in sorted(cells.items()):
        s = (n + 1) / q
        jit = {"K_loss25": -0.014, "K_5pct": 0.0, "N_.05": 0.014, "K_abl": 0.028, "K_d25": 0.042}
        for nm, get in (("K_abl", lambda r: r["abl"]["K_abl"]), ("K_d25", lambda r: r["abl"].get("K_d25")),
                        ("K_5pct", lambda r: r["K_5pct"]), ("N_.05", lambda r: r["N_05"]),
                        ("K_loss25", lambda r: (r.get("loss") or {}).get("normalised", {}).get("K_loss_25"))):
            ys = [get(r) for r in rs if get(r) is not None]
            if not ys:
                continue
            sym, col = marks[nm]
            fig.add_trace(go.Scatter(x=[s + jit[nm]] * len(ys), y=ys, mode="markers", name=nm,
                                     legendgroup=nm, showlegend=nm not in seen,
                                     marker=dict(size=9, symbol=sym, color=col, opacity=0.75),
                                     hovertext=[f"nu{n+1} q{q:g} p{p} lam{lam:g}"] * len(ys)))
            seen.add(nm)
        fig.add_trace(go.Scatter(x=[s], y=[kstar_plot(rs, n, p, q)], mode="markers", name="K* (exact)",
                                 legendgroup="K*", showlegend="K*" not in seen,
                                 marker=dict(size=18, symbol="star", color="black")))
        seen.add("K*")
    fig.update_layout(title="Trained counts vs the programme exponent s = nu/q (black stars = exact K*_free); "
                            "one marker per seed",
                      xaxis_title="s = nu/q   (1 = Morwani L_2,nu endpoint;  3/2 = the matched-s triple)",
                      yaxis_title="count", template="plotly_white", height=520, width=1000)
    fig.write_html(str(OUT / "figs" / "trend_counts.html"))
    try:
        fig.write_image(str(OUT / "figs" / "trend_counts.png"), scale=2)
    except Exception:                                                    # noqa: BLE001
        pass
    print("wrote trend figure")


if __name__ == "__main__":
    main()
