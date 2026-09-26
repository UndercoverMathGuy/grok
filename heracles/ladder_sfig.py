"""K vs s = nu/q: exact programme count (ideal) against trained loss-ablation counts.
Reads FINAL_RUNS/LADDER json only; no training. Writes figs/s_collapse.{html,png}."""
import json, glob, collections, statistics as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = "FINAL_RUNS/LADDER/"
cells = json.load(open(ROOT + "json/cells.json"))
small = json.load(open(ROOT + "json/prime_axis_small.json"))

def exact(p):
    """exact K*_free by s for prime p (chain cells + prime-axis pass); s=1 is dense (p-1)/2"""
    out = {}
    for c in cells.values():
        if c["family"] == "chain" and c["p"] == p and c["s"] > 1.0 + 1e-9:
            out[round(c["s"], 4)] = c["K_star_free"]
    for v in small[str(p)]["per_s"].values():
        out.setdefault(round(v["s"], 4), v["K_star_free"])
    return dict(sorted(out.items()))

runs = []
for f in sorted(glob.glob(ROOT + "train/loss_n*.json")):
    runs += json.load(open(f))
grp = collections.defaultdict(list)
for r in runs:
    lam_ok = not (r["nu"] == 3 and abs(r["q"] - 2) < 1e-6 and r["lam"] != 1e-5)  # one lam per cell
    if lam_ok:
        grp[(r["nu"], round(r["q"], 4), r["p"])].append(r["normalised"]["K_loss_25"])

INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
HUE = {3: "#2a78d6", 4: "#eb6834", 5: "#1baf7a"}
fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.09, subplot_titles=(
    "Count against s = ν/q  (log count; s = 1 is the dense endpoint)",
    "Same points against 1/(1 − q/ν): the ideal is close to a line"))

for p, dash in ((59, "solid"), (71, "dot")):
    e = exact(p)
    s = np.array(list(e)); k = np.array(list(e.values()))
    fig.add_trace(go.Scatter(x=np.r_[1.0, s], y=np.r_[(p - 1) // 2, k], mode="lines+markers",
        line=dict(color=INK if p == 59 else MUTED, width=2, dash=dash, shape="linear"),
        marker=dict(size=8, symbol="diamond", color=INK if p == 59 else MUTED),
        name=f"exact optimum K*, p = {p}", legendgroup=f"e{p}",
        hovertemplate="s = %{x:.3f}<br>exact K* = %{y}<extra>p = " + str(p) + "</extra>"), 1, 1)
    fig.add_trace(go.Scatter(x=s / (s - 1), y=k, mode="lines+markers",
        line=dict(color=INK if p == 59 else MUTED, width=2, dash=dash),
        marker=dict(size=8, symbol="diamond", color=INK if p == 59 else MUTED),
        showlegend=False, legendgroup=f"e{p}",
        hovertemplate="1/(1−q/ν) = %{x:.2f}<br>exact K* = %{y}<extra>p = " + str(p) + "</extra>"), 1, 2)

# least-squares line through the origin on the p = 59 exact points (right panel)
e = exact(59); s = np.array(list(e)); k = np.array(list(e.values())); x = s / (s - 1)
c = float((x * k).sum() / (x * x).sum())
fig.add_trace(go.Scatter(x=[0, 6.4], y=[0, 6.4 * c], mode="lines", line=dict(color=MUTED, width=1, dash="dash"),
    name=f"K = {c:.2f} / (1 − q/ν)", hoverinfo="skip"), 1, 2)

seen = set()
for (nu, q, p), ks in sorted(grp.items()):
    sv = nu / q
    jit = {3: -0.012, 4: 0.0, 5: 0.012}[nu]
    sym = "circle" if p == 59 else "circle-open"
    for col, xv in ((1, sv + jit), (2, None if sv < 1 + 1e-9 else sv / (sv - 1) + jit * 8)):
        if xv is None:
            continue
        fig.add_trace(go.Scatter(x=[xv] * len(ks), y=ks, mode="markers",
            marker=dict(size=7, color=HUE[nu], opacity=0.45, symbol=sym, line=dict(width=1.5, color=HUE[nu])),
            showlegend=False, legendgroup=f"nu{nu}",
            hovertemplate=f"ν = {nu}, q = {q:g}, p = {p}<br>one seed: K_loss = %{{y}}<extra></extra>"), 1, col)
        fig.add_trace(go.Scatter(x=[xv], y=[st.median(ks)], mode="markers",
            marker=dict(size=13, color=HUE[nu], symbol=sym, line=dict(width=2, color=SURF if p == 59 else HUE[nu])),
            name=f"trained, ν = {nu} (median of seeds)", legendgroup=f"nu{nu}", showlegend=(nu not in seen and col == 1),
            hovertemplate=f"ν = {nu}, q = {q:g}, p = {p}<br>median K_loss = %{{y}}  (n = {len(ks)})<extra></extra>"), 1, col)
        if col == 1:
            seen.add(nu)

fig.update_yaxes(type="log", title_text="number of frequencies", row=1, col=1, tickvals=[3, 4, 5, 7, 10, 15, 20, 30], gridcolor=GRID)
fig.update_yaxes(title_text="number of frequencies", row=1, col=2, range=[0, 15], gridcolor=GRID)
fig.update_xaxes(title_text="s = ν/q", row=1, col=1, gridcolor=GRID)
fig.update_xaxes(title_text="1/(1 − q/ν) = s/(s − 1)", row=1, col=2, range=[0, 6.5], gridcolor=GRID)
fig.update_layout(template="simple_white", paper_bgcolor=SURF, plot_bgcolor=SURF, font=dict(color=INK, size=13),
    width=1250, height=560, legend=dict(orientation="h", y=-0.2),
    title=dict(text="Feature count depends on the degree ν and the penalty exponent q only through s = ν/q<br>"
        "<sup>Diamonds: exact max-margin count (no training). Discs: trained gated-chain nets, loss-ablation count "
        "(filled p = 59, open p = 71; small = seeds, large = median).</sup>"))
fig.write_html(ROOT + "figs/s_collapse.html", include_plotlyjs="cdn")
try:
    fig.write_image(ROOT + "figs/s_collapse.png", scale=2)
except Exception as ex:
    print("png skipped:", ex)
print("slope c =", round(c, 3), "| cells:", {k: sorted(v) for k, v in grp.items()})
