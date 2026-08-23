"""Plotly figures + table from notes/wspace_local.json -> notes/wspace_figs.html"""
import json, numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
runs = [r for r in json.load(open(ROOT / "notes" / "wspace_local.json")) if r["task"] == "add113"]
KEYS = [("diag_comm", "committee diag (logit)"), ("diag_junk", "other-freq diag (logit) = r's junk"),
        ("off2d", "off-diag 2D = memorizer (logit)"), ("wE_junk", "W_E off-committee (weight)"),
        ("wU_junk", "W_U off-committee (weight)"), ("pnorm", "total ||θ||²")]
grp = lambda wd: "wd<1" if wd < 1 else ("wd=1" if wd == 1 else "wd>1")
groups = [g for g in ("wd<1", "wd=1", "wd>1") if any(grp(r["wd"]) == g for r in runs)]
W = lambda r: [x for x in r["rows"] if x["in_wait"]]
tfrac = lambda r, w: (np.array([x["epoch"] for x in w]) - r["t_mem"]) / (r["t_gen"] - r["t_mem"])

summary, lines = {}, []
hdr = f"{'group':6s} {'n':>3s} " + " ".join(f"{k:>10s}" for k, _ in KEYS) + "   excl≥2x  restr≤½"
lines.append(hdr)
for g in groups:
    rs = [r for r in runs if grp(r["wd"]) == g]
    up = {k: float(np.mean([W(r)[-1][k] > W(r)[0][k] for r in rs])) for k, _ in KEYS}
    ex = float(np.mean([W(r)[-1]["loss_excl"] > 2 * W(r)[0]["loss_excl"] for r in rs]))
    rd = float(np.mean([W(r)[-1]["loss_restr"] < 0.5 * W(r)[0]["loss_restr"] for r in rs]))
    summary[g] = dict(n=len(rs), frac_up=up, excl_2x=ex, restr_half=rd,
                      wds=sorted({r["wd"] for r in rs}), cohorts=sorted({r["cohort"] for r in rs}))
    lines.append(f"{g:6s} {len(rs):3d} " + " ".join(f"{up[k]:10.2f}" for k, _ in KEYS) + f"   {ex:.2f}     {rd:.2f}")
print("\n".join(lines))
json.dump(summary, open(ROOT / "notes" / "wspace_summary.json", "w"), indent=1)

fig = make_subplots(rows=len(groups), cols=len(KEYS),
                    subplot_titles=[t for _, t in KEYS] * len(groups),
                    shared_xaxes=True, vertical_spacing=0.08, horizontal_spacing=0.035)
for i, g in enumerate(groups, 1):
    rs = [r for r in runs if grp(r["wd"]) == g]
    for j, (k, _) in enumerate(KEYS, 1):
        for r in rs:
            w = W(r); y = np.log10(np.array([x[k] for x in w]) / max(w[0][k], 1e-12))
            fig.add_trace(go.Scatter(x=tfrac(r, w), y=y, mode="lines", showlegend=False,
                                     line=dict(width=1, color="#1f77b4" if y[-1] > 0 else "#d62728"),
                                     opacity=.55, hovertext=f"{r['run']} wd={r['wd']}"), row=i, col=j)
        fig.add_hline(y=0, line=dict(color="black", width=.6), row=i, col=j)
        if j == 1:
            fig.update_yaxes(title_text=f"{g} (n={len(rs)})<br>log10 rel. to wait start", row=i, col=j)
fig.update_xaxes(title_text="wait fraction t_mem→t_gen", row=len(groups))
fig.update_layout(height=300 * len(groups) + 80, width=330 * len(KEYS),
                  title="Absolute energy through the delay — blue ends higher, red ends lower (add113, checkpointed cohorts, final committee)")

fig2 = make_subplots(rows=1, cols=len(groups), shared_yaxes=True,
                     subplot_titles=[f"{g} (n={summary[g]['n']})" for g in groups])
for i, g in enumerate(groups, 1):
    for r in [r for r in runs if grp(r["wd"]) == g]:
        w = W(r); t = tfrac(r, w)
        for key, col in (("loss_excl", "#d62728"), ("loss_restr", "#2ca02c"), ("loss_full", "#555")):
            fig2.add_trace(go.Scatter(x=t, y=[x[key] for x in w], mode="lines", showlegend=False,
                                      line=dict(width=1, color=col), opacity=.5,
                                      hovertext=f"{r['run']} {key}"), row=1, col=i)
fig2.update_yaxes(type="log", title_text="train CE — red excluded, green restricted, grey full", row=1, col=1)
fig2.update_xaxes(title_text="wait fraction")
fig2.update_layout(height=420, width=420 * len(groups),
                   title="Nanda excluded / restricted loss through the delay (final committee)")

# fraction-up bars
fig3 = go.Figure()
for g in groups:
    fig3.add_trace(go.Bar(name=f"{g} (n={summary[g]['n']})", x=[t for _, t in KEYS],
                          y=[summary[g]["frac_up"][k] for k, _ in KEYS]))
fig3.add_hline(y=0.5, line=dict(dash="dash"))
fig3.update_layout(barmode="group", height=380, width=900, yaxis_title="fraction of waits ending HIGHER",
                   title="Does it grow during the wait? (deletion predicts <0.5 for everything but committee)")

html = "<h2>Deletion vs outcompetition — existing checkpointed waits</h2><pre>" + "\n".join(lines) + "</pre>"
html += fig3.to_html(full_html=False, include_plotlyjs="cdn")
html += fig.to_html(full_html=False, include_plotlyjs=False)
html += fig2.to_html(full_html=False, include_plotlyjs=False)
(ROOT / "notes" / "wspace_figs.html").write_text(html)
print("wrote notes/wspace_figs.html")
