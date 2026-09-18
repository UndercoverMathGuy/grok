"""nu4_table.py -- tables for RESULTS_NU4.md.
(1) fixed-support ladders FINAL_RUNS/NU4/fixed_*: measured gamma(K) per lam stage vs predicted M_K / K^{nu/2} (kstar.py).
(2) free-GD runs (FINAL_RUNS/NU4 + the nu>=4 cells of runs/NS): count vs normalised margin, gamma as a fraction of the
    measured fixed-support ceiling; scatter to heracles/nu4_scatter.html (plotly).
Usage: uv run python heracles/nu4_table.py"""
import json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from kstar import ladder, G  # noqa: E402
from nu4_program import solve  # noqa: E402  (scipy: run on the pod)
import math


def bound_const(act, n, p, tied):
    """clean amplitude of one single-frequency neuron of squared norm m is <= c m^{nu/2} (AM-GM over the n+1 factors):
    prod untied c = 2^-n (2/(p nu))^{nu/2}; tied: x n^{n/2}; pow (x^n, untied): x n! (multinomial)."""
    nu = n + 1; c = 2.0 ** -n * (2 / (p * nu)) ** (nu / 2)
    return c * (n ** (n / 2) if tied else 1.0) * (math.factorial(n) if act == "pow" else 1.0)


def gamma_S(p, S, nu):
    g = G(p)[:, :(p - 1) // 2][np.array(S) - 1]; K = len(S); rng = np.random.default_rng(0)
    return max(solve(g, nu / 2, b0)[0] for b0 in [np.full(K, 1 / K)] + [rng.dirichlet(np.full(K, 3.0)) for _ in range(6)])

NU4 = ROOT / "FINAL_RUNS" / "NU4"; NS = ROOT / "runs" / "NS"
ceil = {}                                   # (act, n, p, tied) -> {K: best measured gamma}

print("## 1. Fixed-support ladders\n")
for cell in sorted(NU4.glob("fixed_*")):
    runs = [json.loads(f.read_text()) for f in sorted(cell.glob("K*_seed*/metrics.json"))]
    if not runs:
        continue
    r0 = runs[0]; nu, p = r0["nu"], r0["p"]
    byK = defaultdict(list)
    for r in runs:
        byK[r["K"]].append(r)
    g_all = G(p)
    M = {K: float(g_all[np.array(rs[0]["S"]) - 1].sum(0).min()) for K, rs in byK.items()}     # coverage of the set actually used
    c = bound_const(r0["act"], r0["n"], p, r0["tied"])
    GS = {K: gamma_S(p, rs[0]["S"], nu) for K, rs in byK.items()}
    pj = ROOT / f"nu4_program_p{p}.json"
    Gstar = json.loads(pj.read_text())[str(float(nu))]["full_top"][0]["value"] if pj.exists() else max(GS.values())
    nf = (p - 1) // 2
    GS = {K: (max(v, Gstar) if K >= nf else v) for K, v in GS.items()}           # S = all frequencies: Gamma_S = Gamma*
    n_ = r0["n"]; print(f"bound constant c = {c:.4e}; Gamma* (free-amplitude programme, all frequencies) = {Gstar:.5f}; c Gamma* = {c * Gstar:.4e}; "
                        f"audit conjecture (n+1)^-((nu-2)/2) = {(n_ + 1) ** (-(nu - 2) / 2):.4f}; 2^n cube population = {(2 ** n_) ** (-(nu - 2) / 2):.4f}\n")
    stages = [s["stage"] for s in r0["stages"]]
    pred = {K: M[K] / K ** (nu / 2) for K in byK}; pmax = max(pred.values())
    last = {K: np.array([r["stages"][-1]["gamma"] for r in rs]) for K, rs in byK.items()}
    best = {K: max(max(s["gamma"] for s in r["stages"][1:]) for r in rs) for K, rs in byK.items()}
    ceil[(r0["act"], r0["n"], p, r0["tied"])] = best
    sp = [K for K in last if K <= 12]                       # normalise by the best SPARSE rung (dense rungs may leave their population)
    mmax = max(last[K].mean() for K in sp); bmax = max(best[K] for K in sp)
    print(f"### {cell.name}  (nu {nu}, p {p}; gamma = min-margin / ||theta||^nu, fp64; {len(runs)} runs)\n")
    print("| K | S_K | m | seeds | gamma constr. | " + " | ".join(f"gamma {s}" for s in stages[1:]) + " | spread (last) | measured/max | best/max | predicted M_K/K^(nu/2) /max | free-amplitude Gamma_S /max | gamma / (c Gamma_S) | gamma / (c Gamma*) | margin | clean | N_.05 |")
    print("|---|---|---|---|---|" + "---|" * (len(stages) - 1) + "---|---|---|---|---|---|---|---|---|---|")
    gsmax = max(GS.values())
    for K in sorted(byK):
        rs = byK[K]
        cols = [np.mean([r["stages"][i]["gamma"] for r in rs]) for i in range(len(stages))]
        fin = [r["stages"][-1] for r in rs]
        print(f"| {K} | {rs[0]['S']} | {rs[0]['m']} | {len(rs)} | " + " | ".join(f"{c:.3e}" for c in cols) +
              f" | {last[K].min():.3e}..{last[K].max():.3e} | {last[K].mean() / mmax:.3f} | {best[K] / bmax:.3f} | {pred[K] / pmax:.3f} | {GS[K] / gsmax:.3f} | {best[K] / (c * GS[K]):.4f} | {best[K] / (c * Gstar):.4f} | "
              f"{np.mean([f['margin'] for f in fin]):.2f} | {np.mean([f['clean_frac'] for f in fin]):.3f} | {', '.join(str(f['N_05']) for f in fin)} |")
    km = max(last, key=lambda K: last[K].mean()); kb = max(best, key=best.get); kp = max(pred, key=pred.get)
    print(f"\nargmax: measured (mean, last stage) K = {km}; measured (best) K = {kb}; predicted K = {kp}.  "
          f"Plateau >= 0.95 of max: measured {[K for K in sorted(last) if last[K].mean() / mmax >= 0.95]}, predicted {[K for K in sorted(pred) if pred[K] / pmax >= 0.95]}\n")

print("## 2. Free / polished runs vs the measured ceiling\n")
rows = []
def add(f, src):
    r = json.loads(f.read_text())
    if "nu" not in r or r.get("act") not in ("prod", "pow") or r["nu"] < 4 or r.get("q", 2.0) != 2.0 or r.get("penalty", "l2") != "l2":
        return
    c = ceil.get((r["act"], r["n"], r["p"], False)) or ceil.get((r["act"], r["n"], r["p"], True)) or {}
    cmax = max(c.values()) if c else float("nan")
    base = dict(src=src, cell=f.parent.parent.name, seed=r["seed"], act=r["act"], nu=r["nu"], p=r["p"], tied=r["tied"], lam=r["lam"], m=r["m"], cmax=cmax)
    for t in r.get("trace", []):
        rows.append(base | dict(step=t["step"], final=False, **{k: t[k] for k in ("margin", "acc", "gamma", "clean_frac", "K_2pct", "K_5pct", "N_05")}))
    rows.append(base | dict(step=r["steps"], final=True, **{k: r[k] for k in ("margin", "acc", "gamma", "clean_frac", "K_2pct", "K_5pct", "N_05")}))
for f in sorted(NS.glob("n[34]_p*/seed*/metrics.json")) + sorted(NS.glob("n[34]_*/seed*/metrics.json")):
    if "_st120000" in str(f) or "_from" in str(f):
        add(f, "NS")
for f in sorted(NU4.glob("n*/seed*/metrics.json")):
    if "seed0/" in str(f) or True:
        add(f, "NU4")
seen = set(); uniq = []
for r in rows:
    k = (r["cell"], r["seed"], r["step"])
    if k not in seen:
        seen.add(k); uniq.append(r)
rows = uniq
print("| cell | seed | acc | margin | gamma | gamma / ceiling | N_.05 | K_5pct | K_2pct | clean |")
print("|---|---|---|---|---|---|---|---|---|---|")
for r in rows:
    if r["final"]:
        print(f"| {r['cell']} | {r['seed']} | {r['acc']:.4f} | {r['margin']:.2f} | {r['gamma']:.2e} | {r['gamma'] / r['cmax']:.2f} | {r['N_05']} | {r['K_5pct']} | {r['K_2pct']} | {r['clean_frac']:.2f} |")
for cell in sorted(NU4.glob("polish_*")) + sorted(NU4.glob("free_*")):
    print(f"\n### {cell.name}\n\n| run | stage | acc | margin | gamma | N_.05 | K_5pct | K_2pct | clean |\n|---|---|---|---|---|---|---|---|---|")
    for f in sorted(cell.glob("*/metrics.json")):
        r = json.loads(f.read_text())
        for s in r["stages"]:
            print(f"| {f.parent.name[-40:]} | {s['stage']} | {s['acc']:.4f} | {s['margin']:.2f} | {s['gamma']:.2e} | {s['N_05']} | {s['K_5pct']} | {s['K_2pct']} | {s['clean_frac']:.2f} |")
            rows.append(dict(src=cell.name, cell=cell.name, seed=f.parent.name, act=r["act"], nu=r["nu"], p=r["p"], step=0, final=s is r["stages"][-1],
                             cmax=max(ceil.get((r["act"], r["n"], r["p"], r["tied"]), {0: float("nan")}).values()), **{k: s[k] for k in ("margin", "acc", "gamma", "clean_frac", "K_2pct", "K_5pct", "N_05")}))
json.dump(rows, open(ROOT / "nu4_scatter.json", "w"))
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=1, cols=2, subplot_titles=("N_.05 vs gamma / measured ceiling", "clean fraction vs gamma / ceiling"))
    for (act, nu) in sorted({(r["act"], r["nu"]) for r in rows}):
        sel = [r for r in rows if r["act"] == act and r["nu"] == nu and r["acc"] > 0.99]
        x = [r["gamma"] / r["cmax"] for r in sel]; txt = [f"{r['cell'][:60]} s{r['seed']} step {r['step']}" for r in sel]
        rng = np.random.default_rng(0); jit = rng.uniform(-.15, .15, len(sel))
        fig.add_trace(go.Scatter(x=x, y=[r["N_05"] + j for r, j in zip(sel, jit)], mode="markers", name=f"{act} nu{nu}", text=txt,
                                 marker=dict(symbol=["circle" if r["final"] else "circle-open" for r in sel])), 1, 1)
        fig.add_trace(go.Scatter(x=x, y=[r["clean_frac"] for r in sel], mode="markers", name=f"{act} nu{nu}", text=txt, showlegend=False), 1, 2)
    fig.update_xaxes(title="gamma / best measured fixed-support gamma"); fig.update_yaxes(title="N_.05", row=1, col=1); fig.update_yaxes(title="clean fraction", row=1, col=2)
    fig.write_html(str(ROOT / "nu4_scatter.html"), include_plotlyjs="cdn")
except Exception as e:                                            # noqa: BLE001
    print("plot skipped:", e, file=sys.stderr)
