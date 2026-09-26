"""ladder_report.py -- tables, ceilings.json (for the pod) and plotly figures from the exact ladders.

Reads FINAL_RUNS/LADDER/json/cells.json (ladder_exact.py) and prime_axis.json (ladder_prime.py).
Writes FINAL_RUNS/LADDER/tables.md, .../json/ceilings.json, .../figs/*.html (+ .png if kaleido).

Usage: uv run python heracles/ladder_report.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from ladder_core import OUT  # noqa: E402

FIG = OUT / "figs"
LAM_MAIN = {2.0: "0.0001", 2.5: "1e-06", 8 / 3: "1e-06", 3.0: "1e-06", 10 / 3: "1e-06", 4.0: "1e-06", 5.0: "1e-06"}
# the lambda each cell's archived training actually used (RESULTS_PROD note 4): 1e-4/1e-5 at q = 2, 1e-6 for q > 2
LAM_ARCH = {"q2_n2": "0.0001", "q2_n3": "1e-05", "q2_n4": "1e-05"}


ACT_TOL = 1e-3      # a share counts as ACTIVE if b_k > ACT_TOL (0.1% of the l_q budget)
VAL_TOL = 1e-3      # a rung REACHES the optimum if its value >= (1 - VAL_TOL) * Gamma_s


def n_active(b, tol=ACT_TOL):
    return int((np.asarray(b) > tol).sum())


def K_star_of(cell, act_tol=ACT_TOL, val_tol=VAL_TOL):
    """K*_free, defined so that it does not depend on which rung the search happened to solve best.

    Gamma_s = the largest value found anywhere on the free ladder.  K*_free = the SMALLEST number of
    active shares (b_k > act_tol) among all rungs whose value is within val_tol of Gamma_s.
    Returns (K*_free, the rung it came from, Gamma_s).

    Why this and not "argmax over K": the free ladder is non-decreasing in K (a K-support contains
    every smaller support, so a rung can always copy a lower rung's solution and set the extra shares
    to zero).  Its argmax is therefore the LAST rung up to search noise; the informative number is
    where it first reaches the top, and how many frequencies that solution actually uses."""
    vals = [r["free"]["val"] for r in cell["rows"]]
    best = max(vals)
    cand = [(n_active(cell["rows"][i]["free"]["b"], act_tol), i)
            for i, v in enumerate(vals) if v >= best * (1 - val_tol)]
    k, i = min(cand)
    return k, i + 1, best


def fmt(x, d=4):
    return "—" if x is None else f"{x:.{d}f}"


def main():
    cells = json.loads((OUT / "json" / "cells.json").read_text())
    FIG.mkdir(parents=True, exist_ok=True)
    lines = []

    # ---------------------------------------------------------------- T1 headline
    lines += ["## T1 — headline: one row per cell\n",
              f"`K*_free` = size of the support of the FREE-amplitude programme's maximiser (share threshold "
              f"{ACT_TOL:g}, value tolerance {VAL_TOL:g}); `K*_eq` = argmax of the EQUAL-amplitude ladder "
              "`M_K K^-s` (what `kstar.py` computes). These are different objects and are flagged when they "
              "differ. `gamma_free/gamma_eq` compares the two constructed nets' normalized margins. "
              "`argmin obj*` = the K whose constructed net reaches the lowest training objective at the "
              "lambda the archived runs used; `dense/best` = the all-frequency equal-share net's normalized "
              "margin over the best sparse rung's.\n",
              "| family | p | nu | q | s=nu/q | K*_free | K*_eq | same? | gamma_free/gamma_eq | argmin obj* (lam) | dense/best gamma | dense/best obj* |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    ceil = {}
    for key, c in sorted(cells.items()):
        kact, kfirst, gbest = K_star_of(c)
        gam_eq = [r["eq"]["gamma"] for r in c["rows"]]
        lam = LAM_ARCH.get(f"q{int(c['q'])}_n{c['n']}") if abs(c["q"] - 2) < 1e-9 else "1e-06"
        amin = c["argmin_obj"][lam]
        objs = [r["eq"]["obj"][lam] for r in c["rows"]]
        dr_obj = c["dense"]["obj"][lam] / min(objs)
        keq = c["argmax_gamma_eq"]
        gfree = max(r["free"]["gamma"] for r in c["rows"])
        geq = max(gam_eq)
        same = "yes" if kact == keq else "**NO**"
        lines.append(f"| {c['family']} | {c['p']} | {c['nu']} | {c['q']:.4g} | {c['s']:.3f} | **{kact}** | "
                     f"{keq} | {same} | {gfree / geq:.4f} | {amin} (lam {lam}) | {c['dense_ratio_eq']:.4f} | "
                     f"{dr_obj:.6f} |")
        # ceiling for the trained-run scorer: the exact block on the programme's global optimum
        ceil[key] = dict(gamma_ceiling=max(r["free"]["gamma"] for r in c["rows"]), K_star=kact,
                         Gamma=gbest, S=c["rows"][kfirst - 1]["free"]["S"], b=c["rows"][kfirst - 1]["free"]["b"],
                         gamma_eq_best=max(gam_eq), argmax_eq=c["argmax_gamma_eq"],
                         dense_gamma=c["dense"]["gamma"])
    (OUT / "json" / "ceilings.json").write_text(json.dumps(ceil, indent=1))

    # ---------------------------------------------------------------- T2 K-band
    lines += ["\n## T2 — how sharply is K* resolved? (equal-share ladder, relative to its peak)\n",
              "`gamma(K)/gamma(K*)` for the rungs around the peak. A cell where the neighbours sit within the "
              "theorem's slack `F` is resolved only to +-1.\n",
              "| family | p | nu | q | K*_eq | K*-2 | K*-1 | **K*** | K*+1 | K*+2 | K*+3 |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for key, c in sorted(cells.items()):
        gam = [r["eq"]["gamma"] for r in c["rows"]]
        ks = c["argmax_gamma_eq"]
        row = []
        for d in (-2, -1, 0, 1, 2, 3):
            K = ks + d
            row.append("—" if K < 1 or K > len(gam) else f"{gam[K - 1] / gam[ks - 1]:.4f}")
        lines.append(f"| {c['family']} | {c['p']} | {c['nu']} | {c['q']:.4g} | {ks} | " + " | ".join(row) + " |")

    # ---------------------------------------------------------------- T3 lambda stability
    lines += ["\n## T3 — lambda-stability of argmin obj* (equal-share ladder)\n",
              "| family | p | nu | q | " + " | ".join(f"lam {l}" for l in cells[list(cells)[0]]["rows"][0]["eq"]["obj"]) + " |",
              "|---" * (4 + len(cells[list(cells)[0]]["rows"][0]["eq"]["obj"])) + "|"]
    for key, c in sorted(cells.items()):
        lines.append(f"| {c['family']} | {c['p']} | {c['nu']} | {c['q']:.4g} | "
                     + " | ".join(str(c["argmin_obj"][l]) for l in c["rows"][0]["eq"]["obj"]) + " |")

    # ---------------------------------------------------------------- T4 dense flat
    lines += ["\n## T4 — the dense flat competitor (all (p-1)/2 frequencies, equal shares)\n",
              "| family | p | nu | q | s | gamma_dense / gamma_best | verdict |",
              "|---|---|---|---|---|---|---|"]
    for key, c in sorted(cells.items()):
        r = c["dense_ratio_eq"]
        v = "dense WINS" if r > 1.0001 else ("tie" if r > 0.9999 else "sparse wins")
        lines.append(f"| {c['family']} | {c['p']} | {c['nu']} | {c['q']:.4g} | {c['s']:.3f} | {r:.4f} | {v} |")

    # ---------------------------------------------------------------- T5 threshold sensitivity
    lines += ["\n## T5 — how much does K*_free depend on the thresholds?\n",
              "`K*_free` recomputed with the active-share threshold `b_k > t_a` and the value tolerance "
              "`value >= (1-t_v) Gamma`.\n",
              "| family | p | nu | q | t_a=1e-2 | t_a=1e-3 | t_a=1e-4 | t_v=1e-2 | t_v=1e-3 | t_v=1e-4 | K*_eq |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for key, c in sorted(cells.items()):
        r = [K_star_of(c, act_tol=t)[0] for t in (1e-2, 1e-3, 1e-4)]
        r2 = [K_star_of(c, val_tol=t)[0] for t in (1e-2, 1e-3, 1e-4)]
        lines.append(f"| {c['family']} | {c['p']} | {c['nu']} | {c['q']:.4g} | " + " | ".join(map(str, r + r2))
                     + f" | {c['argmax_gamma_eq']} |")

    # ---------------------------------------------------------------- T6 prime axis, both K*
    for tag, fn in (("small primes, all six s", "prime_axis_small.json"),
                    ("large primes, s = 3/2, 2, 5/2", "prime_axis_big.json")):
        pth = OUT / "json" / fn
        if not pth.exists():
            continue
        pa = json.loads(pth.read_text())
        ss = sorted({float(s) for row in pa.values() for s in row["per_s"]})
        lines += [f"\n## T6 — prime axis ({tag}): K*_free / K*_eq\n",
                  "| p | " + " | ".join(f"s={s:.3f}" for s in ss) + " |",
                  "|---" * (1 + len(ss)) + "|"]
        for pk in sorted(pa, key=int):
            row = pa[pk]
            cells_ = []
            for s in ss:
                d = row["per_s"].get(f"{s:.6f}")
                if d is None:
                    cells_.append("—")
                    continue
                kf = min([int(K) for K, v in d["free_values"].items()
                          if v >= d["Gamma"] * (1 - VAL_TOL)] or [0])
                kfree = d["free_active"][str(kf)] if kf else d["K_star_free"]
                cells_.append(f"{kfree}/{d['K_star_eq']}" + ("" if kfree == d["K_star_eq"] else " ✗"))
            lines.append(f"| {pk} | " + " | ".join(cells_) + " |")

    # ---------------------------------------------------------------- T7 solver cross-check
    gp = OUT / "json" / "gamma_check.json"
    if gp.exists():
        gc = json.loads(gp.read_text())
        lines += ["\n## T7 — is Gamma_s a local-optimiser artefact? (ladder_gamma.py)\n",
                  "SLSQP epigraph (nu4_program) vs an independent projected-supergradient ascent in "
                  "x = b^s, and a branch-and-bound certificate on the same support.\n",
                  "| p | s | K | SLSQP | independent | rel. diff | certified [LB, UB] | gap |",
                  "|---|---|---|---|---|---|---|---|"]
        for r in gc:
            lines.append(f"| {r['p']} | {r['s']:.3f} | {r['K']} | {r['slsqp']:.8f} | {r['independent']:.8f} | "
                         f"{r['rel_disagreement']:.1e} | [{r['bnb_lb']:.6f}, {r['bnb_ub']:.6f}] | "
                         f"{r['bnb_gap'] * 100:.2f}% |")

    (OUT / "tables.md").write_text("\n".join(lines) + "\n")
    print("wrote", OUT / "tables.md", "and", OUT / "json" / "ceilings.json")

    # ---------------------------------------------------------------- figures
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("plotly not installed; skipping figures")
        return

    for fam in ("chain", "xn"):
        for p in sorted({c["p"] for c in cells.values()}):
            sel = {k: c for k, c in cells.items() if c["family"] == fam and c["p"] == p}
            if not sel:
                continue
            fig = make_subplots(rows=1, cols=2,
                                subplot_titles=("gamma_con(K) / max_K  (exact, no training)",
                                                "obj*(K) / min_K obj*  (at the cell's own lambda)"))
            for key, c in sorted(sel.items(), key=lambda kv: (kv[1]["nu"], kv[1]["q"])):
                Ks = [r["K"] for r in c["rows"]]
                gam = np.array([r["eq"]["gamma"] for r in c["rows"]])
                lam = c.get("lam_main", "1e-05")
                obj = np.array([r["eq"]["obj"][lam] for r in c["rows"]])
                obj = obj / obj.min()
                coll = c.get("collapsed", {}).get(lam, False)
                nm = f"nu{c['nu']} q{c['q']:.3g} (s{c['s']:.2f}){' [collapsed]' if coll else ''}"
                fig.add_trace(go.Scatter(x=Ks, y=gam / gam.max(), name=nm, mode="lines+markers"), 1, 1)
                fig.add_trace(go.Scatter(x=Ks, y=obj, name=nm, mode="lines+markers",
                                         showlegend=False), 1, 2)
                ks = c["argmax_gamma_eq"]
                fig.add_trace(go.Scatter(x=[ks], y=[1.0], mode="markers", marker=dict(size=13, symbol="star"),
                                         name=f"K* {nm}", showlegend=False), 1, 1)
            fig.update_xaxes(title_text="K (frequencies)", row=1, col=1)
            fig.update_xaxes(title_text="K (frequencies)", row=1, col=2)
            fig.update_yaxes(title_text="gamma_con(K) / max_K", row=1, col=1)
            fig.update_yaxes(title_text="obj*(K) / min_K obj*", type="log", row=1, col=2)
            fig.update_layout(title=f"Exact ladders — {fam}, p = {p} (share-weighted junk-free blocks, no training)",
                              template="plotly_white", height=460, width=1150)
            f = FIG / f"ladder_{fam}_p{p}.html"
            fig.write_html(str(f))
            try:
                fig.write_image(str(f.with_suffix(".png")), scale=2)
            except Exception:                                            # noqa: BLE001
                pass
    print("wrote ladder figures to", FIG)

    # prime axis
    pa = {}
    for fn in ("prime_axis_big.json", "prime_axis_small.json"):     # small overrides big (denser search)
        q = OUT / "json" / fn
        if q.exists():
            for k, v in json.loads(q.read_text()).items():
                pa.setdefault(k, {}).setdefault("per_s", {})
                pa[k]["per_s"].update(v["per_s"])
    if pa:
        fig = go.Figure()
        for s, tag in [(1.5, "s=3/2 (nu3 q2 / nu4 q8/3 / nu5 q10/3)"), (2.0, "s=2 (nu4 q2)"),
                       (2.5, "s=5/2 (nu5 q2)"), (1.2, "s=6/5 (nu3 q2.5)"), (4 / 3, "s=4/3 (nu4 q3)"),
                       (5 / 3, "s=5/3 (nu5 q3)")]:
            ps, ke, kf = [], [], []
            for pk, row in sorted(pa.items(), key=lambda kv: int(kv[0])):
                d = row["per_s"].get(f"{s:.6f}")
                if d:
                    kk = min([int(K) for K, v in d["free_values"].items()
                              if v >= d["Gamma"] * (1 - VAL_TOL)] or [1])
                    ps.append(int(pk)); ke.append(d["K_star_eq"]); kf.append(d["free_active"][str(kk)])
            fig.add_trace(go.Scatter(x=ps, y=ke, mode="lines+markers", name=tag + " [equal share]"))
            fig.add_trace(go.Scatter(x=ps, y=kf, mode="markers", marker=dict(symbol="x"),
                                     name=tag + " [free amplitude]"))
        fig.update_layout(title="K*(p) from the programme (exact, no training)", xaxis_title="prime p",
                          yaxis_title="K*", xaxis_type="log", template="plotly_white", height=520, width=1000)
        fig.write_html(str(FIG / "prime_axis.html"))
        try:
            fig.write_image(str(FIG / "prime_axis.png"), scale=2)
        except Exception:                                                # noqa: BLE001
            pass
        print("wrote prime-axis figure")


if __name__ == "__main__":
    main()
