"""Figures for the paper's experiments section. Every plotted value is read from files in heracles-emp:
  run directories (experiments/E*/runs/*/metrics.json, via experiments/common/alib.py),
  experiments/{E1,E7,E8}/analysis/A1_per_run.csv (Amendment A1, post hoc), experiments/common/A2_pairs.csv and
  A2_summary.json (Amendment A2, post hoc), experiments/E*/predictions.json and experiments/common/pred/switch_p37.json.
Nothing is typed by hand except axis labels. Converged runs (status ok) are filled markers; not-converged runs, where
shown, are hollow.
Writes paper_figs/fig_*.pdf and paper_figs/FIGURE_NUMBERS.json (every number quoted in a caption).
Usage: uv run python paper_figs/make_figs.py
"""
import csv, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "experiments", "common")); sys.path.insert(0, ROOT)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import alib

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "lines.linewidth": 1.1,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
})
W = 5.5  # ICLR text width, inches
NU_COL = {3: "#0072B2", 4: "#D55E00", 5: "#009E73"}          # Okabe-Ito
S_COL = {1.0: "#000000", 1.25: "#0072B2", 1.5: "#D55E00", 2.0: "#009E73", 2.5: "#CC79A7"}
COUNT = "thr_20"  # registered definition (A_j >= 20% of max A); its choice as headline is post hoc (AUDIT)
NUM = {}


def a1(exps):
    rows = []
    for e in exps:
        for r in csv.DictReader(open(os.path.join(ROOT, "experiments", e, "analysis", "A1_per_run.csv"))):
            def num(v):
                try:
                    return float(v)
                except ValueError:
                    return v
            r = {k: (v if k in ("exp", "run_id", "status", "model", "label") else num(v)) for k, v in r.items()}
            rows.append(r)
    return rows


def preds(exp):
    return json.load(open(os.path.join(ROOT, "experiments", exp, "predictions.json")))["cells"]


def jitter(n, w=0.012, seed=0):
    return np.random.default_rng(seed).uniform(-w, w, n)


# ---------------------------------------------------------------- Fig 1: count vs the exponent s
P_COL = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]


def mini_sweep(ax, nu, p, R, C, show_y, show_x):
    """One (nu, p) sweep: K*(s) in black, every converged seed as a faint dot, their median in colour."""
    ss = sorted(s for q, s in C if q == p); X = {s: i for i, s in enumerate(ss)}  # equally spaced s values
    ax.plot(range(len(ss)), [C[(p, s)] for s in ss], color="k", lw=1.0, marker="s", ms=3, zorder=2)
    sm = [s for s in ss if any(r["p"] == p and r["s"] == s for r in R)]
    for s in sm:
        v = np.array([r[COUNT] for r in R if r["p"] == p and r["s"] == s], float)
        ax.scatter(X[s] + jitter(len(v), 0.08, int(100 * s) + p), v + jitter(len(v), 0.08, p), s=6, color=NU_COL[nu], alpha=0.35, lw=0, zorder=3)
    ax.plot([X[s] for s in sm], [np.median([r[COUNT] for r in R if r["p"] == p and r["s"] == s]) for s in sm], color=NU_COL[nu], lw=1.2, marker="o", ms=3.2, zorder=4)
    ax.set_xticks(range(len(ss))); ax.set_xticklabels([f"{x:g}" for x in ss] if show_x else [])
    ax.set_xlim(-0.4, len(ss) - 0.6)
    ax.set_title(fr"$\nu={nu}$, $p={p}$", fontsize=7, pad=2)
    ax.tick_params(labelsize=6)
    if not show_y:
        ax.set_yticklabels([])


def fig_exponent():
    fig = plt.figure(figsize=(W, 2.6))
    gs = fig.add_gridspec(2, 5, width_ratios=[2.3, 1, 1, 1, 1], wspace=0.3, hspace=0.62, top=0.84, bottom=0.14, left=0.08, right=0.99)
    a0 = fig.add_subplot(gs[:, 0])
    # (a) E7: nu = 4, p = 37, fine grid in s
    sw = json.load(open(os.path.join(ROOT, "experiments", "common", "pred", "switch_p37.json")))
    for pl in sw["plateaus"]:
        a0.plot([pl["s_from"], pl["s_to"]], [pl["K"]] * 2, color="k", lw=1.3, solid_capstyle="butt", zorder=2)
    for a_, b_ in zip(sw["plateaus"][:-1], sw["plateaus"][1:]):
        a0.plot([a_["s_to"]] * 2, [a_["K"], b_["K"]], color="k", lw=0.5, ls=":", zorder=1)
    R = a1(["E7"])
    for st in ("ok", "not_converged"):
        rr = [r for r in R if r["status"] == st]
        x = np.array([r["s"] for r in rr]); y = np.array([r[COUNT] for r in rr])
        a0.scatter(x + jitter(len(x), 0.006), y, s=14, marker="o", zorder=3,
                   facecolors=NU_COL[4] if st == "ok" else "none", edgecolors=NU_COL[4], linewidths=0.8,
                   label="converged" if st == "ok" else "not converged")
    a0.plot([], [], color="k", lw=1.3, label=r"$K^*(s)$")
    a0.set_xlabel(r"$s=\nu/b$"); a0.set_ylabel("number of frequencies")
    a0.set_title(r"(a) $\nu=4$, $p=37$, fine sweep in $b$", loc="left")
    a0.set_yticks([0, 3, 6, 9, 12, 15, 18]); a0.set_ylim(0, 19.5)
    a0.legend(frameon=False, loc="upper right", handlelength=1.2, fontsize=6.5, borderaxespad=0.1)
    NUM["E7_n_ok"] = sum(r["status"] == "ok" for r in R); NUM["E7_n_nc"] = sum(r["status"] == "not_converged" for r in R)
    # (b) every sweep of the main grid with >= 2 converged s values: nu = 4 (top row), nu = 5 (bottom row)
    Rm = [r for r in a1(["E1"]) if r["status"] == "ok"]
    shown, skipped = [], []
    for row, nu in enumerate((4, 5)):
        C = {(c["p"], c["s"]): c["K_star"] for c in preds("E1") if c.get("nu") == nu}
        R = [r for r in Rm if r["nu"] == nu]
        ps = [p for p in sorted({p for p, _ in C}) if len({r["s"] for r in R if r["p"] == p}) >= 2]
        skipped += [(nu, p) for p in sorted({p for p, _ in C}) if p not in ps]
        ymax = max(max(C.values()), max(r[COUNT] for r in R)) + 0.8
        for k, p in enumerate(ps[:4]):
            ax = fig.add_subplot(gs[row, 1 + k])
            mini_sweep(ax, nu, p, R, C, show_y=(k == 0), show_x=True)
            ax.set_ylim(0, ymax); ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))
            shown.append((nu, p))
    fig.text(0.40, 0.985, r"(b) every other sweep in $b$ (black: $K^*$; colour: median count; dots: seeds)", fontsize=8, va="top")
    fig.text(0.70, 0.0, r"$s=\nu/b$ (equally spaced)", fontsize=7.5, ha="center", va="bottom")
    NUM["fig1_panels_b"] = {"shown": shown, "skipped_fewer_than_2_converged_cells": skipped}
    NUM["fig1_direction_test"] = json.load(open(os.path.join(ROOT, "experiments", "common", "A2_summary.json")))["ok"][COUNT]["T1_qsweep_E1E8"]
    fig.savefig(os.path.join(HERE, "fig_exponent.pdf")); plt.close(fig)


# ---------------------------------------------------------------- Fig 2: count across p at nu = 3
def fig_prime():
    fig, ax = plt.subplots(figsize=(W, 2.1))
    R = [r for r in a1(["E1", "E8"]) if r["nu"] == 3 and r["status"] == "ok"]
    C = {(c["p"], c["s"]): c for c in preds("E8")}
    C.update({(c["p"], c["s"]): c for c in preds("E1") if c.get("nu", 3) == 3})
    A3 = json.load(open(os.path.join(ROOT, "experiments", "common", "A3_bestknown.json")))["cells"]
    for c3 in A3.values():  # Amendment A3: best-known optimum where an independent search found a higher Gamma
        if (c3["p"], c3["s"]) in C and c3["s"] == 1.5:
            C[(c3["p"], c3["s"])] = {**C[(c3["p"], c3["s"])], "K_star": c3["K_best_known"]}
    out = {}
    for s, dx in ((1.25, -0.012), (1.5, 0.012)):
        ps = sorted({int(r["p"]) for r in R if r["s"] == s})
        med, lo, hi, ks, ex = [], [], [], [], []
        for p in ps:
            v = np.array([r[COUNT] for r in R if r["s"] == s and int(r["p"]) == p])
            med.append(np.median(v)); lo.append(v.min()); hi.append(v.max())
            ks.append(C[(p, s)]["K_star"]); ex.append(C[(p, s)]["label"].startswith("exact"))
        ps = np.array(ps, float); med = np.array(med); ks = np.array(ks); ex = np.array(ex)
        xs = ps * (1 + dx)
        ax.plot(ps[ex], ks[ex], color=S_COL[s], lw=1.0, zorder=2, label=fr"$K^*$, $s={s}$")
        ax.plot(ps[ex.sum() - 1:], ks[ex.sum() - 1:], color=S_COL[s], lw=1.0, ls=(0, (3, 2)), zorder=2)
        ax.scatter(ps[ex], ks[ex], color=S_COL[s], marker="_", s=40, zorder=3, lw=1.4)
        ax.scatter(ps[~ex], ks[~ex], color=S_COL[s], marker="_", s=40, zorder=3, lw=1.4, alpha=0.5)
        ax.errorbar(xs, med, yerr=[med - np.array(lo), np.array(hi) - med], fmt="o", ms=3.4, color=S_COL[s], mec="white", mew=0.4,
                    elinewidth=0.7, capsize=0, zorder=4, alpha=0.9, label=fr"trained (median, range), $s={s}$")
        out[str(s)] = {"p": ps.tolist(), "K_star": ks.tolist(), "median": med.tolist(), "exact": ex.tolist()}
    # per-cell agreement of the median count with K* (within 1), and the best constant per s as a baseline
    agree = sum(abs(m - k) <= 1 for o in out.values() for m, k in zip(o["median"], o["K_star"]))
    ncell = sum(len(o["median"]) for o in out.values())
    const = sum(max(sum(abs(m - c) <= 1 for m in o["median"]) for c in range(0, 15)) for o in out.values())
    out["cells_within_1"] = int(agree); out["cells"] = int(ncell); out["best_constant_per_s_within_1"] = int(const)
    ax.set_xscale("log"); ax.set_xticks([11, 17, 23, 37, 59, 113]); ax.set_xticklabels(["11", "17", "23", "37", "59", "113"])
    ax.minorticks_off()
    ax.set_xlabel(r"$p$ (log scale)"); ax.set_ylabel(f"number of frequencies")
    ax.set_title(r"$\nu=3$, $b=2.4$ ($s=1.25$) and $b=2$ ($s=1.5$)", loc="left")
    ax.legend(frameon=False, fontsize=6.5, loc="center left", bbox_to_anchor=(1.01, 0.5), ncol=1, handlelength=1.4)
    ax.set_ylim(0, 14); ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    NUM["fig2"] = out
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_prime.pdf")); plt.close(fig)


# ---------------------------------------------------------------- Fig 3: amplitude spectra, dense to sparse
def fig_spectra():
    """One run per panel, chosen by rule: the converged run with the lowest seed. Bars: A_j / max_k A_k against j."""
    runs = {e: alib.load(e) for e in ("E1", "E3")}
    P1 = {(c["p"], c["s"]): c["K_star"] for c in preds("E1") if c.get("nu", 3) == 3}
    rows = ((1.0, r"$b=3$ ($s=1$)"), (1.25, r"$b=2.4$ ($s=1.25$)"), (1.5, r"$b=2$ ($s=1.5$)"))
    fig, ax = plt.subplots(3, 2, figsize=(W, 3.0), sharey=True, gridspec_kw={"width_ratios": [1, 2.3]})
    out = {}
    for c, p in enumerate((23, 113)):
        for rr_i, (s, lab) in enumerate(rows):
            if s == 1.0:
                cand = [r for r in runs["E3"] if r["status"] == "ok" and r["config"]["model"] == "product" and r["config"]["n"] == 2 and r["config"]["p"] == p]
                kstar = (p - 1) // 2
            else:
                cand = [r for r in runs["E1"] if r["status"] == "ok" and r["config"]["n"] == 2 and r["config"]["p"] == p and abs(3 / r["config"]["q"] - s) < 1e-6]
                kstar = P1[(p, s)]
            r = sorted(cand, key=lambda r: (int(r["config"]["seed"]), r["run_id"]))[0]
            A = np.asarray(r["metrics"]["measurement"]["A"], float); A = A / A.max()
            j = np.arange(1, len(A) + 1); on = A >= 0.2
            a = ax[rr_i, c]
            a.vlines(j, 0, A, color=np.where(on, S_COL[s], "#bbbbbb"), lw=1.6 if p == 23 else 1.0)
            a.axhline(0.2, color="grey", lw=0.5, ls=":")
            a.text(0.995, 1.0, fr"counted {int(on.sum())}, $K^*={kstar}$", transform=a.transAxes, ha="right", va="bottom", fontsize=6.8)
            a.set_xlim(0.3, len(A) + 0.7); a.set_ylim(0, 1.08)
            a.spines["left"].set_bounds(0, 1); a.set_yticks([0, 1])
            if c == 0:
                a.set_ylabel(lab, rotation=0, ha="right", va="center", fontsize=7.5)
            if rr_i < 2:
                a.set_xticklabels([])
            out[f"p{p}_s{s}"] = {"run": r["run_id"], "counted": int(on.sum()), "K_star": kstar}
        ax[0, c].set_title(fr"$p={p}$", loc="left")
        ax[2, c].set_xlabel("frequency $j$")
    NUM["fig3_runs"] = out
    fig.tight_layout(h_pad=0.3, w_pad=0.8); fig.savefig(os.path.join(HERE, "fig_spectra.pdf")); plt.close(fig)


# ---------------------------------------------------------------- Fig 4: ablation bound and programme efficiency
def fig_ablation_eff():
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.2), gridspec_kw={"width_ratios": [1, 1.25]})
    # (a) measured relative margin loss vs the bound 1 - (1 - w_j)^s, one point per (run, frequency) with w_j >= 0.01
    runs = [r for r in alib.load("E1") if r["status"] == "ok"]
    X, Y = [], []
    for r in runs:
        m = r["metrics"]["measurement"]
        if "ablation" not in m or m["margin"] <= 0:
            continue
        w = np.asarray(m["budget_shares"], float); rl = np.asarray(m["ablation"]["rel_loss"], float); sel = w >= 0.01
        X += (1 - (1 - w[sel]) ** m["s"]).tolist(); Y += rl[sel].tolist()
    X, Y = np.array(X), np.array(Y); ok = Y >= X; Yc = np.minimum(Y, 1.0)
    ax[0].fill_between([0, 1], [0, 1], [1, 1], color="#0072B2", alpha=0.07, lw=0)
    ax[0].plot([0, 1], [0, 1], color="k", lw=0.8)
    ax[0].scatter(X[ok], Yc[ok], s=5, color="#0072B2", alpha=0.45, lw=0)
    ax[0].scatter(X[~ok], Yc[~ok], s=16, color="#D55E00", lw=0, zorder=3)
    ax[0].set_xlim(0, 1); ax[0].set_ylim(0, 1.02); ax[0].set_aspect("equal")
    ax[0].set_xlabel(r"predicted minimum loss $1-(1-w_j)^s$"); ax[0].set_ylabel("measured loss (capped at 1)")
    ax[0].set_title("(a) deleting one frequency", loc="left")
    ax[0].text(0.97, 0.06, f"{int(ok.sum())} of {len(X)} on or\nabove the line", transform=ax[0].transAxes, ha="right", va="bottom", fontsize=7)
    NUM["fig4a"] = {"pairs": int(len(X)), "above": int(ok.sum()), "capped_at_1": int((Y > 1).sum())}
    # (b) programme efficiency: trained runs vs the per-cell nulls and the positive control
    R = [r for r in a1(["E1"]) if r["status"] == "ok"]
    eff = np.array([r["eff"] for r in R])
    N = json.load(open(os.path.join(ROOT, "experiments", "E1", "analysis", "A1_summary.json")))["nulls_and_positive"]
    groups = [("constructed optimum", np.array([v["positive_control"] for v in N.values()]), "#000000"),
              ("trained networks", eff, "#0072B2"),
              ("network at initialization", np.array([v["random_init_net"] for v in N.values() if v.get("random_init_net") is not None]), "#999999"),
              ("random allocation", np.array([v["random_allocation_median"] for v in N.values()]), "#999999")]
    for k, (lab, v, col) in enumerate(groups):
        y = len(groups) - 1 - k
        ax[1].scatter(v, y + jitter(len(v), 0.13, k), s=6, color=col, alpha=0.6, lw=0)
        ax[1].plot([np.median(v)] * 2, [y - 0.3, y + 0.3], color="k", lw=1.4)
        ax[1].text(np.median(v), y + 0.33, f"{np.median(v):.2f}", ha="center", va="bottom", fontsize=6.5)
    ax[1].set_yticks(range(len(groups))); ax[1].set_yticklabels([g[0] for g in groups][::-1])
    ax[1].set_xlim(0, 1.05); ax[1].set_ylim(-0.6, len(groups) - 0.2)
    ax[1].set_xlabel(r"worst-offset coverage / optimum $\Gamma_s(p)$")
    ax[1].set_title("(b) how good is the learned allocation", loc="left")
    ax[1].spines["left"].set_visible(False); ax[1].tick_params(axis="y", length=0)
    NUM["fig4b"] = {g[0]: {"n": int(len(g[1])), "median": float(np.median(g[1]))} for g in groups}
    fig.tight_layout(w_pad=1.2); fig.savefig(os.path.join(HERE, "fig_ablation_eff.pdf")); plt.close(fig)


if __name__ == "__main__":
    fig_exponent(); fig_prime(); fig_spectra(); fig_ablation_eff()
    json.dump(NUM, open(os.path.join(HERE, "FIGURE_NUMBERS.json"), "w"), indent=1, default=float)
    print(json.dumps(NUM, indent=1, default=float)[:3000])
