"""E1 + E1b analysis (light, local). Writes analysis/per_run.csv, cells.csv, summary.json, tables.md.
Usage: python experiments/E1/analysis/analyze_E1.py"""
import itertools, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(ROOT, "experiments", "common")); sys.path.insert(0, ROOT)
import numpy as np
import alib
from core import metrics as MT, measure as MS

runs = alib.load("E1")
idx, P = alib.pred_index("E1")
rows, scored = [], []
for r in runs:
    if r["status"] in ("crashed", "diverged") or "measurement" not in r.get("metrics", {}):
        rows.append({"run_id": r["run_id"], "status": r["status"]}); continue
    c = r["config"]; key = ("product", c["n"] + 1, c["p"], round((c["n"] + 1) / c["q"], 6))
    cell = idx[key]; sc = alib.score(r, cell)
    row = alib.flat_row(r, cell, sc); rows.append(row); scored.append((r, cell, sc, row))
alib.write_csv(rows, os.path.join(HERE, "per_run.csv"))

def usable(row, which):
    if which == "ok":
        return row["status"] == "ok"
    return row["status"] == "not_converged" and row["margin"] > 0

summary = {"script": "experiments/E1/analysis/analyze_E1.py", "n_runs_total": len(runs),
           "status_counts": {s: sum(r["status"] == s for r in runs) for s in sorted({r["status"] for r in runs})}}
COUNT_KEYS = MS.COUNT_KEYS
# ---- per cell
cells = []
for key in sorted(idx):
    _, nu, p, s = key
    rs = [row for (_, cell, sc, row) in scored if (row["nu"], row["p"], row["s"]) == (nu, p, s)]
    allr = [r for r in runs if r.get("config") and (r["config"]["n"] + 1, r["config"]["p"], round((r["config"]["n"] + 1) / r["config"]["q"], 6)) == (nu, p, s)]
    ok = [row for row in rs if usable(row, "ok")]
    nc = [row for row in rs if usable(row, "nc")]
    cellp = idx[key]
    d = {"nu": nu, "p": p, "s": s, "q": round(nu / s, 4), "K_star": cellp["K_star"], "label": cellp["label"],
         "runner_up_rel_gap": (cellp.get("runner_up") or {}).get("rel_gap"), "n_launched": len(allr), "n_ok": len(ok),
         "n_not_converged_pos_margin": len(nc), "n_other": len(allr) - len(ok) - len(nc)}
    for grp, name in ((ok, "ok"), (nc, "nc")):
        if not grp:
            continue
        med = lambda k: float(np.median([g[k] for g in grp]))
        d[f"{name}_median_thr10"] = med("cnt_thr_10")
        d[f"{name}_frac_thr10_exact"] = float(np.mean([g["cnt_thr_10"] == g["K_star"] for g in grp]))
        d[f"{name}_frac_thr10_within1"] = float(np.mean([abs(g["cnt_thr_10"] - g["K_star"]) <= 1 for g in grp]))
        for k in ("share_tv", "amp_cosine", "amp_r2_identity", "amp_r2_linear_fit", "amp_pearson", "amp_spearman", "share_pearson",
                  "share_spearman", "share_r2_identity", "jaccard_thr_10", "gamma_over_universal_product", "gamma_over_class_IF",
                  "energy_junk", "energy_offdiag"):
            d[f"{name}_median_{k}"] = med(k)
        d[f"{name}_counts_all_defs"] = {k: [g[f"cnt_{k}"] for g in grp] for k in COUNT_KEYS}
        d[f"{name}_run_match"] = [bool(g["cnt_thr_10"] == g["K_star"] and g["share_tv"] <= 0.10 and g["amp_cosine"] >= 0.95) for g in grp]
    if ok:
        d["supported"] = bool(d["ok_median_thr10"] == d["K_star"] and d["ok_median_share_tv"] <= 0.10 and d["ok_median_amp_cosine"] >= 0.95)
        seeds = [[g["share_tv"]] for g in ok]
        d["ok_tv_median_ci"] = alib.boot_ci(seeds)[1]
        d["ok_cos_median_ci"] = alib.boot_ci([[g["amp_cosine"]] for g in ok])[1]
    else:
        d["supported"] = None
    cells.append(d)
alib.write_csv([{k: v for k, v in c.items() if not isinstance(v, (list, dict))} for c in cells], os.path.join(HERE, "cells.csv"))
summary["cells"] = cells
# ---- sweep level: every count definition vs K*
for name in ("ok", "nc"):
    grp = [row for (_, _, _, row) in scored if usable(row, name)]
    if not grp:
        continue
    sw = {}
    for k in COUNT_KEYS:
        sw[k] = MT.sweep_summary([g[f"cnt_{k}"] for g in grp], [g["K_star"] for g in grp])
        sw[k + "_vs_same_def"] = MT.sweep_summary([g[f"cnt_{k}"] for g in grp], [g[f"cnt_{k}_pred"] for g in grp])
    for nu in (3, 4, 5):
        gg = [g for g in grp if g["nu"] == nu]
        if gg:
            sw[f"thr_10_nu{nu}"] = MT.sweep_summary([g["cnt_thr_10"] for g in gg], [g["K_star"] for g in gg])
    agg = {}
    for k in ("share_tv", "amp_cosine", "amp_r2_identity", "amp_r2_linear_fit", "amp_pearson", "amp_spearman", "share_pearson",
              "share_spearman", "share_cosine", "share_r2_identity", "share_r2_linear_fit", "share_max_abs_error", "amp_max_abs_error",
              "jaccard_thr_1", "jaccard_thr_5", "jaccard_thr_10", "jaccard_thr_20", "gamma_over_universal_product", "gamma_over_class_IF"):
        v = np.array([g[k] for g in grp], float)
        agg[k] = {"median": float(np.nanmedian(v)), "mean": float(np.nanmean(v)), "q10": float(np.nanpercentile(v, 10)),
                  "q90": float(np.nanpercentile(v, 90)), "n": int(np.isfinite(v).sum())}
    agg["frac_run_match"] = float(np.mean([g["cnt_thr_10"] == g["K_star"] and g["share_tv"] <= 0.10 and g["amp_cosine"] >= 0.95 for g in grp]))
    agg["max_gamma_over_universal"] = float(max(g["gamma_over_universal_product"] for g in grp))
    summary[f"sweep_{name}"] = {"n": len(grp), "counts": sw, "per_run_metric_distribution": agg}
# ---- E1b: equal s across nu
def share_sorted(run, h):
    w = np.sort(np.asarray(run["metrics"]["measurement"]["budget_shares"]))[::-1]
    return w
okruns = [(r, row) for (r, _, _, row) in scored if usable(row, "ok")]
e1b = []
for p in sorted({row["p"] for _, row in okruns}):
    for s in sorted({row["s"] for _, row in okruns}):
        by_nu = {}
        for r, row in okruns:
            if row["p"] == p and row["s"] == s:
                by_nu.setdefault(row["nu"], []).append((share_sorted(r, 0), row["cnt_thr_10"]))
        if len(by_nu) < 2:
            continue
        within_s = [MT.tv_distance(a[0], b[0]) for n1, n2 in itertools.combinations(by_nu, 2) for a in by_nu[n1] for b in by_nu[n2]]
        within_s_K = [abs(a[1] - b[1]) for n1, n2 in itertools.combinations(by_nu, 2) for a in by_nu[n1] for b in by_nu[n2]]
        between = []; between_K = []; noise = []
        for nu in by_nu:
            others = [(r, row) for r, row in okruns if row["p"] == p and row["nu"] == nu and row["s"] != s]
            for a in by_nu[nu]:
                for r2, row2 in others:
                    between.append(MT.tv_distance(a[0], share_sorted(r2, 0))); between_K.append(abs(a[1] - row2["cnt_thr_10"]))
            noise += [MT.tv_distance(a[0], b[0]) for a, b in itertools.combinations(by_nu[nu], 2)]
        e1b.append({"p": p, "s": s, "nus": sorted(by_nu), "n_per_nu": {int(k): len(v) for k, v in by_nu.items()},
                    "within_s_tv": float(np.mean(within_s)), "between_s_tv": float(np.mean(between)) if between else None,
                    "within_cell_tv": float(np.mean(noise)) if noise else None,
                    "ratio": float(np.mean(within_s) / np.mean(between)) if between else None,
                    "within_s_absdK": float(np.mean(within_s_K)), "between_s_absdK": float(np.mean(between_K)) if between_K else None,
                    "supported": bool(np.mean(within_s) / np.mean(between) < 0.5) if between else None})
summary["E1b"] = e1b
# ---- controls
cf = os.path.join(HERE, "controls.json")
if os.path.exists(cf):
    C = json.load(open(cf))
    pc = []
    for c in C["cells"]:
        sp, sn = c["positive"]["score"], c["null_init"]["score"]
        pc.append({"nu": c["cell"]["nu"], "p": c["cell"]["p"], "s": c["cell"]["s"], "K_star": c["cell"]["K_star"],
                   "pos_thr10": sp["counts"]["thr_10"]["meas"], "pos_tv": sp["share"]["tv"], "pos_cos": sp["amp"]["cosine"],
                   "pos_gamma_over_IF": sp["gamma_over_ceiling"]["class_IF"],
                   "null_init_thr10": sn["counts"]["thr_10"]["meas"], "null_init_tv": sn["share"]["tv"], "null_init_cos": sn["amp"]["cosine"],
                   "null_rand_tv_median": c["null_random_allocation"]["tv"]["median"], "null_rand_cos_median": c["null_random_allocation"]["cos_amp"]["median"],
                   "null_rand_thr10_exact_frac": c["null_random_allocation"]["thr10_exact"]["mean"]})
    summary["controls"] = pc
    summary["controls_all_positive_perfect"] = bool(all(x["pos_thr10"] == x["K_star"] and x["pos_tv"] < 1e-6 and abs(x["pos_gamma_over_IF"] - 1) < 1e-6 for x in pc))
json.dump(summary, open(os.path.join(HERE, "summary.json"), "w"), indent=1, default=float)
# ---- console tables
print("status:", summary["status_counts"])
print("nu p s K* label gap | n_ok n_nc other | med_thr10 fracExact fracPm1 medTV medCos medR2id medR2fit jac10 g/univ g/IF | supported")
for d in cells:
    if d["n_ok"] or d["n_not_converged_pos_margin"]:
        g = "ok" if d["n_ok"] else "nc"
        print(d["nu"], d["p"], d["s"], d["K_star"], d["label"][:8], alib.fmt(d["runner_up_rel_gap"], 2), "|", d["n_ok"], d["n_not_converged_pos_margin"], d["n_other"], "|",
              f"[{g}]", alib.fmt(d[f"{g}_median_thr10"]), alib.fmt(d[f"{g}_frac_thr10_exact"], 2), alib.fmt(d[f"{g}_frac_thr10_within1"], 2),
              alib.fmt(d[f"{g}_median_share_tv"]), alib.fmt(d[f"{g}_median_amp_cosine"]), alib.fmt(d[f"{g}_median_amp_r2_identity"]),
              alib.fmt(d[f"{g}_median_amp_r2_linear_fit"]), alib.fmt(d[f"{g}_median_jaccard_thr_10"]), alib.fmt(d[f"{g}_median_gamma_over_universal_product"]),
              alib.fmt(d[f"{g}_median_gamma_over_class_IF"]), "|", d["supported"])
for name in ("ok", "nc"):
    if f"sweep_{name}" in summary:
        sw = summary[f"sweep_{name}"]
        print(f"sweep[{name}] n={sw['n']}", {k: (alib.fmt(v['frac_exact'], 2), alib.fmt(v['frac_within_tol'], 2), alib.fmt(v['r2_identity'], 2), alib.fmt(v['spearman'], 2)) for k, v in sw["counts"].items() if "same" not in k})
        print("  dist:", {k: alib.fmt(v["median"]) for k, v in sw["per_run_metric_distribution"].items() if isinstance(v, dict)}, "frac_run_match", alib.fmt(sw["per_run_metric_distribution"]["frac_run_match"]))
for e in e1b:
    print("E1b", e)
