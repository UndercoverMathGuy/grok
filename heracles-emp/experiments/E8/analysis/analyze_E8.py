"""E8 analysis (light, local): K across p at fixed s (nu = 3), E8 runs + E1 nu = 3 runs. Usage: python experiments/E8/analysis/analyze_E8.py"""
import json, math, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(ROOT, "experiments", "common")); sys.path.insert(0, ROOT)
import numpy as np
import alib
from core import metrics as MT

idx, P = alib.pred_index("E8")
rows = []
for exp in ("E8", "E1"):
    for r in alib.load(exp):
        m = r.get("metrics", {}).get("measurement")
        if m is None or r["config"]["n"] != 2:
            continue
        key = ("product", 3, m["p"], round(m["s"], 6))
        if key not in idx:
            continue
        cell = idx[key]; sc = alib.score(r, cell); row = alib.flat_row(r, cell, sc); row["source_exp"] = exp
        rows.append(row)
alib.write_csv(rows, os.path.join(HERE, "per_run.csv"))
out = {"script": "experiments/E8/analysis/analyze_E8.py", "n_rows": len(rows)}
for s in (1.25, 1.5):
    for name in ("ok", "not_converged"):
        g = [x for x in rows if x["s"] == s and x["status"] == name and x["margin"] > 0]
        if not g:
            continue
        per_p = []
        for p in sorted({x["p"] for x in g}):
            gp = [x for x in g if x["p"] == p]
            per_p.append({"p": p, "K_star": gp[0]["K_star"], "label": gp[0]["label"][:12], "n": len(gp), "median_thr10": float(np.median([x["cnt_thr_10"] for x in gp])),
                          "thr10_values": [x["cnt_thr_10"] for x in gp], "K_star_over_log_p": gp[0]["K_star"] / math.log(p),
                          "median_thr10_over_log_p": float(np.median([x["cnt_thr_10"] for x in gp])) / math.log(p), "median_tv": float(np.median([x["share_tv"] for x in gp]))})
        out[f"s{s}_{name}"] = {"per_p": per_p, "sweep_thr10": MT.sweep_summary([x["cnt_thr_10"] for x in g], [x["K_star"] for x in g]),
                               "sweep_pr_budget": MT.sweep_summary([x["cnt_pr_budget"] for x in g], [x["K_star"] for x in g]),
                               "sweep_mass_95": MT.sweep_summary([x["cnt_mass_95"] for x in g], [x["K_star"] for x in g]),
                               "Kstar_over_logp_range": [min(q["K_star_over_log_p"] for q in per_p), max(q["K_star_over_log_p"] for q in per_p)],
                               "frac_p_median_match": float(np.mean([q["median_thr10"] == q["K_star"] for q in per_p]))}
json.dump(out, open(os.path.join(HERE, "summary.json"), "w"), indent=1, default=float)
for k, v in out.items():
    if isinstance(v, dict) and "per_p" in v:
        print(k, "frac p match", alib.fmt(v["frac_p_median_match"]), "sweep thr10 exact/pm1/R2id/spearman", alib.fmt(v["sweep_thr10"]["frac_exact"]), alib.fmt(v["sweep_thr10"]["frac_within_tol"]), alib.fmt(v["sweep_thr10"]["r2_identity"]), alib.fmt(v["sweep_thr10"]["spearman"]), "K*/logp", [round(x, 2) for x in v["Kstar_over_logp_range"]])
        for q in v["per_p"]: print("   ", q)
