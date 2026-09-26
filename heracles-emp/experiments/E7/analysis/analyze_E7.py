"""E7 analysis (light, local): plateaus in q at p = 37, nu = 4. Usage: python experiments/E7/analysis/analyze_E7.py"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(ROOT, "experiments", "common")); sys.path.insert(0, ROOT)
import numpy as np
import alib
from core import metrics as MT, measure as MS, programme as PG

P = json.load(open(os.path.join(ROOT, "experiments", "E7", "predictions.json")))
plateaus = P["plateaus"]; p = 37
reps = [np.asarray(pl["rep"]) for pl in plateaus]
dil = [PG.all_dilates(v, p) for v in reps]
def plateau_of(s):
    for k, pl in enumerate(plateaus):
        if pl["s_from"] - 1e-12 <= s <= pl["s_to"] + 1e-12:
            return k
runs = alib.load("E7"); idx = {round(c["s"], 6): c for c in P["cells"]}
rows = []
for r in runs:
    m = r.get("metrics", {}).get("measurement")
    if m is None:
        rows.append({"run_id": r["run_id"], "status": r["status"]}); continue
    s = round(m["s"], 6); cell = idx[s]
    sc = alib.score(r, cell); row = alib.flat_row(r, cell, sc)
    w = np.asarray(m["budget_shares"])
    tvs = [min(MT.tv_distance(w, MS.budget_shares(y, m["s"])) for y in D) for D in dil]
    kbest = int(np.argmin(tvs)); kpred = plateau_of(m["s"])
    row.update({"plateau_pred": kpred, "plateau_best": kbest, "tv_to_pred_plateau": tvs[kpred], "tv_to_best_plateau": tvs[kbest],
                "K_pred_plateau": len(plateaus[kpred]["support"]), "K_best_plateau": len(plateaus[kbest]["support"]),
                "plateau_match": kbest == kpred, "sorted_yhat": np.sort(np.asarray(m["yhat"]))[::-1][:20].tolist()})
    rows.append(row)
alib.write_csv([{k: v for k, v in x.items() if k != "sorted_yhat"} for x in rows], os.path.join(HERE, "per_run.csv"))
out = {"script": "experiments/E7/analysis/analyze_E7.py", "switch_points": P["switch_points"],
       "status_counts": {s: sum(r["status"] == s for r in runs) for s in sorted({r["status"] for r in runs})}}
for name in ("ok", "not_converged"):
    g = [x for x in rows if x.get("status") == name and x.get("margin", 0) > 0]
    if not g:
        continue
    out[name] = {"n": len(g), "frac_plateau_match": float(np.mean([x["plateau_match"] for x in g])),
                 "frac_thr10_eq_Kstar": float(np.mean([x["cnt_thr_10"] == x["K_star"] for x in g])),
                 "count_sweep_thr10": MT.sweep_summary([x["cnt_thr_10"] for x in g], [x["K_star"] for x in g]),
                 "median_tv_pred": float(np.median([x["tv_to_pred_plateau"] for x in g])),
                 "per_s": [{"s": x["s"], "seed": x["seed"], "K*": x["K_star"], "thr10": x["cnt_thr_10"], "plateau_pred": x["plateau_pred"], "plateau_best": x["plateau_best"],
                            "tv_pred": round(x["tv_to_pred_plateau"], 4), "tv_best": round(x["tv_to_best_plateau"], 4), "share_tv_aligned": round(x["share_tv"], 4)} for x in sorted(g, key=lambda z: (z["s"], z["seed"]))]}
    # within-plateau vs across-adjacent-plateau spread of sorted yhat
    within, across = [], []
    for a in g:
        for b in g:
            if a["run_id"] >= b["run_id"]:
                continue
            d = 0.5 * np.abs(np.asarray(a["sorted_yhat"]) - np.asarray(b["sorted_yhat"])).sum()
            if a["plateau_pred"] == b["plateau_pred"]:
                within.append(d)
            elif abs(a["plateau_pred"] - b["plateau_pred"]) == 1:
                across.append(d)
    out[name]["sorted_yhat_L1half_within_plateau"] = float(np.mean(within)) if within else None
    out[name]["sorted_yhat_L1half_adjacent_plateaus"] = float(np.mean(across)) if across else None
json.dump(out, open(os.path.join(HERE, "summary.json"), "w"), indent=1, default=float)
print(out["status_counts"])
for name in ("ok", "not_converged"):
    if name in out:
        d = out[name]; print(name, "n", d["n"], "plateau match", alib.fmt(d["frac_plateau_match"]), "thr10==K*", alib.fmt(d["frac_thr10_eq_Kstar"]), "within/adj", alib.fmt(d["sorted_yhat_L1half_within_plateau"]), alib.fmt(d["sorted_yhat_L1half_adjacent_plateaus"]))
        for x in d["per_s"]: print("  ", x)
