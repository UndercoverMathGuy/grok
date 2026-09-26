"""E3 analysis (light, local): q = nu dense endpoint. Usage: python experiments/E3/analysis/analyze_E3.py"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(ROOT, "experiments", "common")); sys.path.insert(0, ROOT)
import numpy as np
import alib

runs = alib.load("E3")
idx, P = alib.pred_index("E3")
rows = []
for r in runs:
    if r["status"] in ("crashed", "diverged") or "measurement" not in r.get("metrics", {}):
        rows.append({"run_id": r["run_id"], "status": r["status"]}); continue
    c = r["config"]; key = (c["model"], c["n"] + 1, c["p"], 1.0)
    cell = idx[key]; sc = alib.score(r, cell); row = alib.flat_row(r, cell, sc)
    A = np.asarray(r["metrics"]["measurement"]["A"]); h = len(A)
    w = A / A.sum()  # s = 1: budget shares are amplitude shares
    row.update({"h": h, "flat_min_over_max": float(A.min() / A.max()), "cv_A": float(A.std() / A.mean()),
                "tv_to_uniform": float(0.5 * np.abs(w - 1.0 / h).sum()), "pr_amp_over_h": row["cnt_pr_amp"] / h,
                "gamma_over_ceiling_model": row["gamma_over_universal_power"] if c["model"] == "power" else row["gamma_over_universal_product"]})
    row["match"] = bool(row["cnt_thr_10"] == h and row["flat_min_over_max"] >= 0.5 and row["tv_to_uniform"] <= 0.05)
    rows.append(row)
alib.write_csv(rows, os.path.join(HERE, "per_run.csv"))
cells = []
for key in sorted(idx):
    model, nu, p, s = key
    allr = [x for x in rows if x.get("model") == model and x.get("nu") == nu and x.get("p") == p]
    st = [r["status"] for r in runs if r.get("config") and (r["config"]["model"], r["config"]["n"] + 1, r["config"]["p"]) == (model, nu, p)]
    for name, grp in (("ok", [x for x in allr if x["status"] == "ok"]), ("nc", [x for x in allr if x["status"] == "not_converged" and x["margin"] > 0])):
        if not grp:
            continue
        med = lambda k: float(np.median([g[k] for g in grp]))
        cells.append({"model": model, "nu": nu, "p": p, "group": name, "n": len(grp), "statuses": {s_: st.count(s_) for s_ in set(st)},
                      "h": grp[0]["h"], "median_thr10": med("cnt_thr_10"), "median_thr20": med("cnt_thr_20"), "median_pr_amp_over_h": med("pr_amp_over_h"),
                      "median_flat_min_over_max": med("flat_min_over_max"), "median_cv_A": med("cv_A"), "median_tv_to_uniform": med("tv_to_uniform"),
                      "median_gamma_over_ceiling": med("gamma_over_ceiling_model"), "median_energy_junk": med("energy_junk"),
                      "median_energy_offdiag": med("energy_offdiag"), "frac_match": float(np.mean([g["match"] for g in grp])),
                      "supported": bool(med("cnt_thr_10") == grp[0]["h"] and med("flat_min_over_max") >= 0.5 and med("tv_to_uniform") <= 0.05)})
out = {"script": "experiments/E3/analysis/analyze_E3.py", "status_counts": {s: sum(r["status"] == s for r in runs) for s in sorted({r["status"] for r in runs})}, "cells": cells}
cf = os.path.join(HERE, "controls.json")
if os.path.exists(cf):
    C = json.load(open(cf)); ctl = []
    for c in C["cells"]:
        for nm in ("positive", "null_init"):
            A = np.asarray(c[nm]["measurement"]["A"]); w = A / A.sum(); h = len(A)
            ctl.append({"model": c["cell"]["model"], "nu": c["cell"]["nu"], "p": c["cell"]["p"], "which": nm, "thr10": int((A >= 0.1 * A.max()).sum()), "h": h,
                        "flat_min_over_max": float(A.min() / A.max()), "tv_to_uniform": float(0.5 * np.abs(w - 1 / h).sum()),
                        "gamma_over_ceiling": c[nm]["score"]["gamma_over_ceiling"]["universal_power" if c["cell"]["model"] == "power" else "universal_product"]})
    out["controls"] = ctl
json.dump(out, open(os.path.join(HERE, "summary.json"), "w"), indent=1)
print(out["status_counts"])
for c in cells:
    print(c["model"], c["nu"], c["p"], c["group"], "n", c["n"], c["statuses"], "h", c["h"], "thr10", c["median_thr10"], "thr20", c["median_thr20"], "PR/h", alib.fmt(c["median_pr_amp_over_h"]),
          "min/max", alib.fmt(c["median_flat_min_over_max"]), "CV", alib.fmt(c["median_cv_A"]), "TVu", alib.fmt(c["median_tv_to_uniform"]),
          "g/ceil", alib.fmt(c["median_gamma_over_ceiling"]), "junk", alib.fmt(c["median_energy_junk"]), "off", alib.fmt(c["median_energy_offdiag"]), "match", alib.fmt(c["frac_match"]), c["supported"])
for c in out.get("controls", []):
    print("ctl", c)
