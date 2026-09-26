"""Mechanical recipe selection (declared in rounds.py for R3, reused for R4; convergence quality only).
Usage: python select_R3.py ROUND RUNS_GLOB... -> prints and writes analysis/selection_<ROUND>.json
Rule per (model, nu), candidate (c, lambda-or-lt): (1) most 'ok' runs over the cell's p x q x seeds;
(2) ties -> smaller lambda (lt); (3) ties -> lower median final grad_rel. Diverged runs count as not ok."""
import glob, json, os, sys
import numpy as np
rnd = sys.argv[1]
rows = []
for g in sys.argv[2:]:
    for d in sorted(glob.glob(g)):
        try:
            c = json.load(open(os.path.join(d, "config.json"))); m = json.load(open(os.path.join(d, "metrics.json")))
        except Exception:
            continue
        v = m.get("convergence", {}).get("values", {})
        rows.append({"run_id": c["run_id"], "model": c["model"], "nu": c["n"] + 1, "p": c["p"], "q": c["q"],
                     "c": c.get("lr_c"), "lam_key": c.get("lam_tilde", c["lam"]), "status": m["status"],
                     "grad_rel": v.get("grad_rel", float("inf")) if m["status"] != "diverged" else float("inf"),
                     "fails": [k for k, x in m.get("convergence", {}).get("checks", {}).items() if not x]})
out = {"round": rnd, "n_runs": len(rows), "selection": {}, "table": {}}
for key in sorted({(r["model"], r["nu"]) for r in rows}):
    sub = [r for r in rows if (r["model"], r["nu"]) == key]
    cands = sorted({(r["c"], r["lam_key"]) for r in sub})
    tab = []
    for c, lk in cands:
        rr = [r for r in sub if (r["c"], r["lam_key"]) == (c, lk)]
        n_ok = sum(r["status"] == "ok" for r in rr)
        gr = [r["grad_rel"] for r in rr if np.isfinite(r["grad_rel"])]
        tab.append({"c": c, "lam_key": lk, "n": len(rr), "n_ok": n_ok, "median_grad_rel": float(np.median(gr)) if gr else float("inf"),
                    "statuses": [f"p{r['p']} q{r['q']:g} {r['status']} {','.join(r['fails'])}" for r in rr]})
    best = sorted(tab, key=lambda t: (-t["n_ok"], t["lam_key"], t["median_grad_rel"]))[0]
    out["selection"][f"{key[0]}_nu{key[1]}"] = {"c": best["c"], "lam_key": best["lam_key"], "n_ok": best["n_ok"], "n": best["n"]}
    out["table"][f"{key[0]}_nu{key[1]}"] = tab
    print(key, "->", out["selection"][f"{key[0]}_nu{key[1]}"])
    for t in tab:
        print("   ", t["c"], t["lam_key"], f"{t['n_ok']}/{t['n']}", f"{t['median_grad_rel']:.1e}", "|", "; ".join(t["statuses"]))
os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis"), exist_ok=True)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis", f"selection_{rnd}.json"), "w"), indent=1)
