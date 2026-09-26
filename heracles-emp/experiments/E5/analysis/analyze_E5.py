"""E5 analysis (light, local): bottleneck offsets on E1 networks + constructed controls. Usage: python experiments/E5/analysis/analyze_E5.py"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(ROOT, "experiments", "common")); sys.path.insert(0, ROOT)
import numpy as np
import alib
from core import metrics as MT, programme as PG

runs = alib.load("E1"); idx, P = alib.pred_index("E1")
def near(hf, eps):
    hf = np.asarray(hf, float); mn = hf.min()
    return set((np.flatnonzero(hf <= mn + eps * abs(mn)) + 1).tolist())
rows = []
for r in runs:
    m = r.get("metrics", {}).get("measurement")
    if m is None or r["status"] not in ("ok", "not_converged") or m["margin"] <= 0:
        continue
    c = r["config"]; cell = idx[("product", c["n"] + 1, c["p"], round((c["n"] + 1) / c["q"], 6))]
    al = MT.align(np.asarray(m["A"]), np.asarray(cell["y_all"]), m["s"])
    y = np.asarray(cell["y_all"])[al["index"]]
    tight = set(PG.tight_offsets(y, c["p"]).tolist())
    hf = m["per_offset_margin_folded"]
    K = m["counts"]["thr_10"]
    row = {"run_id": r["run_id"], "status": r["status"], "nu": m["nu"], "p": m["p"], "s": round(m["s"], 6), "K_thr10": K, "K_star": cell["K_star"],
           "n_tight_pred": len(tight)}
    for eps in (0.01, 0.05):
        S = near(hf, eps); row[f"n_near_{eps}"] = len(S); row[f"jacc_near_{eps}_vs_tight"] = MT.jaccard(S, tight)
    row["match"] = bool(row["n_near_0.05"] >= K and row["jacc_near_0.05_vs_tight"] >= 0.5)
    rows.append(row)
alib.write_csv(rows, os.path.join(HERE, "per_run.csv"))
out = {"script": "experiments/E5/analysis/analyze_E5.py"}
for name in ("ok", "not_converged"):
    g = [x for x in rows if x["status"] == name]
    if not g:
        continue
    out[name] = {"n": len(g), "frac_match": float(np.mean([x["match"] for x in g])),
                 "n_near_0.05_minus_K": MT.sweep_summary([x["n_near_0.05"] for x in g], [x["K_thr10"] for x in g]),
                 "n_near_0.01_vs_n_tight_pred": MT.sweep_summary([x["n_near_0.01"] for x in g], [x["n_tight_pred"] for x in g]),
                 "median_jacc_0.05": float(np.median([x["jacc_near_0.05_vs_tight"] for x in g])),
                 "median_jacc_0.01": float(np.median([x["jacc_near_0.01_vs_tight"] for x in g])),
                 "frac_n_near_0.05_ge_K": float(np.mean([x["n_near_0.05"] >= x["K_thr10"] for x in g])),
                 "supported": bool(np.mean([x["match"] for x in g]) >= 0.5)}
cf = os.path.join(ROOT, "experiments", "E1", "analysis", "controls.json")
if os.path.exists(cf):
    C = json.load(open(cf)); ctl = []
    from core import construct as CO, measure as MS
    for c in C["cells"]:
        cell = idx[("product", c["cell"]["nu"], c["cell"]["p"], round(c["cell"]["s"], 6))]
        y0 = np.asarray(cell["y_star_canonical"]); tight = set(PG.tight_offsets(y0, cell["p"]).tolist())
        # folded per-offset margins of the constructed network = coverage of y0 (exact identity for nu-blocks)
        cov = PG.coverage_matrix(cell["p"]) @ y0
        S = near(cov, 1e-9)
        ctl.append({"nu": cell["nu"], "p": cell["p"], "s": cell["s"], "K_star": cell["K_star"], "n_tight": len(tight), "n_near": len(S), "jacc": MT.jaccard(S, tight)})
    out["positive_controls"] = ctl
    out["positive_controls_all_match"] = bool(all(x["jacc"] == 1.0 and x["n_tight"] >= x["K_star"] for x in ctl))
json.dump(out, open(os.path.join(HERE, "summary.json"), "w"), indent=1, default=float)
for k in ("ok", "not_converged"):
    if k in out:
        d = out[k]; print(k, "n", d["n"], "frac_match", alib.fmt(d["frac_match"]), "frac n_near>=K", alib.fmt(d["frac_n_near_0.05_ge_K"]), "jacc .05/.01", alib.fmt(d["median_jacc_0.05"]), alib.fmt(d["median_jacc_0.01"]),
                          "n_near.01 vs n_tight exact", alib.fmt(d["n_near_0.01_vs_n_tight_pred"]["frac_exact"]), "supported", d["supported"])
if "positive_controls" in out:
    print("controls all match:", out["positive_controls_all_match"])
