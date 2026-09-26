"""AUDIT check 3: E1 / E1b REG + A1 headline numbers, recomputed with own code from metrics.json + predictions.json."""
import itertools, json, math, os, sys
from collections import defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_audit import *

R = runs("E1"); idx, _ = preds("E1")
out = {"status": {}}
for r in R: out["status"][r["status"]] = out["status"].get(r["status"], 0) + 1
rows = []
for r in R:
    if r["status"] not in ("ok", "not_converged") or r["m"] is None: continue
    m, c = r["m"], r["c"]; cell = idx[key_of(r)]
    if m["margin"] <= 0: continue
    s = (c["n"] + 1) / c["q"]; p = c["p"]; nu = c["n"] + 1; A = np.asarray(m["A"])
    Y = np.asarray(cell["y_all"]); y = Y[align(A, Y, s)]
    G = cell["Gamma"]; B = Bp(p, nu)
    row = {"id": r["run_id"], "st": r["status"], "nu": nu, "p": p, "s": round(s, 6), "seed": c["seed"], "K": cell["K_star"],
           "thr10": thr(A, .1), "thr20": thr(A, .2), "tv": tv(shares(A, s), shares(y, s)), "cos": cos(bnorm(A, s), y),
           "g_univ": m["gamma"] / (B * G), "g_IF": m["gamma"] / (nu ** (-(nu - c["q"]) / c["q"]) * B * G),
           "eff": prog_value(A, p, s) / G, "clean": m["energy_frac"]["clean"], "A": A, "w": shares(A, s)}
    yh = bnorm(A, s); ss = ((yh - yh.mean()) ** 2).sum()
    row["amp_r2id"] = float(1 - ((yh - y) ** 2).sum() / ss)
    row["amp_pearson"] = float(np.corrcoef(yh, y)[0, 1])
    row["jac10"] = len(set(np.flatnonzero(A >= .1 * A.max())) & set(np.flatnonzero(y > 0))) / len(set(np.flatnonzero(A >= .1 * A.max())) | set(np.flatnonzero(y > 0)))
    row.update(energy_counts(A))
    row["match"] = row["thr10"] == row["K"] and row["tv"] <= 0.10 and row["cos"] >= 0.95
    rows.append(row)
for grp in ("ok", "not_converged"):
    g = [x for x in rows if x["st"] == grp]
    d = {"n": len(g), "frac_match": float(np.mean([x["match"] for x in g]))}
    for k in ("thr10", "thr20", "K_2pct", "K_5pct", "N_05", "N_10"):
        d[k] = sweep([x[k] for x in g], [x["K"] for x in g])
        by = defaultdict(list)
        for x in g: by[x["seed"]].append(abs(x[k] - x["K"]) <= 1)
        d[k]["within1_ci_seeds"] = boot_seeds(list(by.values()), np.mean)
    for k in ("tv", "cos", "amp_r2id", "amp_pearson", "jac10", "g_IF", "g_univ", "eff", "clean"):
        v = np.array([x[k] for x in g]); d[k] = {"median": float(np.median(v)), "q10": float(np.percentile(v, 10)), "q90": float(np.percentile(v, 90)), "min": float(v.min()), "max": float(v.max())}
    e = np.array([x["eff"] for x in g])
    by = defaultdict(list)
    for x in g: by[x["seed"]].append(x["eff"])
    d["eff"].update({"ci": boot_seeds(list(by.values()), np.median), "ge99": float(np.mean(e >= .99)), "ge95": float(np.mean(e >= .95)), "ge90": float(np.mean(e >= .90))})
    out[grp] = d
# per cell (REG support + A1 per-cell medians)
cells = []
for key in sorted({(x["nu"], x["p"], x["s"]) for x in rows}):
    for grp in ("ok", "not_converged"):
        g = [x for x in rows if (x["nu"], x["p"], x["s"]) == key and x["st"] == grp]
        if not g: continue
        med = lambda k: float(np.median([x[k] for x in g]))
        cells.append({"nu": key[0], "p": key[1], "s": key[2], "grp": grp, "n": len(g), "K": g[0]["K"], "thr10": med("thr10"), "K2": med("K_2pct"),
                      "N05": med("N_05"), "eff": round(med("eff"), 3), "min_eff": round(min(x["eff"] for x in g), 3), "tv": round(med("tv"), 3), "cos": round(med("cos"), 3),
                      "supported": grp == "ok" and med("thr10") == g[0]["K"] and med("tv") <= 0.10 and med("cos") >= 0.95})
out["cells"] = cells
out["n_cells_with_ok"] = sum(1 for c in cells if c["grp"] == "ok")
out["n_cells_supported"] = sum(1 for c in cells if c["supported"])
out["supported_cells"] = [c for c in cells if c["supported"]]
# E1b REG
ok = [x for x in rows if x["st"] == "ok"]
e1b = []
for p in sorted({x["p"] for x in ok}):
    for s in sorted({x["s"] for x in ok}):
        by = defaultdict(list)
        for x in ok:
            if x["p"] == p and x["s"] == s: by[x["nu"]].append(x)
        if len(by) < 2: continue
        srt = lambda x: np.sort(x["w"])[::-1]
        within = [tv(srt(a), srt(b)) for n1, n2 in itertools.combinations(by, 2) for a in by[n1] for b in by[n2]]
        between = [tv(srt(a), srt(b)) for nu in by for a in by[nu] for b in ok if b["p"] == p and b["nu"] == nu and b["s"] != s]
        e1b.append({"p": p, "s": s, "ratio": round(float(np.mean(within) / np.mean(between)), 3) if between else None})
out["E1b_reg"] = e1b
# E1b A1
med = defaultdict(list)
for x in ok: med[(x["p"], x["nu"], x["s"])].append(x["K_2pct"])
med = {k: float(np.median(v)) for k, v in med.items()}
within = []; between = []
for (p, nu, s) in med:
    for (p2, nu2, s2) in med:
        if p2 == p and s2 == s and nu < nu2: within.append(abs(med[(p, nu, s)] - med[(p, nu2, s)]))
        if p2 == p and nu2 == nu and s < s2: between.append(abs(med[(p, nu, s)] - med[(p, nu, s2)]))
out["E1b_A1"] = {"within_mean": float(np.mean(within)), "n_within": len(within), "frac_within_le1": float(np.mean(np.array(within) <= 1)),
                 "between_mean": float(np.mean(between)), "n_between": len(between)}
json.dump(out, open(os.path.join(ROOT, "audit", "recompute_E1.json"), "w"), indent=1, default=float)
def pr(d, keys): return {k: (round(d[k], 3) if isinstance(d[k], float) else d[k]) for k in keys}
print("status", out["status"])
for grp in ("ok", "not_converged"):
    d = out[grp]; print(grp, "n", d["n"], "match", round(d["frac_match"], 3))
    for k in ("thr10", "thr20", "K_2pct", "K_5pct", "N_05", "N_10"):
        print("  ", k, pr(d[k], ["exact", "within1", "r2_id", "r2_fit", "pearson", "spearman", "kendall", "slope", "bias"]), [round(v, 2) for v in d[k]["slope_ci"]], [round(v, 2) for v in d[k]["within1_ci_seeds"]])
    for k in ("tv", "cos", "amp_r2id", "amp_pearson", "jac10", "g_IF", "g_univ", "eff", "clean"):
        print("  ", k, {kk: round(v, 3) if isinstance(v, float) else v for kk, v in d[k].items()})
print("cells with ok", out["n_cells_with_ok"], "supported", out["n_cells_supported"], out["supported_cells"])
for c in cells: print("  ", c)
print("E1b reg", e1b); print("E1b A1", out["E1b_A1"])
