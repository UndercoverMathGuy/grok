"""E6 analysis (light, local): dilation equivariance (paired relabelled inits) and marginals. Usage: python experiments/E6/analysis/analyze_E6.py"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(ROOT, "experiments", "common")); sys.path.insert(0, ROOT)
import numpy as np, torch
from scipy import stats
import alib
from core import programme as PG, train as TR

runs = {r["run_id"]: r for r in alib.load("E6")}
p = 23; h = 11
out = {"script": "experiments/E6/analysis/analyze_E6.py", "status_counts": {}}
for r in runs.values():
    out["status_counts"][r["status"]] = out["status_counts"].get(r["status"], 0) + 1
pairs = []
for seed in range(6):
    a = runs.get(f"E6_pair_seed{seed}_u1")
    if not a or "measurement" not in a.get("metrics", {}):
        continue
    Aa = np.asarray(a["metrics"]["measurement"]["A"]); Ua, Wa = TR.load_final(a["dir"])
    for u in (2, 5):
        b = runs.get(f"E6_pair_seed{seed}_u{u}")
        if not b or "measurement" not in b.get("metrics", {}):
            continue
        Ab = np.asarray(b["metrics"]["measurement"]["A"]); Ub, Wb = TR.load_final(b["dir"])
        js = np.arange(1, h + 1); mapped = Aa[PG.fold(u * js, p) - 1]
        src = torch.tensor([(pow(u, -1, p) * x) % p for x in range(p)])
        Ur, Wr = Ua[..., src], Wa[..., src]
        drift = float(torch.sqrt(((Ub - Ur) ** 2).sum() + ((Wb - Wr) ** 2).sum()) / torch.sqrt((Ua ** 2).sum() + (Wa ** 2).sum()))
        sa = set((np.flatnonzero(Aa >= 0.1 * Aa.max()) + 1).tolist()); sb = set((np.flatnonzero(Ab >= 0.1 * Ab.max()) + 1).tolist())
        sa_mapped = {j for j in range(1, h + 1) if int(PG.fold(u * j, p)) in sa}
        pairs.append({"seed": seed, "u": u, "status_a": a["status"], "status_b": b["status"], "max_rel_amp_diff": float(np.abs(Ab - mapped).max() / Aa.max()),
                      "param_drift_rel": drift, "support_a": sorted(sa), "support_b": sorted(sb), "support_a_mapped": sorted(sa_mapped),
                      "support_map_exact": sb == sa_mapped, "gamma_rel_diff": abs(a["metrics"]["measurement"]["gamma"] - b["metrics"]["measurement"]["gamma"]) / abs(a["metrics"]["measurement"]["gamma"])})
out["pairs"] = pairs
if pairs:
    out["pairs_summary"] = {"n": len(pairs), "frac_support_map_exact": float(np.mean([x["support_map_exact"] for x in pairs])),
                            "max_rel_amp_diff_max": float(max(x["max_rel_amp_diff"] for x in pairs)), "median_param_drift": float(np.median([x["param_drift_rel"] for x in pairs])),
                            "max_param_drift": float(max(x["param_drift_rel"] for x in pairs))}
marg = [r for k, r in runs.items() if k.startswith("E6_marg_") and "measurement" in r.get("metrics", {})]
for name, grp in (("ok", [r for r in marg if r["status"] == "ok"]), ("all_pos_margin", [r for r in marg if r["metrics"]["measurement"]["margin"] > 0])):
    if not grp:
        continue
    act = np.zeros(h); Ks = []
    for r in grp:
        A = np.asarray(r["metrics"]["measurement"]["A"]); s_ = A >= 0.1 * A.max(); act += s_; Ks.append(int(s_.sum()))
    exp_ = np.full(h, act.sum() / h)
    chi = stats.chisquare(act, exp_)
    out[f"marginals_{name}"] = {"n_runs": len(grp), "P_active_per_j": (act / len(grp)).tolist(), "E_K_over_h": float(np.mean(Ks) / h),
                                "chi2": float(chi.statistic), "chi2_p": float(chi.pvalue), "dof": h - 1, "K_distribution": {int(k): Ks.count(k) for k in sorted(set(Ks))}}
json.dump(out, open(os.path.join(HERE, "summary.json"), "w"), indent=1, default=str)
print(out["status_counts"]); print(out.get("pairs_summary"))
for x in pairs: print(x)
for k in ("marginals_ok", "marginals_all_pos_margin"):
    if k in out: print(k, {kk: (np.round(v, 3).tolist() if isinstance(v, list) else v) for kk, v in out[k].items()})
