"""E4 analysis (light, local): ablation bound on E1 networks + constructed controls. Usage: python experiments/E4/analysis/analyze_E4.py"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(ROOT, "experiments", "common")); sys.path.insert(0, ROOT)
import numpy as np
import alib

rng = np.random.default_rng(0)
runs = alib.load("E1")
recs = []
for r in runs:
    m = r.get("metrics", {}).get("measurement")
    if m is None or "ablation" not in m or r["status"] not in ("ok", "not_converged") or m["margin"] <= 0:
        continue
    s = m["s"]; w = np.asarray(m["budget_shares"]); rl = np.asarray(m["ablation"]["rel_loss"], float)
    bound = 1 - (1 - w) ** s
    perm = []
    for _ in range(100):
        wp = rng.permutation(w); perm.append(np.mean((rl >= 1 - (1 - wp) ** s - 1e-9)[wp >= 0.01]) if np.any(wp >= 0.01) else np.nan)
    recs.append({"run_id": r["run_id"], "status": r["status"], "nu": m["nu"], "p": m["p"], "s": round(s, 6), "w": w, "rel_loss": rl, "bound": bound,
                 "perm_frac": float(np.nanmean(perm))})
def frac(group, active_only=True):
    sat, tot, viol = 0, 0, []
    for x in group:
        sel = x["w"] >= 0.01 if active_only else np.ones_like(x["w"], bool)
        ok = x["rel_loss"][sel] >= x["bound"][sel] - 1e-9
        sat += int(ok.sum()); tot += int(sel.sum())
        for j in np.flatnonzero(sel)[~ok]:
            viol.append({"run_id": x["run_id"], "j": int(j + 1), "w": float(x["w"][j]), "rel_loss": float(x["rel_loss"][j]), "bound": float(x["bound"][j])})
    return {"n_pairs": tot, "n_satisfied": sat, "frac": sat / tot if tot else float("nan"), "violations": viol}
out = {"script": "experiments/E4/analysis/analyze_E4.py", "n_runs": len(recs)}
for name, g in (("ok", [x for x in recs if x["status"] == "ok"]), ("nc", [x for x in recs if x["status"] == "not_converged"])):
    d = {"active_w_ge_0.01": frac(g), "all_j": frac(g, False), "perm_null_frac_mean": float(np.mean([x["perm_frac"] for x in g])) if g else None}
    d["by_nu"] = {nu: {k: v for k, v in frac([x for x in g if x["nu"] == nu]).items() if k != "violations"} for nu in (3, 4, 5)}
    d["by_s"] = {str(s): {k: v for k, v in frac([x for x in g if x["s"] == s]).items() if k != "violations"} for s in sorted({x["s"] for x in g})}
    marg = np.concatenate([(x["rel_loss"] - x["bound"])[x["w"] >= 0.01] for x in g]) if g else np.array([])
    d["excess_quantiles_(rel_loss-bound)"] = {q: float(np.percentile(marg, q)) for q in (1, 5, 25, 50, 75)} if len(marg) else None
    d["supported"] = bool(d["active_w_ge_0.01"]["frac"] >= 0.90) if g else None
    d["n_runs"] = len(g)
    out[name] = d
cf = os.path.join(ROOT, "experiments", "E1", "analysis", "controls.json")
if os.path.exists(cf):
    C = json.load(open(cf)); pos = []
    for c in C["cells"]:
        m = c["positive"]["measurement"]; s = c["cell"]["s"]
        w = np.asarray(m["A"]) ** (1 / s); w = w / w.sum(); rl = np.asarray(m["ablation"]["rel_loss"], float)
        pos.append({"w": w, "rel_loss": rl, "bound": 1 - (1 - w) ** s, "run_id": f"ctl_nu{c['cell']['nu']}_p{c['cell']['p']}_s{s}", "nu": c["cell"]["nu"], "s": s})
    fp = frac(pos)
    out["positive_controls"] = {k: v for k, v in fp.items()}
    out["positive_controls_all_satisfy"] = bool(fp["n_satisfied"] == fp["n_pairs"])
json.dump(out, open(os.path.join(HERE, "summary.json"), "w"), indent=1, default=float)
for k in ("ok", "nc"):
    if k in out:
        d = out[k]; print(k, "runs", d["n_runs"], "active frac", alib.fmt(d["active_w_ge_0.01"]["frac"]), f"({d['active_w_ge_0.01']['n_satisfied']}/{d['active_w_ge_0.01']['n_pairs']})",
                          "all-j", alib.fmt(d["all_j"]["frac"]), "perm-null", alib.fmt(d["perm_null_frac_mean"]), "excess q", d["excess_quantiles_(rel_loss-bound)"], "by_nu", d["by_nu"], "supported", d["supported"])
if "positive_controls" in out:
    print("positive controls:", out["positive_controls"]["n_satisfied"], "/", out["positive_controls"]["n_pairs"], out["positive_controls"]["violations"][:5])
