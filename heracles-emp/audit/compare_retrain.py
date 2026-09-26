"""AUDIT 4b: compare retrained runs (audit/retrain/runs/*_audit) with the originals; own re-measurement of both."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_audit import *
from remeasure import measure
res = []
for o in json.load(open(os.path.join(ROOT, "audit", "sample.json"))):
    if not o["retrain"]: continue
    d0 = os.path.join(EX, o["exp"], "runs", o["run_id"]); d1 = os.path.join(ROOT, "audit", "retrain", "runs", o["run_id"] + "_audit")
    m0 = json.load(open(os.path.join(d0, "metrics.json"))); m1 = json.load(open(os.path.join(d1, "metrics.json")))
    a0, a1 = measure(d0), measure(d1)
    A0, A1 = np.asarray(a0["A"]), np.asarray(a1["A"]); s = a0["nu"] / a0["q"]
    res.append({"run_id": o["run_id"], "status": (m0["status"], m1["status"]), "gamma": (a0["gamma"], a1["gamma"]), "gamma_rel_diff": abs(a1["gamma"] - a0["gamma"]) / abs(a0["gamma"]),
                "margin": (a0["margin"], a1["margin"]), "thr10": (thr(A0, .1), thr(A1, .1)), "thr20": (thr(A0, .2), thr(A1, .2)),
                "K_2pct": (energy_counts(A0)["K_2pct"], energy_counts(A1)["K_2pct"]),
                "support10_same": set(np.flatnonzero(A0 >= .1 * A0.max())) == set(np.flatnonzero(A1 >= .1 * A1.max())),
                "A_max_diff_rel": float(np.abs(A0 - A1).max() / A0.max()), "share_tv": tv(shares(A0, s), shares(A1, s)),
                "audit_vs_metrics_retrained_gamma": abs(a1["gamma"] - m1["measurement"]["gamma"]) / abs(m1["measurement"]["gamma"]),
                "device": json.load(open(os.path.join(d1, "config.json")))["environment"]["cuda_device"]})
    print(res[-1])
json.dump(res, open(os.path.join(ROOT, "audit", "retrain", "compare.json"), "w"), indent=1, default=lambda x: bool(x) if isinstance(x, np.bool_) else str(x))
