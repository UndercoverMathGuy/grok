"""AUDIT: independent frequency ablation (project +-j out of every weight vector) on sampled E1 runs vs metrics.json."""
import json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from remeasure import load, table
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def margin(model, U, W):
    n, m, p = U.shape; F = table(model, U, W); a = np.indices((p,) * n).reshape(n, -1); y = a.sum(0) % p
    fy = F[np.arange(len(y)), y]; F[np.arange(len(y)), y] = -np.inf; return float((fy - F.max(1)).min())
out = {}
for rid in ["E1_nu5_p13_s1.25_seed2", "E1_nu5_p17_s2_seed1", "E1_nu5_p17_s2.5_seed0"]:
    d = os.path.join(ROOT, "experiments", "E1", "runs", rid); c = json.load(open(os.path.join(d, "config.json")))
    m = json.load(open(os.path.join(d, "metrics.json")))["measurement"]
    U, W = load(d); p = c["p"]; h0 = margin(c["model"], U, W); x = np.arange(p); rl = []
    for j in range(1, (p - 1) // 2 + 1):
        B = np.stack([np.cos(2 * np.pi * j * x / p), np.sin(2 * np.pi * j * x / p)]); B /= np.linalg.norm(B, axis=1, keepdims=True)
        pr = lambda V: V - (V @ B.T) @ B
        rl.append(1 - margin(c["model"], pr(U), pr(W)) / h0)
    out[rid] = float(np.abs(np.array(rl) - np.array(m["ablation"]["rel_loss"])).max())
    print(rid, "max |rel_loss audit - metrics|", out[rid])
json.dump(out, open(os.path.join(ROOT, "audit", "ablation_check.json"), "w"), indent=1)
