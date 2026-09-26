"""AUDIT: independent per-offset worst margins (folded) on sampled E1 runs vs metrics.json (E5 input)."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from remeasure import load, table
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out = {}
for o in json.load(open(os.path.join(ROOT, "audit", "sample.json"))):
    if o["exp"] != "E1": continue
    d = os.path.join(ROOT, "experiments", "E1", "runs", o["run_id"]); c = json.load(open(os.path.join(d, "config.json")))
    m = json.load(open(os.path.join(d, "metrics.json")))["measurement"]
    U, W = load(d); p = c["p"]; n = c["n"]; F = table(c["model"], U, W)
    a = np.indices((p,) * n).reshape(n, -1); y = a.sum(0) % p; fy = F[np.arange(len(y)), y]
    D = fy[:, None] - F; delta = (y[:, None] - np.arange(p)[None, :]) % p
    per = np.array([D[delta == dd].min() for dd in range(1, p)]); h = (p - 1) // 2
    fold = np.minimum(per[:h], per[::-1][:h])
    out[o["run_id"]] = float(np.abs(fold - np.array(m["per_offset_margin_folded"])).max())
    print(o["run_id"], out[o["run_id"]])
json.dump(out, open(os.path.join(ROOT, "audit", "peroffset_check.json"), "w"), indent=1)
