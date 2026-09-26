"""Exact switch points of the optimal vertex class (core.programme.switch_points) for E7 planning/prediction.
Runs ON A POD. Usage: python experiments/common/switches.py OUT.json p s_lo s_hi"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
from core import programme as P
out, p, lo, hi = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
t0 = time.time()
r = P.switch_points(p, lo, hi)
def conv(o):
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, (np.floating, np.integer)): return o.item()
    if isinstance(o, dict): return {k: conv(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [conv(v) for v in o]
    return o
r = conv(r); r["p"] = p; r["s_range"] = [lo, hi]; r["seconds"] = time.time() - t0; r["programme_sha256"] = P.SELF_SHA256
json.dump(r, open(out, "w"), indent=1)
print(p, [round(s, 5) for s in r["switches"]], [(pl["s_from"], pl["s_to"], len(pl["support"])) for pl in r["plateaus"]], f"{r['seconds']:.1f}s")
