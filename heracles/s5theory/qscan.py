"""qscan.py -- symmetric-class argmax (exact, schoolbook kappa at even nu) across the whole q range [2, nu]; unrestricted too."""
import sys, json
from fractions import Fraction
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import gt_core as gc
from programme import kap_fn
out = {}; lines = []
for g, ex in [("s5", True), ("s4", True), ("a5", False)]:
    G = gc.get(g); irr, _ = gc.programme_data(G); cov = np.array([p["cov"] for p in irr])
    verts = gc.vertices(np.array([[Fraction(x).limit_denominator(10**6) for x in row] for row in cov], dtype=object), exact=True) if ex else gc.vertices(cov, exact=False)
    for nu in (3, 4, 5, 6, 8):
        for mode in ("school", "unres"):
            prev = None; segs = []
            for q in np.linspace(2, nu, 81):
                r = q / nu
                c = gc.prices(irr, nu, q, kap_fn(mode, nu, q)); Gam, k, costs = gc.solve(verts, c, r)
                srt = np.sort(costs); gap = srt[1] / srt[0] if len(srt) > 1 else np.inf
                sup = "+".join(gc.support_names(irr, verts[k]))
                if sup != prev: segs.append([round(float(q), 3), sup, round(float(gap ** (-1 / r)), 4)]); prev = sup
            out[f"{g}_nu{nu}_{mode}"] = segs
            lines.append(f"{g} nu {nu} [{mode:6s}] argmax by q (start of each segment, runner-up/argmax Gamma at that q): " + "; ".join(f"q>={a}: {b} ({c})" for a, b, c in segs))
print("\n".join(lines)); (HERE / "out" / "qscan.log").write_text("\n".join(lines) + "\n"); (HERE / "out" / "qscan.json").write_text(json.dumps(out, indent=1))
