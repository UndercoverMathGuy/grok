"""initaudit_match.py -- is the count difference attributable to INIT SCALE, or only to SOLUTION QUALITY?
gamma = min margin / ||theta||^3 (scale-free normalised margin; higher = closer to the max-margin direction the
theory is about).  If init 1 and init 4 never reach the same gamma, init scale and quality are collinear in this
design and the count difference cannot be attributed to init.  Where they DO overlap, we can compare at matched gamma.
"""
import json, statistics as st
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent
cells = defaultdict(list)
for f in sorted((ROOT / "FINAL_RUNS" / "REP").glob("n2_pow_p*_l2q2_lam0.0001_*/seed*/metrics.json")):
    r = json.loads(f.read_text())
    if r["q"] != 2.0 or r["acc"] < 0.999: continue
    cells[(r["p"], r["init_scale"])].append(r)
print("== gamma ranges per cell: do the inits overlap at all? ==")
for k in sorted(cells):
    g = sorted(r["gamma"] for r in cells[k])
    print(f"  p{k[0]:>3} init{k[1]:g}: gamma [{g[0]:.3e}, {g[-1]:.3e}]  median {st.median(g):.3e}  (n={len(g)})")
print()
print("== gamma-MATCHED comparison of K_2pct, inside the overlap window of each init pair ==")
for p in (59, 71, 89, 97, 113):
    ks = sorted(k[1] for k in cells if k[0] == p)
    for i in range(len(ks) - 1):
        for j in range(i + 1, len(ks)):
            A, B = cells[(p, ks[i])], cells[(p, ks[j])]
            lo = max(min(r["gamma"] for r in A), min(r["gamma"] for r in B))
            hi = min(max(r["gamma"] for r in A), max(r["gamma"] for r in B))
            a = [r for r in A if lo <= r["gamma"] <= hi]; b = [r for r in B if lo <= r["gamma"] <= hi]
            if hi <= lo or len(a) < 3 or len(b) < 3:
                print(f"  p{p} init{ks[i]:g} vs init{ks[j]:g}: NO USABLE OVERLAP in gamma "
                      f"({'disjoint' if hi <= lo else f'{len(a)} vs {len(b)} seeds in window'}) -> init and quality are collinear")
                continue
            print(f"  p{p} init{ks[i]:g} vs init{ks[j]:g}: window [{lo:.2e},{hi:.2e}] n={len(a)} vs {len(b)} | "
                  f"K_2pct {st.median([r['K_2pct'] for r in a]):.1f} vs {st.median([r['K_2pct'] for r in b]):.1f} | "
                  f"K_5pct {st.median([r['K_5pct'] for r in a]):.1f} vs {st.median([r['K_5pct'] for r in b]):.1f} | "
                  f"N_.05 {st.median([r['N_05'] for r in a]):.1f} vs {st.median([r['N_05'] for r in b]):.1f} | "
                  f"gamma {st.median([r['gamma'] for r in a]):.2e} vs {st.median([r['gamma'] for r in b]):.2e}")
