"""initaudit_place.py -- place the measured normalised margins against the exact constructed ceiling (H3),
and count how many 'fit' seeds are actually non-separating (H4 instrument leak)."""
import json, statistics as st
from collections import defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parent
REP = ROOT / "FINAL_RUNS" / "REP"
CEIL = {p: json.loads((ROOT / "FINAL_RUNS" / "INITAUDIT" / f"ceiling_p{p}.json").read_text()) for p in (59, 71)}
cells = defaultdict(list)
for f in sorted(REP.glob("n2_pow_*_l2q2*/seed*/metrics.json")):
    r = json.loads(f.read_text())
    if r["q"] != 2.0: continue
    cells[(r["p"], r["lam"], r["init_scale"], "_from" in f.parent.parent.name)].append(r)
med = st.median
print("== H3d: measured gamma as a FRACTION of the exact constructed ceiling (p 59 / 71 only) ==")
print("   cell                             | gamma med  | / gamma_con(K*) | / gamma_con(K=10) | K2 med | non-separating seeds (margin<=0) of fit")
for k in sorted(cells):
    p, lam, init, res = k
    if p not in CEIL: continue
    rs = [r for r in cells[k] if r["acc"] >= 0.999]
    if not rs: continue
    kstar = "6" if p == 59 else "5"
    g = med([r["gamma"] for r in rs])
    neg = sum(1 for r in rs if r["margin"] <= 0)
    print(f"  p{p} lam{lam:g} init{init:g}{' +cont' if res else '      '} ({len(rs):2d} fit) | {g:.3e}  | {g/CEIL[p][kstar]:15.3f} | "
          f"{g/CEIL[p]['10']:17.3f} | {med([r['K_2pct'] for r in rs]):6} | {neg}/{len(rs)}")
print()
print("== ceiling reference ==")
for p in (59, 71):
    print(f"  p{p}: " + "  ".join(f"K{K}:{CEIL[p][str(K)]:.3e}" for K in (3,4,5,6,7,8,10,12)))
print()
print("== H4c: how many seeds in each cell fail to SEPARATE (min margin <= 0) although acc >= 0.999 ==")
for k in sorted(cells):
    rs = cells[k]
    fit = [r for r in rs if r["acc"] >= 0.999]
    neg = [r["seed"] for r in fit if r["margin"] <= 0]
    if neg:
        print(f"  p{k[0]} lam{k[1]:g} init{k[2]:g}{' cont' if k[3] else ''}: {len(neg)}/{len(fit)} fit seeds have margin<=0: {neg}")
