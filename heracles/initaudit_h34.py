"""initaudit_h34.py -- H3 (margin) and H4 (survivorship) on the saved REP metrics.
Symbols:
  norm2   = ||theta||^2 = sum of squares of ALL weights (both embedding tables E_0,E_1 and the readout W).
  margin  = min over all p^2 table rows of (logit of the correct answer - largest wrong logit).
  gamma   = margin / norm2^{nu/2} with nu = 3  (scale-free "normalised margin": what the theory maximises).
  obj     = CE + lam*norm2  = the actual training objective being minimised (lam = 1e-4, q = 2).
  esc     = first logged step (multiples of 2000) at which train accuracy >= 0.99.
"""
import json, statistics as st
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent
REP = ROOT / "FINAL_RUNS" / "REP"
cells = defaultdict(list)
for f in sorted(REP.glob("n2_pow_*_l2q2_*/seed*/metrics.json")):
    r = json.loads(f.read_text())
    if r["q"] != 2.0 or "_from" in f.parent.parent.name: continue
    cells[(r["p"], r["lam"], r["init_scale"])].append(r)
med = st.median

print("== H3: training objective and normalised margin, medians over fit seeds (acc>=0.999) ==")
print("   p  lam    init  nfit |   CE      lam*n2    obj     | norm2    margin   gamma      | K2   K5")
for k in sorted(cells):
    rs=[r for r in cells[k] if r["acc"]>=0.999]
    if not rs: continue
    ce=[r["final_ce"] for r in rs]; n2=[r["norm2"] for r in rs]
    obj=[r["final_ce"]+k[1]*r["norm2"] for r in rs]
    print(f"{k[0]:>4} {k[1]:>7g} {k[2]:>4g} {len(rs):>5} | {med(ce):.4f}  {k[1]*med(n2):.4f}  {med(obj):.4f} | "
          f"{med(n2):7.1f} {med([r['margin'] for r in rs]):7.3f} {med([r['gamma'] for r in rs]):.3e} | "
          f"{med([r['K_2pct'] for r in rs]):4} {med([r['K_5pct'] for r in rs]):4}")

print()
print("== H3b: per-seed gamma vs K_2pct, p=71 lam=1e-4, all three inits (16 seeds each) ==")
print(" init seed  K2  K5  gamma      margin   norm2   CE      obj     clean")
allrows=[]
for init in (1.0,2.0,4.0):
    for r in sorted(cells[(71,1e-4,init)], key=lambda r:r["seed"]):
        ok = r["acc"]>=0.999
        allrows.append((init,r))
        print(f" {init:>4g} {r['seed']:>4} {r['K_2pct']:>3} {r['K_5pct']:>3} {r['gamma']:>10.3e} {r['margin']:>7.3f} "
              f"{r['norm2']:>7.1f} {r['final_ce']:.4f} {r['final_ce']+1e-4*r['norm2']:.4f} {r['clean_frac']:.3f}{'' if ok else '  UNFIT'}")
# correlation gamma vs K within and across inits
import itertools
def spear(x,y):
    x=np.asarray(x,float); y=np.asarray(y,float)
    rx=np.argsort(np.argsort(x)).astype(float); ry=np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx,ry)[0,1])
print()
print("== H3c: Spearman rank corr of gamma vs count, pooled over all x^2 q2 lam1e-4 fit seeds ==")
fit=[r for k in cells if k[1]==1e-4 for r in cells[k] if r["acc"]>=0.999]
for kk in ("K_2pct","K_5pct","N_05"):
    print(f"  pooled all p: rho(gamma, {kk}) = {spear([r['gamma'] for r in fit],[r[kk] for r in fit]):+.3f}  (n={len(fit)})")
for p in (59,71,89,97,113):
    sub=[r for r in fit if r["p"]==p]
    if len(sub)<10: continue
    print(f"  p={p:>4} (n={len(sub)}): " + "  ".join(f"rho(gamma,{kk})={spear([r['gamma'] for r in sub],[r[kk] for r in sub]):+.3f}" for kk in ("K_2pct","K_5pct")))
# within-init (fixed init, does count still track gamma?)
print("  within-cell (fixed p and init):")
for k in sorted(cells):
    if k[1]!=1e-4: continue
    sub=[r for r in cells[k] if r["acc"]>=0.999]
    if len(sub)<10: continue
    print(f"    p={k[0]} init{k[2]:g}: rho(gamma,K2)={spear([r['gamma'] for r in sub],[r['K_2pct'] for r in sub]):+.3f} "
          f"rho(gamma,K5)={spear([r['gamma'] for r in sub],[r['K_5pct'] for r in sub]):+.3f} "
          f"rho(gamma,clean)={spear([r['gamma'] for r in sub],[r['clean_frac'] for r in sub]):+.3f}")

print()
print("== H4: survivorship -- every seed of the collapsed / marginal init-1 cells ==")
for k in sorted(cells):
    if k[1]!=1e-4: continue
    rs=sorted(cells[k],key=lambda r:r["seed"])
    nfit=sum(1 for r in rs if r["acc"]>=0.999)
    if nfit==len(rs): continue
    print(f"  p={k[0]} init{k[2]:g}: {nfit}/{len(rs)} fit; acc = " + " ".join(f"{r['acc']:.3f}" for r in rs))
print()
print("== H4b: escape step (first log step with acc>=0.99) vs count, per cell ==")
print("   p init | esc distribution                         | rho(esc,K2) rho(esc,gamma)")
for k in sorted(cells):
    if k[1]!=1e-4: continue
    rs=[r for r in cells[k] if r["acc"]>=0.999]
    if len(rs)<10: continue
    esc=[]
    for r in rs:
        e=[L["step"] for L in r["log"] if L["acc"]>=0.99]; esc.append(e[0] if e else 10**9)
    from collections import Counter
    c=Counter(esc)
    print(f"{k[0]:>4}{k[2]:>4g} | " + " ".join(f"{s}:{n}" for s,n in sorted(c.items())).ljust(40) +
          f" | {spear(esc,[r['K_2pct'] for r in rs]):+.3f}      {spear(esc,[r['gamma'] for r in rs]):+.3f}")
