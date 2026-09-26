"""arch_det.py -- adversarial minimisation of  det(M_+) / m^D  (the top rung of (E+); by Maclaurin
this is the whole conjecture).  Also records the k-rungs and the Maclaurin chain at the minimiser.
usage: arch_det.py n N nstart seed [families]      families default: block,random,aligned,twist,collapse,pop
"""
import numpy as np, sys, json, time
from scipy.optimize import minimize
from arch_core2 import *

n = int(sys.argv[1]); N = int(sys.argv[2]); nstart = int(sys.argv[3]); seed = int(sys.argv[4])
fams = sys.argv[5].split(',') if len(sys.argv) > 5 else ['block', 'random', 'aligned', 'twist', 'collapse', 'pop']
D = n + 1; rng = np.random.default_rng(seed)
Gfast = {1: None, 2: 180, 3: 24, 4: 10}[n]

def obj(z, fast=True):
    c, v = unpack(z, N, n); x = np.abs(c)
    m = aimed(c, v, G=Gfast, polish=not fast) if n > 1 else aimed(c, v)
    if m <= 1e-4 * x.sum():
        return 1e3 + max(0.0, -m)  # only m>0 populations matter
    return det_ratio(x, v, m)

best = 1e9; rec = None; log = []
t0 = time.time()
for t in range(nstart):
    for fam in fams:
        z0 = starts(fam, N, n, rng, ns=rng.choice([0.2, 0.5, 1.0]))
        r = minimize(obj, z0, method='Nelder-Mead', options=dict(maxiter=4000, maxfev=4000, xatol=1e-9, fatol=1e-13))
        r2 = minimize(obj, r.x, method='Powell', options=dict(maxiter=3000, xtol=1e-9, ftol=1e-13))
        z = r2.x if r2.fun < r.fun else r.x
        val = obj(z, fast=False)   # accurate m at the end
        log.append(dict(fam=fam, val=float(val)))
        if val < best:
            best = val; rec = z.copy()
            c, v = unpack(z, N, n); x = np.abs(c)
            print(f"  [{fam} t={t}] det/m^D = {val:.9f}   |c|/max = {np.round(x / x.max(), 3)}", flush=True)

c, v = unpack(rec, N, n); x = np.abs(c)
m = aimed(c, v)
rat, ev = Ek_ratios(x, v, m)
print(f"n={n} N={N}: min det(M+)/m^D = {best:.9f}; at minimiser E_k ratios = {np.round(rat, 6)}, eig/m = {np.round(ev / m, 4)}  ({time.time()-t0:.0f}s)")
json.dump(dict(n=n, N=N, seed=seed, best=float(best), Ek_ratios=rat.tolist(), eig_over_m=(ev / m).tolist(),
               c=[[float(a.real), float(a.imag)] for a in c], v_angles=np.angle(v).tolist(), log=log),
          open(f'FINAL_RUNS/ARCH/det_n{n}_N{N}_s{seed}.json', 'w'))
