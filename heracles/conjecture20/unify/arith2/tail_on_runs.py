# For every near-optimal allocation on file (relations3/large_*.json and supports.json), s=1.5:
# efficiency e = m(a)/(Gamma_ref ||a||), kappa-mass of the support at the binding offsets (min and max over the binding set),
# and the TDP ratio Phi_TDP(a)/Gamma_ref.  Tail-free (kappa-mass <= 1/c^2 at some binding d) means Phi_TDP = e*Gamma_ref.
import numpy as np, json, os
from lib import *
s = 1.5; r, c, tp = consts(s)
rows = []
d = json.load(open(os.path.join(REL3, 'supports.json')))
for p in sorted(map(int, d)):
    if p < 61: continue
    e = d[str(p)]; F = (p-1)//2; y = np.zeros(F); y[np.array(e['S'])-1] = e['y']; rows.append((p, e['Gamma'], [y]))
for p in [2003, 5003, 10007, 30011, 100003]:
    fn = os.path.join(REL3, f'large_{p}.json')
    if not os.path.exists(fn): continue
    D = json.load(open(fn)); F = (p-1)//2; G = max(x['Gamma'] for x in D['runs']); ys = []
    for run in D['runs']:
        y = np.zeros(F); y[np.array(run['S'])-1] = run['y']; ys.append(y)
    rows.append((p, G, ys))
for p, G, ys in rows:
    K = kmat(p); out = []
    for y in ys:
        cov = K @ y; m = cov.min(); B = np.nonzero(cov <= m*(1+1e-7))[0]
        S = y > 0
        km = [K[b, S].sum() for b in B]
        tdp = min((y*K[dd]).sum() + c*tp*tail_bound(y, K[dd], c) for dd in range(K.shape[0]))
        out.append((S.sum(), m/(G*norm_r(y, s)), min(km), max(km), tdp/(G*norm_r(y, s))))
    out.sort(key=lambda x: -x[1])
    for o in out[:4]:
        print(f"p={p:6d} K={o[0]:3d} e={o[1]:.4f}  kappa-mass on binding set: [{o[2]:.2f},{o[3]:.2f}]  Phi_TDP/Gamma={o[4]:.4f}", flush=True)
