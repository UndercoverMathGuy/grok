"""arch_local2.py -- adversarial second-order test of (PW) at B = I + eps H with the probe's second-order
multi-affine correction chosen optimally (LP).  Value(H) = min_g max_v [ (2/D^2)(Im R)^2 - tau + g ]  restricted to
the free part; (PW) holds to second order in direction H iff Value(H) <= 0.  tau = ||H_off||^2/D (diag H = 0, worst).
Precisely: need g (zero-mean multi-affine real trig poly) with  (2/D^2)(Im R(v))^2 + g(v) <= tau  for all v.
usage: arch_local2.py n nstart seed
"""
import numpy as np, sys, json
from scipy.optimize import minimize, linprog
from itertools import combinations

n = int(sys.argv[1]); nst = int(sys.argv[2]); seed = int(sys.argv[3])
D = n + 1; rng = np.random.default_rng(seed)
intervals = [(k, kp) for k in range(D) for kp in range(k + 1, D)]
G = {1: 2000, 2: 120, 3: 28, 4: 12}[n]
th = 2 * np.pi * np.arange(G) / G
grid = np.stack(np.meshgrid(*([th] * n), indexing='ij'), -1).reshape(-1, n)
# interval characters on the grid
CHR = np.stack([np.exp(1j * grid[:, I[0]:I[1]].sum(1)) for I in intervals], 1)   # (P, #intervals)
subsets = [T for k in range(1, n + 1) for T in combinations(range(n), k)]
cols = []
for T in subsets:
    ph = grid[:, list(T)].sum(1); cols += [np.cos(ph), np.sin(ph)]
A = np.stack(cols, 1); nc = A.shape[1]
cobj = np.zeros(nc + 1); cobj[-1] = 1
Aub = np.concatenate([A, -np.ones((len(grid), 1))], 1)

def value(hvec):
    h = hvec[0::2] + 1j * hvec[1::2]
    nrm = float((np.abs(h) ** 2).sum())
    if nrm < 1e-12: return -1.0
    h = h / np.sqrt(nrm)
    R = CHR @ h
    f = (2 / D ** 2) * (R.imag ** 2)          # need f + g <= tau = 1/D
    r = linprog(cobj, A_ub=Aub, b_ub=1 / D - f, bounds=[(None, None)] * (nc + 1), method='highs')
    return float(r.fun)   # = min_g max_v [f + g - tau]  (<= 0 means PASS)

best = -1e9; bh = None
for s in range(nst):
    h0 = rng.normal(size=2 * len(intervals))
    r = minimize(lambda h: -value(h), h0, method='Nelder-Mead', options=dict(maxiter=600, xatol=1e-6, fatol=1e-9))
    if -r.fun > best: best = -r.fun; bh = r.x
    print(f"  start {s}: sup_H Value so far {best:+.3e}", flush=True)
h = bh[0::2] + 1j * bh[1::2]; h /= np.linalg.norm(h)
print(f"n={n}: sup over H of the second-order LP value = {best:+.3e}  ->  {'PASS (second-order (PW) holds)' if best <= 1e-6 else 'FAIL'}")
print("   worst H_off:", {str(I): np.round(hh, 4) for I, hh in zip(intervals, h)})
json.dump(dict(n=n, sup_value=float(best), worstH=[[float(a.real), float(a.imag)] for a in h], G=G), open(f'FINAL_RUNS/ARCH/local2_n{n}_s{seed}.json', 'w'))
