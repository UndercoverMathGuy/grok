# Referee check of R1 (sawtooth counterexample to PB2) and R3 (PBD#), independent code.
import numpy as np, itertools, sys
from fractions import Fraction as Fr
from scipy.optimize import minimize

def planes(p):
    X = np.array(list(itertools.product(range(p), repeat=3)))
    return X, X.sum(1) % p

def mM_all(f, p, X, S):
    F = f[0][X[:,0]] + f[1][X[:,1]] + f[2][X[:,2]]
    mins = np.array([F[S == c].min() for c in range(p)]); maxs = np.array([F[S == c].max() for c in range(p)])
    return -mins, maxs           # m_c, M_c'

def bounds_exact(fs, p):
    # fs: lists of Fractions, mean zero. returns (V, b1, b2-ish as float, b3)
    V = [sum(x*x for x in f)/p for f in fs]; Vt = sum(V)
    A = [sorted(f, reverse=True) for f in fs]
    cross = sum(sum(A[i][a]*A[j][p-1-a] for a in range(p)) for i, j in [(0,1),(0,2),(1,2)])
    b1 = Vt + Fr(2, p*(p+1))*cross
    b2 = float(Vt) - 2/(p+1)*sum(float(V[i]*V[j])**0.5 for i, j in [(0,1),(0,2),(1,2)])
    b3 = Fr(p-1, p+1)*Vt
    return Vt, b1, b2, b3

# ---- R1: sawtooth, exact
print("R1 sawtooth exact:")
for p in [3, 5, 7, 11, 13, 17]:
    r = [Fr((-x) % p) - Fr(p-1, 2) for x in range(p)]
    X, S = planes(p)
    F = {}
    vals = {c: [] for c in range(p)}
    for x in itertools.product(range(p), repeat=3):
        vals[sum(x) % p].append(r[x[0]] + r[x[1]] + r[x[2]])
    m = {c: -min(vals[c]) for c in range(p)}; M = {c: max(vals[c]) for c in range(p)}
    V, b1, b2, b3 = bounds_exact([r, r, r], p)
    ratio = m[1]*M[2]/V
    worst = min(m[c]*M[cp]/V for c in range(p) for cp in range(p))
    print(f"  p={p}: m_1={m[1]} M_2={M[2]} V={V} ratio={ratio} (p-1)/(p+1)={Fr(p-1,p+1)}; b1={b1/V} b3/V={b3/V}; min over all (c,c')={worst}")

# ---- R3 exhaustive at p=3 over integer functions in [-K,K]
def exhaustive(p, K, sample=None, seed=0):
    rng = np.random.default_rng(seed)
    base = [t for t in itertools.product(range(-K, K+1), repeat=p) if sum(t) == 0 and any(t)]
    X, S = planes(p)
    worst = (np.inf, None); worst3 = np.inf; nviol = 0; ntest = 0
    triples = itertools.product(base, repeat=3) if sample is None else (tuple(base[k] for k in rng.integers(len(base), size=3)) for _ in range(sample))
    for tr in triples:
        f = [np.array(t, dtype=np.int64) for t in tr]
        m, M = mM_all(f, p, X, S)
        fs = [[Fr(int(v)) for v in t] for t in tr]
        V, b1, b2, b3 = bounds_exact(fs, p)
        prod_min = Fr(int(m.min()))*Fr(int(M.min()))     # min over (c,c') of m_c M_c' = (min m)(min M)
        ntest += 1
        if prod_min < b1: nviol += 1
        rr = prod_min/V
        if rr < worst[0]: worst = (rr, tr)
        worst3 = min(worst3, float(prod_min - b1)/float(V))
    return ntest, nviol, worst, worst3

for (p, K, samp) in [(3, 3, None), (5, 2, 30000), (7, 1, 20000)]:
    nt, nv, w, w3 = exhaustive(p, K, samp)
    print(f"R3 p={p} K={K}: tested {nt}, violations of mM>=b1: {nv}; min mM/V = {w[0]} = {float(w[0]):.6f} vs (p-1)/(p+1)={(p-1)/(p+1):.6f}; min (mM-b1)/V = {w3:.3e}; argmin {w[1]}")

# ---- adversarial (Nelder-Mead on exact non-smooth ratio) at p=3,5,7
def ratio(z, p, X, S, c, cp):
    f = [z[k*p:(k+1)*p] - z[k*p:(k+1)*p].mean() for k in range(3)]
    V = sum((fi**2).mean() for fi in f)
    if V < 1e-12: return 10.0
    F = f[0][X[:,0]] + f[1][X[:,1]] + f[2][X[:,2]]
    return (-F[S == c].min())*(F[S == cp].max())/V
rng = np.random.default_rng(1)
for p in [3, 5, 7]:
    X, S = planes(p); best = np.inf
    for t in range(60 if p < 7 else 30):
        c, cp = int(rng.integers(p)), int(rng.integers(p))
        z = rng.normal(size=3*p)
        if t % 3 == 0: z = np.concatenate([((-np.arange(p)) % p)*1.0]*3) + 0.3*rng.normal(size=3*p)
        res = minimize(ratio, z, args=(p, X, S, c, cp), method='Nelder-Mead', options=dict(maxiter=6000, xatol=1e-10, fatol=1e-12))
        best = min(best, res.fun)
    print(f"adversarial NM p={p}: min mM/V found {best:.6f}  bound (p-1)/(p+1) = {(p-1)/(p+1):.6f}")
