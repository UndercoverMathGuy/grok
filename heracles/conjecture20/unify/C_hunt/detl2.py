# Can class-C fluctuations be FLAT-BOTTOMED below the L2 aggregate at a pair of offsets (+-delta)?
#   D_delta(x) = sum_{l in S, T in Lam} Re(a_{l,T} c_l(delta) chi_{l,T}(x)),  c_l(delta) = 1 - rho^{-l delta}.
#   pair detection  P(a) = max( max_x -D_delta(x), max_x -D_{-delta}(x) )   (convex, piecewise linear in (Re a, Im a))
#   L2(a) = ||D_delta||_2 = (sum |a_{l,T}|^2 kappa_l(delta))^{1/2}.
#   Lemma L2: P >= L2 whenever Lam is consistent.  Here: rho2(p, n, Lam) = min_a P(a) / L2(a), any Lam.
# Method: convex-concave procedure; each step an exact LP:  min t  s.t. -D_{+-delta}(x; a) <= t (all x),
#   <a0, a>_kappa = 1  (a0 = current point, L2(a0) = 1)  =>  L2(a) >= 1, so P(a)/L2(a) <= t; iterate. Many random starts.
#   avg mode (argv[7] == 'avg'): P_avg(a) = ( max_x -D_delta + max_x -D_{-delta} ) / 2  (the version that bounds h for complex A_l)
# Usage: python detl2.py p n nstarts seed [Lam] [S] [avg]   Lam: all (default) | three (n=2 only: all three) ; S: F | Sstar
import sys, itertools, json, time
import numpy as np
from scipy.optimize import linprog

p = int(sys.argv[1]); n = int(sys.argv[2]); nst = int(sys.argv[3]); seed = int(sys.argv[4])
Lam = sys.argv[5] if len(sys.argv) > 5 else 'all'
Smode = sys.argv[6] if len(sys.argv) > 6 else 'F'
AVG = len(sys.argv) > 7 and 'avg' in sys.argv[7]
KSEED = len(sys.argv) > 7 and 'kseed' in sys.argv[7]
rng = np.random.default_rng(seed)
F = (p - 1) // 2; d0 = 1
if Smode == 'F': S = list(range(1, F + 1))
else:
    import hlib2 as H2
    R = H2.prog_bb(p, (n + 1) / 2.0, maxnodes=3000); S = R['supp']
X = np.indices((p,) * n).reshape(n, -1).T
Ts = [T for T in itertools.product([0, 1], repeat=n) if any(T)]
cols = [(l, T) for l in S for T in Ts]; nc = len(cols)
chi = np.array([np.exp(2j * np.pi * 2 * l * (X @ np.array(T)) / p) for (l, T) in cols])        # (nc, N)
kap = np.array([1 - np.cos(2 * np.pi * l * d0 / p) for (l, T) in cols])
rows = []
for d in (d0, p - d0):
    cv = np.array([1 - np.exp(-2j * np.pi * l * d / p) for (l, T) in cols])
    E = cv[:, None] * chi                                                                       # (nc, N)
    # D = sum Re(a E) = sum (Re a Re E - Im a Im E)  ->  -D <= t
    rows.append(np.hstack([-E.real.T, E.imag.T]))
M = np.vstack(rows)                                                                             # rows: -D values per (x, sign)
NR = M.shape[0]


NX = X.shape[0]


def P_of(z):
    v = M @ z
    return 0.5 * (v[:NX].max() + v[NX:].max()) if AVG else v.max()


def L2_of(z):
    a2 = z[:nc] ** 2 + z[nc:] ** 2
    return np.sqrt(np.sum(a2 * kap))


t0 = time.time(); best = (np.inf, None)
for it in range(nst):
    if KSEED and it % 2 == 0:      # magnitudes 1/kappa_l (the structure of the p = 5, 7, 11 minimisers), random phases
        mag = 1 / kap; ph = rng.uniform(0, 2 * np.pi, nc)
        z = np.r_[mag * np.cos(ph), mag * np.sin(ph)]
    else:
        z = rng.standard_normal(2 * nc) * (rng.random(2 * nc) < rng.uniform(0.2, 1.0))
    if not z.any(): z[0] = 1.0
    z /= L2_of(z)
    ratio = P_of(z)
    for k in range(60):
        w = np.r_[z[:nc] * kap, z[nc:] * kap]                                                   # <a0, a>_kappa = w . z
        if AVG:
            c = np.r_[np.zeros(2 * nc), 0.5, 0.5]
            A_ub = np.vstack([np.hstack([M[:NX], -np.ones((NX, 1)), np.zeros((NX, 1))]),
                              np.hstack([M[NX:], np.zeros((NX, 1)), -np.ones((NX, 1))])])
            Aeq = np.r_[w, 0.0, 0.0][None, :]; nb = 2
        else:
            c = np.r_[np.zeros(2 * nc), 1.0]
            A_ub = np.hstack([M, -np.ones((NR, 1))]); Aeq = np.r_[w, 0.0][None, :]; nb = 1
        res = linprog(c, A_ub=A_ub, b_ub=np.zeros(NR), A_eq=Aeq, b_eq=[1.0],
                      bounds=[(None, None)] * (2 * nc + nb), method='highs')
        if res.status != 0: break
        zn = res.x[:2 * nc]; zn /= L2_of(zn)
        rn = P_of(zn)
        if rn < ratio - 1e-12: z, ratio = zn, rn
        else: break
    if ratio < best[0]: best = (ratio, z.copy())
z = best[1]
a = z[:nc] + 1j * z[nc:]
used = sorted(set(cols[k][1] for k in range(nc) if abs(a[k]) > 1e-6 * np.abs(a).max()))
print(json.dumps(dict(p=p, n=n, S=S, Lam=Lam, avg=AVG, starts=nst, rho2=best[0], chars_used=[list(T) for T in used],
                      n_used=int((np.abs(a) > 1e-6 * np.abs(a).max()).sum()), time=round(time.time() - t0, 1))), flush=True)
np.savez(f"out2/detl2_p{p}_n{n}_{Smode}{'_avg' if AVG else ''}.npz", a=a, cols=np.array([(l,) + tuple(T) for (l, T) in cols]), rho2=best[0])
