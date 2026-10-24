# DETECTION LP (n = 2, class C, no payment constraint): how flat can the total deficit be?
#   D(x, delta) = sum_{l in S, T} Re( a_{l,T} c_l(delta) chi_{l,T}(x) ),  chi_{l,T}(x) = zeta_l^{sum_{t in T} x_t},
#   c_l(delta) = 1 - rho^{-l delta};  T in {(1,0), (0,1), (1,1)}.  For phases psi fixed, a = r e^{i psi}, r >= 0:
#     rho(psi) = min_r  max_{x, delta in +-B} ( -D(x,delta) )   s.t.  sum_{l,T} r_{l,T} w_l = 1,
#   w_l = min_{delta in B} |c_l(delta)| / 2   (single-character detection weight: -min_x D(x,delta) >= r_{l,T}|c_l(delta)|/2
#   for EVERY (l,T) and delta, by the test 1 - Re(e^{i th} chi)); so rho >= 1/(3|S|) trivially, rho <= 1 at a vertex.
#   rho* = inf_psi rho(psi): small rho* = flat-bottomed compensated fluctuations exist (in principle).
# Usage: python detect_lp.py p b Smode nphase seed [n]   (characters 2l*1_T, T nonempty subset of [n])
import sys, json, time
import numpy as np
from scipy.optimize import linprog
import hlib as H

p = int(sys.argv[1]); bexp = float(sys.argv[2]); Smode = sys.argv[3]; nph = int(sys.argv[4]); seed = int(sys.argv[5])
n = int(sys.argv[6]) if len(sys.argv) > 6 else 2
variant = sys.argv[7] if len(sys.argv) > 7 else 'pair'   # 'pair': +-B ; 'single': one offset B[0] only;
#   'ghost': +-B, plus FREE (uncharged, w = 0) fluctuation columns at every frequency outside spt(y): zero-mean compensators
s = (n + 1) / bexp; F = (p - 1) // 2
import itertools
rng = np.random.default_rng(seed)


def red(l):
    l %= p
    return min(l, p - l)


if Smode == 'F': S = list(range(1, F + 1))
elif Smode == 'Sstar':
    G0, y0, _ = H.programme(p, s); S = [l for l in range(1, F + 1) if y0[l - 1] > 0]
else:
    d, K, a = [int(v) for v in Smode[2:].split('_')]; S = sorted(set(red(a + k * d) for k in range(K)) - {0})
GS, yS, B = H.programme(p, s, S=S, restarts=40)
sup = [l for l in S if yS[l - 1] > 0]
Bfull = sorted(set(B) | set(p - d for d in B)) if variant in ('pair', 'ghost') else [B[0]]
if variant == 'single': B = [B[0]]
X = np.indices((p,) * n).reshape(n, -1).T
Ts = [T for T in itertools.product([0, 1], repeat=n) if any(T)]
cols = [(l, T) for l in sup for T in Ts]
ghosts = [l for l in range(1, F + 1) if l not in sup] if variant == 'ghost' else []
cols += [(l, T) for l in ghosts for T in Ts]
chi = np.array([np.exp(2j * np.pi * 2 * l * (X @ np.array(T)) / p) for (l, T) in cols])        # (ncol, p^2)
cdl = {l: {d: 1 - np.exp(-2j * np.pi * l * d / p) for d in Bfull} for l in range(1, F + 1)}
w = np.array([min(abs(cdl[l][d]) for d in B) / 2 if l in sup else 0.0 for (l, T) in cols])


def rho_of(psi):
    # constraint rows: for each (x, delta): -D = -sum_k r_k Re(e^{i psi_k} c_l(delta) chi_k(x)) <= tau
    rows = []
    for d in Bfull:
        cvec = np.array([cdl[l][d] for (l, T) in cols])
        M = np.real((np.exp(1j * psi) * cvec)[:, None] * chi)                                   # (ncol, p^2)
        rows.append(-M.T)
    A = np.vstack(rows); nc = len(cols)
    Aub = np.hstack([A, -np.ones((A.shape[0], 1))])
    res = linprog(np.r_[np.zeros(nc), 1.0], A_ub=Aub, b_ub=np.zeros(A.shape[0]),
                  A_eq=np.r_[w, 0.0][None, :], b_eq=[1.0], bounds=[(0, None)] * nc + [(None, None)], method='highs')
    return res.fun, res.x[:nc]


t0 = time.time()
best = (np.inf, None, None)
for it in range(nph):
    if it == 0: psi = np.zeros(len(cols))
    elif it == 1: psi = -np.pi / 2 * np.ones(len(cols))
    else: psi = rng.uniform(0, 2 * np.pi, len(cols))
    v, rr = rho_of(psi)
    # local improvement of the phases (coordinate perturbations)
    step = 0.5
    for sweep in range(6):
        for k in range(len(cols)):
            for dlt in (step, -step):
                ps = psi.copy(); ps[k] += dlt
                v2, r2 = rho_of(ps)
                if v2 < v - 1e-12: psi, v, rr = ps, v2, r2
        step *= 0.5
    if v < best[0]: best = (v, psi.copy(), rr.copy())
    print(json.dumps(dict(it=it, rho=v, support_used=int((rr > 1e-9).sum()))), flush=True)
v, psi, rr = best
used = [(cols[k], round(float(rr[k] * w[k]), 4), round(float(psi[k]) % (2 * np.pi), 3)) for k in range(len(cols)) if rr[k] > 1e-9]
print(f"# SUMMARY detect[{variant}] p={p} n={n} b={bexp} S={Smode}:{S} supp={sup} B={B} rho*={v:.6f} trivial lower bound 1/#chars={1 / len(cols):.4f} rho*/2={v / 2:.4f} "
      f"used={used} time={time.time() - t0:.0f}s", flush=True)
