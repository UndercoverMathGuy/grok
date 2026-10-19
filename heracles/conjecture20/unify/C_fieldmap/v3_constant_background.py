# v3: Theorem S (one coupled feature against a constant real background).
# Network = feature l (any class-C atom system) + IF blocks at m != l with REAL amplitudes A_m (any sign).
#  (S1) exact formula  h = min_{d in F} [ b(d) + kappa_l(d) * c_{k(l d)} ],  b(d) = sum_{m != l} A_m kappa_m(d),
#       c_k = min_x cm_k(Phi_l(x)),  cm_k(z) = Re z - cot(pi k/p) |Im z|,  k(j) = reduced residue of +-j in F.
#  (S2) monotonicity c_1 <= c_2 <= ... (cones widen).
#  (S3) sharpness: the hyperplane background b = -G e_{d0} + t 1 (in span{kappa_m : m != l}) makes
#       h -> b(d0) + kappa_l(d0) c_{k(l d0)} for G large, for every d0.
import numpy as np
from fmlib import *

rng = np.random.default_rng(11)
out = []
def say(*a):
    s = ' '.join(str(x) for x in a); print(s); out.append(s)

err1 = 0; mono_bad = 0; ncase = 0
for (p, n) in [(5, 2), (7, 2), (11, 2), (5, 3), (7, 3), (13, 2)]:
    F = (p - 1) // 2
    for trial in range(10):
        l = int(rng.integers(1, F + 1))
        m = int(rng.integers(1, 5))
        C = rng.normal(size=m) + 1j * rng.normal(size=m); xi = np.exp(1j * rng.uniform(0, TAU, (m, n)))
        if trial % 2:
            C0, xi0 = if_atoms(n, 2.0); C = np.concatenate([C0, 0.4 * C]); xi = np.concatenate([xi0, xi])
        feats = {l: (C, xi)}
        Am = {}
        for mm in range(1, F + 1):
            if mm == l: continue
            Am[mm] = rng.normal() * 2
            feats[mm] = if_atoms(n, Am[mm])
        net = CNet(p, n, feats)
        h_direct = net.h()
        d = np.arange(1, F + 1)
        b = sum(Am[mm] * kap(p, mm, d) for mm in Am)
        Phi = net.Phi[l]
        ck = np.array([cm_k(Phi, p, k).min() for k in range(1, F + 1)])
        mono_bad += int(np.any(np.diff(ck) < -1e-12))
        kk = np.array([kred(p, l * dd) for dd in d])
        h_formula = np.min(b + kap(p, l, d) * ck[kk - 1])
        err1 = max(err1, abs(h_direct - h_formula) / (abs(h_direct) + 1e-12)); ncase += 1
say(f'(S1) exact formula over {ncase} networks: max rel error {err1:.2e}')
say(f'(S2) violations of c_1 <= c_2 <= ... : {mono_bad}')

# ---------- (S3) sharpness via hyperplane backgrounds
for (p, n) in [(5, 2), (7, 2), (11, 2), (13, 3)]:
    F = (p - 1) // 2
    l = 1
    C = rng.normal(size=3) + 1j * rng.normal(size=3); xi = np.exp(1j * rng.uniform(0, TAU, (3, n)))
    X = grid(p, n); Phi = field(p, X, l, C, xi)
    ck = np.array([cm_k(Phi, p, k).min() for k in range(1, F + 1)])
    Kmat = kappa_mat(p)[:F]                                   # rows d=1..F, cols m=1..F
    others = [mm for mm in range(1, F + 1) if mm != l]
    for d0 in range(1, F + 1):
        if F == 1: break
        # solve for real A_m (m != l) with sum A_m kappa_m(d) = -G [d=d0] + t, choosing t so it is solvable
        Ko = Kmat[:, [mm - 1 for mm in others]]                # F x (F-1)
        G = 1e3 * (1 + np.abs(ck).max())
        # least-squares with t unknown: unknowns (A, t): Ko A - t 1 = -G e_d0
        Mx = np.hstack([Ko, -np.ones((F, 1))]); rhs = -G * (np.arange(1, F + 1) == d0)
        sol = np.linalg.solve(Mx, rhs); A = sol[:-1]; t = sol[-1]
        feats = {l: (C, xi)}
        for j, mm in enumerate(others): feats[mm] = if_atoms(n, A[j])
        net = CNet(p, n, feats)
        pred = (t - G) + kap(p, l, d0) * ck[kred(p, l * d0) - 1]
        say(f'(S3) p={p} n={n} d0={d0} (direction k={kred(p, l*d0)}): h={net.h():.6f}  predicted={pred:.6f}  '
            f'c_k/c_1={ck[kred(p,l*d0)-1]:.4f}/{ck[0]:.4f}')
open('v3_constant_background.out', 'w').write('\n'.join(out) + '\n')
