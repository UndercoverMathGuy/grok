# Falsification search for (J), paired joint L1 detection in class C, at ONE offset delta0 and its negative:
#   (J)  max_{sign} ( - min_x D(x, sign*delta0) ) >= (1/2) sum_{l,T} |a_{l,T}| |c_l(delta0)|,
#   D(x, delta) = sum_{l in F} sum_{T nonempty} Re( a_{l,T} c_l(delta) zeta_l^{sum_{t in T} x_t} ),  x in Z_p^n.
# S = F is the strongest test (any S is the special case of zero coefficients). For fixed phases psi (a = r e^{i psi})
# the LP  min_r max_{x, sign} (-D)  s.t. sum r |c_l(delta0)|/2 = 1  gives rho(psi); (J) <=> rho(psi) >= 1 for all psi.
# Variant 'single' drops the pairing (one sign only).  Variant 'cplx' is the FIELD form (J'): clean amplitudes A_l = |A_l| e^{i alpha_l}
#   are added as LP variables |A_l| >= 0 (phases alpha searched with psi); rows: sum_l |A_l| (Re(e^{i alpha_l} c_l(s d0)) - kappa_l(d0))
#   + D(x, s d0) >= -tau.  (J') is what the reduction J'+P => class-C RD needs.  Usage: python jtest.py p n nphase seed [pair|single|cplx]
import sys, json, time, itertools
import numpy as np
from scipy.optimize import linprog

p = int(sys.argv[1]); n = int(sys.argv[2]); nph = int(sys.argv[3]); seed = int(sys.argv[4])
variant = sys.argv[5] if len(sys.argv) > 5 else 'pair'
F = (p - 1) // 2; rng = np.random.default_rng(seed)
X = np.indices((p,) * n).reshape(n, -1).T
Ts = [T for T in itertools.product([0, 1], repeat=n) if any(T)]
cols = [(l, T) for l in range(1, F + 1) for T in Ts]
chi = np.array([np.exp(2j * np.pi * 2 * l * (X @ np.array(T)) / p) for (l, T) in cols])
t0 = time.time()
worst = (np.inf, None)
# S = F: dilation l -> d l, delta -> d^{-1} delta permutes the features, so rho*(delta0) does not depend on delta0.
for d0 in [1]:
    signs = [d0, p - d0] if variant in ('pair', 'cplx') else [d0]
    cvecs = [np.array([1 - np.exp(-2j * np.pi * l * d / p) for (l, T) in cols]) for d in signs]
    w = np.abs(cvecs[0]) / 2

    kap0 = np.array([1 - np.cos(2 * np.pi * l * d0 / p) for l in range(1, F + 1)])
    cl = [np.array([1 - np.exp(-2j * np.pi * l * d / p) for l in range(1, F + 1)]) for d in signs]

    def rho_of(psi):
        A = np.vstack([-np.real((np.exp(1j * psi[:len(cols)]) * cv)[:, None] * chi).T for cv in cvecs])
        nA = 0
        if variant == 'cplx':
            al = psi[len(cols):]; nA = F
            blocks = []
            for q in range(len(signs)):
                dev = np.real(np.exp(1j * al) * cl[q]) - kap0                     # (F,)
                blocks.append(np.tile(-dev, (chi.shape[1], 1)))
            A = np.hstack([A, np.vstack(blocks)])
        nv = len(cols) + nA
        res = linprog(np.r_[np.zeros(nv), 1.0], A_ub=np.hstack([A, -np.ones((A.shape[0], 1))]),
                      b_ub=np.zeros(A.shape[0]), A_eq=np.r_[w, np.zeros(nA), 0.0][None, :], b_eq=[1.0],
                      bounds=[(0, None)] * nv + [(None, None)], method='highs')
        return res.fun, res.x[:nv]
    best = (np.inf, None, None)
    for it in range(nph):
        npar = len(cols) + (F if variant == 'cplx' else 0)
        psi = np.zeros(npar) if it == 0 else rng.uniform(0, 2 * np.pi, npar)
        if variant == 'cplx' and it > 0: psi[len(cols):] = rng.normal(0, 0.5, F)
        v, r = rho_of(psi); step = 0.6
        for sweep in range(5):
            act = [k for k in range(len(cols)) if r[k] > 1e-9] + list(rng.choice(len(cols), min(len(cols), 6), replace=False))
            if variant == 'cplx': act += list(range(len(cols), len(cols) + F))
            for k in set(act):
                for dl in (step, -step):
                    ps = psi.copy(); ps[k] += dl; v2, r2 = rho_of(ps)
                    if v2 < v - 1e-12: psi, v, r = ps, v2, r2
            step *= 0.5
        if v < best[0]: best = (v, psi.copy(), r.copy())
    v, psi, r = best
    used = [(cols[k], round(float(r[k] * w[k]), 4)) for k in range(len(cols)) if r[k] > 1e-9]
    if variant == 'cplx': used.append(('|A|', np.round(r[len(cols):], 4).tolist(), 'alpha', np.round(psi[len(cols):], 3).tolist()))
    print(json.dumps(dict(p=p, n=n, variant=variant, delta0=d0, rho=v, J_holds=bool(v >= 1 - 1e-9), nused=len(used),
                          used=used[:12]), default=str), flush=True)
    if v < worst[0]: worst = (v, d0)
print(f"# SUMMARY jtest[{variant}] p={p} n={n}: min_delta0 rho* = {worst[0]:.6f} at delta0={worst[1]} "
      f"((J) needs >= 1) time={time.time() - t0:.0f}s", flush=True)
