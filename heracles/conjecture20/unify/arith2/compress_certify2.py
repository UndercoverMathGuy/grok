# Certified compression of uniform progression blocks {1..q}, larger q: same certificate as compress_certify.py
# (rigorous check of g(x) = sum_l b'_l (1-cos l x) - sum_{j<=q} (1-cos j x) >= 0 on [0, pi] in mpmath, rational b' rounded up),
# but the reweighted-LP search is warm-started on a seed set: the frequencies used by the certified dominators for q <= 48
# ({3,4,12,14,21,22,49,54,70} and friends) together with their dilations by q/24 and q/48.  Only the SEARCH changes;
# the certificate (function certify) is imported unchanged.
import numpy as np, sys, json
from scipy.optimize import linprog
from compress_certify import certify

SEED = [2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 17, 21, 22, 24, 36, 49, 54, 70]

def find_warm(q, s, ngrid=4000, iters=80, starts=6, seed=1):
    r = 1/s; Lmax = max(70, 3*q)
    xg = np.unique(np.concatenate([np.geomspace(0.05/q, 3.0/q, 400), np.linspace(3.0/q, np.pi, ngrid)]))
    L = np.arange(1, Lmax+1)
    Kx = 1-np.cos(np.outer(xg, L))
    rhs = (1-np.cos(np.outer(xg, np.arange(1, q+1)))).sum(1)
    sc = 1/np.maximum(rhs, 1e-12)
    A = -(Kx*sc[:, None])
    cand = set(SEED)
    for f in (q/24, q/48, q/36):
        cand |= {int(round(f*l)) for l in SEED}
    cand = {l for l in cand if 1 <= l <= Lmax}
    rng = np.random.default_rng(seed); best = (np.inf, None)
    for st in range(starts):
        w = np.full(Lmax, 1.7)
        idx = np.array(sorted(cand))-1
        w[idx] = 0.3 if st == 0 else rng.uniform(0.1, 0.6, len(idx))
        if st >= 3: w = w*rng.uniform(0.7, 1.3, Lmax)
        b = None
        for it in range(iters):
            res = linprog(w, A_ub=A, b_ub=-np.ones(len(xg)), bounds=(0, None), method='highs')
            if res.status != 0: break
            b = np.maximum(res.x, 0); wn = (b + 1e-9*max(b.max(), 1e-12))**(r-1)
            if np.allclose(wn, w, rtol=1e-7): break
            w = wn
        if b is None: continue
        b = b*(b > 1e-9*b.max()); Q = np.sum(b[b > 0]**r)
        if Q < best[0]: best = (Q, b.copy())
    return best[1]

if __name__ == '__main__':
    s = float(sys.argv[1]); qs = [int(a) for a in sys.argv[2].split(',')]; tag = sys.argv[3] if len(sys.argv) > 3 else 'warm'
    thr = (2/3)**(s-1); out = {}
    for q in qs:
        b = find_warm(q, s); wq = np.ones(q)
        res = None
        for eta in [1e-3, 3e-3, 1e-2, 3e-2]:
            res, bq = certify(q, s, b, wq, eta)
            if res is not None and res[0]: break
        if res is None: print(f"q={q}: S2 condition failed", flush=True); continue
        ok, rho, xs, ncell, worst, Lm, nsp = res
        print(f"s={s} uniform(warm) q={q}: certified={ok} eta={eta} rho_q <= {rho:.4f} (threshold {thr:.4f}, {'OK' if rho < thr else 'NOT below'}) "
              f"x_s={xs:.2e} cells={ncell} min cell lower bound={worst:.2e} |spt b'|={nsp} maxfreq={Lm}", flush=True)
        out[q] = dict(ok=ok, rho=rho, eta=eta, freqs=[int(l) for l in (np.nonzero(b)[0]+1)], bprime=[str(x) for x in bq])
        json.dump(out, open(f'compress_cert_{tag}_s{s}.json', 'w'), indent=1)
