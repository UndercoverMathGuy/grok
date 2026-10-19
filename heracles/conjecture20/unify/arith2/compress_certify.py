# Certified compression of a uniform progression block (p-uniform, dilation invariant).
# Claim certified for each q: there is b' >= 0 on integer frequencies 1..Lmax with
#   (i)  g(x) := sum_l b'_l (1 - cos l x) - sum_{j=1}^q (1 - cos j x) >= 0   for all real x,
#   (ii) ||b'|| := (sum b'^r)^s <= rho_q * q^s  with  rho_q < (2/3)^{s-1}.
# (i) is a statement about a trigonometric polynomial on [0, pi] (even, 2 pi periodic).  Then at EVERY prime p and every
# dilation d, the clean allocation d*b' (folded into F) dominates the coverage profile of the block d*{1..q} at every
# offset, and its cost is <= ||b'|| (folding merges frequencies, which can only lower sum (.)^r).
# Rigorous verification of (i):
#   * small x in [0, x_s]:  1-cos t in [t^2/2 - t^4/24, t^2/2]  gives g >= x^2/2 [S2' - S2 - S4' x^2/12] >= 0 for
#     x^2 <= 12 (S2' - S2)/S4'   (S2' = sum b' l^2, S2 = sum j^2, S4' = sum b' l^4; requires S2' > S2);
#   * x in [x_s, pi]: grid x_0 = x_s < x_1 < ... < x_N = pi;  on each cell  g >= min(g(x_i), g(x_{i+1})) - M2 h_i^2 / 8,
#     M2 = sum b' l^2 + sum j^2 >= sup |g''|;  g(x_i) evaluated in mpmath at 40 digits (error << margins).
# b' is first found by reweighted LP, then multiplied by (1+eta) and rounded UP to rationals with denominator 10^9.
import numpy as np, sys, json
from scipy.optimize import linprog
import mpmath as mp
mp.mp.dps = 40

def find(q, s, weights=None, Lmax=None, ngrid=4000, iters=60, starts=4, seed=0):
    r = 1/s; Lmax = Lmax or max(40, 6*q)
    wq = np.ones(q) if weights is None else np.asarray(weights, float)
    # grid: geometric near 0, uniform in bulk
    xg = np.unique(np.concatenate([np.geomspace(0.05/q, 3.0/q, 400), np.linspace(3.0/q, np.pi, ngrid)]))
    L = np.arange(1, Lmax+1)
    Kx = 1-np.cos(np.outer(xg, L))
    rhs = ((1-np.cos(np.outer(xg, np.arange(1, q+1))))*wq[None, :]).sum(1)
    # normalise rows near 0 to avoid tiny numbers dominating tolerance
    sc = 1/np.maximum(rhs, 1e-12)
    rng = np.random.default_rng(seed); best = (np.inf, None)
    for st in range(starts):
        w = np.ones(Lmax) if st == 0 else rng.uniform(0.3, 1.7, Lmax)
        b = None
        for it in range(iters):
            res = linprog(w, A_ub=-(Kx*sc[:, None]), b_ub=-np.ones(len(xg)), bounds=(0, None), method='highs')
            if res.status != 0: break
            b = np.maximum(res.x, 0); wn = (b + 1e-9*max(b.max(), 1e-12))**(r-1)
            if np.allclose(wn, w, rtol=1e-7): break
            w = wn
        if b is None: continue
        b = b*(b > 1e-9*b.max()); Q = np.sum(b[b > 0]**r)
        if Q < best[0]: best = (Q, b.copy())
    return best[1], wq

def certify(q, s, b, wq, eta):
    r = 1/s
    L = np.nonzero(b)[0]+1
    # rational rounding up: b'_l = ceil((1+eta) b_l * 1e9)/1e9
    bq = [mp.mpf(int(np.ceil((1+eta)*b[l-1]*1e9)))/mp.mpf(10**9) for l in L]
    J = list(range(1, q+1)); wJ = [mp.mpf(float(x)) for x in wq]
    S2p = sum(bl*l**2 for bl, l in zip(bq, L)); S2 = sum(wj*j**2 for wj, j in zip(wJ, J))
    S4p = sum(bl*l**4 for bl, l in zip(bq, L))
    if S2p <= S2: return None, 'S2'
    xs = mp.sqrt(12*(S2p - S2)/S4p)
    M2 = S2p + sum(wj*j**2 for wj, j in zip(wJ, J))
    def g(x):
        return sum(bl*(1-mp.cos(l*x)) for bl, l in zip(bq, L)) - sum(wj*(1-mp.cos(j*x)) for wj, j in zip(wJ, J))
    # adaptive grid from xs to pi
    x = xs; gx = g(x); ok = True; worst = mp.inf; ncell = 0
    while x < mp.pi:
        h = min(mp.mpf(0.02)*x, mp.mpf(np.pi)/600)
        while True:
            x2 = min(x + h, mp.pi); g2 = g(x2)
            lb = min(gx, g2) - M2*(x2-x)**2/8
            if lb >= 0 or h < mp.mpf(1e-9): break
            h = h/2
        if lb < 0: ok = False; worst = min(worst, lb); break
        worst = min(worst, lb); x, gx = x2, g2; ncell += 1
    cost = (sum(bl**r for bl in bq))**s
    rho = cost/(sum(wj**r for wj in wJ))**s
    return (ok, float(rho), float(xs), ncell, float(worst), int(L.max()), len(L)), bq

if __name__ == '__main__':
    s = float(sys.argv[1]); qs = [int(a) for a in sys.argv[2].split(',')]; kind = sys.argv[3] if len(sys.argv) > 3 else 'uniform'
    thr = (2/3)**(s-1); out = {}
    for q in qs:
        wts = None if kind == 'uniform' else [1 - j/(q+1) for j in range(1, q+1)]
        b, wq = find(q, s, weights=wts)
        res = None
        for eta in [1e-3, 3e-3, 1e-2, 3e-2]:
            res, bq = certify(q, s, b, wq, eta)
            if res is not None and res[0]: break
        if res is None: print(f"q={q}: S2 condition failed", flush=True); continue
        ok, rho, xs, ncell, worst, Lm, nsp = res
        print(f"s={s} {kind} q={q}: certified={ok} eta={eta} rho_q <= {rho:.4f} (threshold {thr:.4f}, {'OK' if rho < thr else 'NOT below'}) "
              f"x_s={xs:.2e} cells={ncell} min cell lower bound={worst:.2e} |spt b'|={nsp} maxfreq={Lm}", flush=True)
        out[q] = dict(ok=ok, rho=rho, eta=eta, freqs=[int(l) for l in (np.nonzero(b)[0]+1)], bprime=[str(x) for x in bq])
    json.dump(out, open(f'compress_cert_{kind}_s{s}.json', 'w'), indent=1)
