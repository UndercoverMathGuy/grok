# Same certificate as compress_certify.py, two changes for larger q:
#  * the block profile P_q(x) = sum_{j<=q} (1 - cos j x) is evaluated by the exact closed form
#      P_q(x) = q + 1/2 - sin((q+1/2) x) / (2 sin(x/2))      (0 < x <= pi),
#    in 40-digit mpmath (so each evaluation costs O(|spt b'|) instead of O(q));
#  * warm-started search (compress_certify2.find_warm).
# Everything else (small-x Taylor window, cell bound min(g(x_i), g(x_{i+1})) - M2 h^2/8 with M2 = S2' + S2 >= sup|g''|,
# rational rounding up of b') is as in compress_certify.py.
import numpy as np, sys, json
import mpmath as mp
from compress_certify2 import find_warm
mp.mp.dps = 40

def certify_fast(q, s, b, eta):
    r = 1/s
    L = np.nonzero(b)[0]+1
    bq = [mp.mpf(int(np.ceil((1+eta)*b[l-1]*1e9)))/mp.mpf(10**9) for l in L]
    Li = [int(l) for l in L]
    S2p = sum(bl*l**2 for bl, l in zip(bq, Li)); S2 = mp.mpf(q*(q+1)*(2*q+1))/6
    S4p = sum(bl*l**4 for bl, l in zip(bq, Li))
    if S2p <= S2: return None, bq
    xs = mp.sqrt(12*(S2p - S2)/S4p)
    M2 = S2p + S2
    half = mp.mpf(1)/2
    def g(x):
        return sum(bl*(1-mp.cos(l*x)) for bl, l in zip(bq, Li)) - (q + half - mp.sin((q+half)*x)/(2*mp.sin(x/2)))
    x = xs; gx = g(x); ok = True; worst = mp.inf; ncell = 0
    while x < mp.pi:
        h = min(mp.mpf(0.02)*x, mp.pi/600)
        while True:
            x2 = min(x + h, mp.pi); g2 = g(x2)
            lb = min(gx, g2) - M2*(x2-x)**2/8
            if lb >= 0 or h < mp.mpf(1e-10): break
            h = h/2
        if lb < 0: ok = False; worst = min(worst, lb); break
        worst = min(worst, lb); x, gx = x2, g2; ncell += 1
    cost = (sum(bl**r for bl in bq))**s
    rho = cost/mp.mpf(q)**s
    return (ok, float(rho), float(xs), ncell, float(worst), int(L.max()), len(L)), bq

if __name__ == '__main__':
    s = float(sys.argv[1]); qs = [int(a) for a in sys.argv[2].split(',')]; tag = sys.argv[3]
    thr = (2/3)**(s-1); out = {}
    for q in qs:
        b = find_warm(q, s)
        res = None
        for eta in [1e-3, 3e-3, 1e-2, 3e-2]:
            res, bq = certify_fast(q, s, b, eta)
            if res is not None and res[0]: break
        if res is None: print(f"s={s} q={q}: S2 condition failed", flush=True); continue
        ok, rho, xs, ncell, worst, Lm, nsp = res
        cover_from = q*(min(rho, thr)/thr)**(1/s)
        print(f"s={s} uniform q={q}: certified={ok} eta={eta} rho_q <= {rho:.4f} (threshold {thr:.4f}, {'OK' if (ok and rho < thr) else 'NOT below'}) "
              f"covers q' in [{int(np.ceil(cover_from))},{q}] by monotonicity; x_s={xs:.2e} cells={ncell} min cell lb={worst:.2e} |spt b'|={nsp} maxfreq={Lm}", flush=True)
        out[q] = dict(ok=ok, rho=rho, eta=eta, freqs=[int(l) for l in (np.nonzero(b)[0]+1)], bprime=[str(x) for x in bq])
        json.dump(out, open(f'compress_cert_{tag}_s{s}.json', 'w'), indent=1)
