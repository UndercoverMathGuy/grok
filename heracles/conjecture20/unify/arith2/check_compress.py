# Sanity check of the compression lemma (relations case, n = 2, s = 3/2) on exact Z_p models:
#   cheap set L' = C' + B, B = d*{1..q} (folded into F) with constant IF price abar, max_delta kappa_{C'}(delta) <= 1/c^2;
#   A_l = a_l + c v_l (largest clean amplitude P12 allows; automatically <= tau' a_l), v <= tau' a on L', v = 0 elsewhere.
# Lemma:  h_model <= min_delta (K (a_{B^c} + b''))_delta,   b'' = tau' abar * (dilated, folded certified dominator b'_q),
#         ||a_{B^c} + b''||^r <= sum_{B^c} a^r + (tau' rho_q)^r abar^r q  <  sum a^r   (so h < Gamma ||a||).
# Phases: random, then pushed adversarially (coordinate search on theta to maximise h_model).
import numpy as np, json
from lib import *

s = 1.5; r, c, tp = consts(s)
cert = {}
for fn in ['compress_cert_uniform_s1.5.json', 'compress_cert_warm_s1.5.json', 'compress_cert_big_s1.5.json']:
    for q, e in json.load(open(fn)).items():
        if e['ok']: cert[int(q)] = (np.array(e['freqs']), np.array([float(x) for x in e['bprime']]))

def fold(p, freqs, w):
    F = (p-1)//2; out = np.zeros(F)
    for f, x in zip(freqs, w):
        m = f % p
        if m == 0: continue
        m = min(m, p-m); out[m-1] += x
    return out

def hmodel(p, A, v, th, K):
    F = (p-1)//2; l = np.arange(1, F+1); w = np.arange(p)
    phi = (v[:, None]*np.cos(2*np.pi*np.outer(l, w)/p + th[:, None])).sum(0)
    best = np.inf
    for d in range(1, p):
        dd = min(d, p-d)
        best = min(best, (A*K[dd-1]).sum() + (phi - np.roll(phi, d)).min())
    return best

rng = np.random.default_rng(3)
worst_ineq = -np.inf; worst_cost = -np.inf; worst_gamma = -np.inf
for trial in range(40):
    p = int(rng.choice([101, 211, 307])); K = kmat(p); F = (p-1)//2
    G = load_support(p)[0]
    q = int(rng.choice([17, 18, 20, 24, 30, 40])); d = int(rng.integers(1, p))
    blk = sorted({min((d*j) % p, p - (d*j) % p) for j in range(1, q+1)})
    if len(blk) < q: continue
    B = np.array(blk)-1
    rest = np.setdiff1d(np.arange(F), B)
    Cp = rng.choice(rest, int(rng.integers(0, 8)), replace=False)                    # |C'| <= 7: kappa-mass <= 14 < 29.7
    other = rng.choice(np.setdiff1d(rest, Cp), int(rng.integers(0, 6)), replace=False)
    a = np.zeros(F); abar = rng.uniform(0.5, 2)
    a[B] = abar; a[Cp] = rng.uniform(0.2, 2, len(Cp)); a[other] = rng.uniform(0.2, 3, len(other))
    v = np.zeros(F); Lp = np.concatenate([B, Cp]).astype(int)
    v[Lp] = tp*a[Lp]*rng.uniform(0.3, 1.0, len(Lp))
    A = a + c*v
    th = rng.uniform(0, 2*np.pi, F)
    h = hmodel(p, A, v, th, K)
    for sweep in range(2):                                                          # adversarial phase pushes
        for l in rng.choice(Lp, min(len(Lp), 10), replace=False):
            best = (h, th[l])
            for t in np.linspace(0, 2*np.pi, 9)[:-1]:
                th[l] = t; hv = hmodel(p, A, v, th, K)
                if hv > best[0]: best = (hv, t)
            th[l] = best[1]; h = best[0]
    fr, bw = cert[q]
    bpp = fold(p, d*fr, tp*abar*bw)
    aBc = a.copy(); aBc[B] = 0
    bound = (K @ (aBc + bpp)).min()
    worst_ineq = max(worst_ineq, h - bound)
    lhs_cost = np.sum((aBc + bpp)[(aBc + bpp) > 0]**r); rhs_cost = np.sum(a[a > 0]**r)
    worst_cost = max(worst_cost, lhs_cost - rhs_cost)
    worst_gamma = max(worst_gamma, h/(G*norm_r(a, s)))
print(f"max [h_model - min_d K(a_Bc + b'')]          = {worst_ineq:+.3e}   (must be <= 0)")
print(f"max [||a_Bc + b''||^r - ||a||^r]               = {worst_cost:+.3e}   (must be < 0)")
print(f"max h_model / (Gamma_ref ||a||) over the trials = {worst_gamma:.4f}     (closure: < 1)")
