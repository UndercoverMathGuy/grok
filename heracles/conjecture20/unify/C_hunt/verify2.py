# Exact re-evaluation of saved networks (any .npz from the hunts):
#   - the network is rebuilt as an explicit product network (U, W) in float64; its weights are exact binary rationals;
#   - margin h = min over ALL x in Z_p^n and ALL c != y(x) of f(x)[y] - f(x)[c], and cost = sum ||omega_i||^b, are enclosed with
#     mpmath interval arithmetic (iv, 30 digits)  =>  rigorous upper bound gamma_up = h_up / cost_low^s;
#   - the feature class T = {l : |beta_l| > tol} and Gamma^{(T)} by exact B&B with an mpmath re-solve of the optimal vertex;
#   - ratio_up = gamma_up / (pi B_p Gamma^{(T)}) with pi = 3^{1-s} (n = 2) or nu^{1-s} (n >= 3: IF reference).
# Usage: python verify2.py file1.npz [file2.npz ...]
import sys, json, numpy as np
import mpmath as mp
import hlib as H, hlib2 as H2
GL = H.GL


def load(fn):
    d = np.load(fn, allow_pickle=True)
    p = int(d['p']); n = int(d['n']); b = float(d['b'])
    if 'U' in d.files:
        U, W = d['U'], d['W']
    else:
        s = (n + 1) / b
        cc = H.CC(p, n, list(d['freqs']), s)
        U, W = cc.to_net(d['x'], b)
    return p, n, b, np.asarray(U, float), np.asarray(W, float), d


for fn in sys.argv[1:]:
    p, n, b, U, W, d = load(fn)
    nu = n + 1; s = nu / b; F = (p - 1) // 2
    nt = GL.Net(U, W, b)
    gam_float = nt.margin() / nt.cost() ** s
    beta = np.abs(nt.clean()); T = [l for l in range(1, F + 1) if beta[l - 1] > 1e-9 * beta.max()]
    hl, hu, costs = H2.exact_margin_table(U, W, dps=30, interval=True)
    mp.iv.dps = 30
    cost_iv = sum(c ** (mp.iv.mpf(b) / 2) for c in costs)                        # sum ||omega||^b, enclosed
    gam_up = float((hu / cost_iv ** mp.iv.mpf(s)).b)                               # rigorous upper bound on gamma
    R = H2.prog_bb(p, s, S=T, maxnodes=200000) if len(T) <= 12 else H2.prog_best(p, s, S=T)
    pi = 3 ** (1 - s) if n == 2 else nu ** (1 - s)
    Dt = pi * H.Bp(p, n) * R['G']
    out = dict(file=fn.split('/')[-1], p=p, n=n, b=b, m=U.shape[0], T=T, gamma_float=gam_float, ratio_float=gam_float / Dt,
               h_interval=[float(hl), float(hu)], ratio_up=gam_up / Dt, GammaT=R['G'], GammaT_gap=R['gap'],
               stored_ratio=float(d['ratio']) if 'ratio' in d.files else None)
    print(json.dumps(out), flush=True)
