# Multi-offset vs pair-offset depth on the hard cases.
# usage: run1.py p T(comma list | 'dense' | 'ap:L' | 'gstar' | 'rand:K:seed') modes(e.g. 1,lone,3) [eps]
import sys, json, time, numpy as np
sys.path.insert(0, '..')
from common import programme
from mo import *

p = int(sys.argv[1]); spec = sys.argv[2]; modes = sys.argv[3].split(',')
h = (p - 1) // 2
rng = np.random.default_rng(7)
GS = {59: 0.31304887778665563, 113: 0.2759357645841438, 211: 0.23634003816804908}
Gam = GS.get(p) or (programme(p, 1.5)[0] if p <= 29 else None)
if spec == 'dense':
    T = list(range(1, h + 1))
elif spec.startswith('ap:'):
    T = list(range(1, int(spec[3:]) + 1))
elif spec == 'gstar':
    T = {59: [1, 7, 11, 23, 27], 113: [1, 6, 16, 31, 35, 44, 55],
         211: [5, 30, 36, 50, 54, 74, 79, 88, 92, 97, 104]}[p]
elif spec.startswith('rand:'):
    _, K, sd = spec.split(':')
    T = sorted(np.random.default_rng(int(sd)).choice(np.arange(1, h + 1), int(K), replace=False).tolist())
else:
    T = [int(x) for x in spec.split(',')]
K = len(T)
eps = float(sys.argv[4]) if len(sys.argv) > 4 else (Gam ** 2 / 12 if Gam else 0.01)
eps = min(eps, 0.5 / K)
pool = None
if spec.startswith('ap:') or spec.startswith('rand:') or (spec not in ('dense', 'gstar')):
    gs = {59: [1, 7, 11, 23, 27], 113: [1, 6, 16, 31, 35, 44, 55], 211: [5, 30, 36, 50, 54, 74, 79, 88, 92, 97, 104]}
    pool = sorted(set(gs.get(p, [])) | set(np.random.default_rng(3).choice(np.arange(1, h + 1), min(h, 30), replace=False).tolist()))
v, bS, bind, C, S = floored_spectrum(p, T, eps, rng, pool=pool)
b = np.array([bS[S.index(k)] for k in T])
lamK = dual_weights(p, S, bS, bind, eps)
out = dict(p=p, spec=spec, T=T, K=K, eps=eps, Gamma=Gam, floored_value=v, value_over_Gamma=(v / Gam if Gam else None),
           shares_T=np.round(b, 4).tolist(), helper_shares={int(k): round(float(x), 4) for k, x in zip(S, bS) if k not in T and x > 1e-3}, binding=bind.tolist(), dual=np.round(lamK, 4).tolist())
print(json.dumps(out), flush=True)
steps = int(sys.argv[5]) if len(sys.argv) > 5 else 300
for mode in modes:
    res = dict(mode=mode)
    t0 = time.time()
    # pair-offset depth at each binding offset (independent rows at D and -D)
    pairs = []
    for D in bind:
        Ds, lam = sym([D], [1.0], p)
        a, _ = depth_search(Body(p, T, Ds, lam, mode='1' if mode == '1' else '3'), mode, rng, steps=steps)
        pairs.append(round(a, 4))
    res['pair_at_binding'] = pairs
    # multi-offset: uniform over binding set, and the KKT dual
    for name, lam0 in (('uniform_binding', np.ones(len(bind)) / len(bind)), ('dual', lamK)):
        Ds, lam = sym(bind, lam0, p)
        a, vbest = depth_search(Body(p, T, Ds, lam, mode='1' if mode == '1' else '3'), mode, rng, steps=steps)
        res[name] = round(a, 4)
    # reference: uniform over ALL offsets (admissible only for spectra whose coverage is flat)
    Ds = np.arange(1, p); lam = np.ones(p - 1) / (p - 1)
    if mode == '1' or p <= 59:
        a, _ = depth_search(Body(p, T, Ds, lam, mode='1' if mode == '1' else '3'), mode, rng, steps=steps)
        res['uniform_all'] = round(a, 4)
    res['secs'] = round(time.time() - t0)
    print(json.dumps(res), flush=True)
