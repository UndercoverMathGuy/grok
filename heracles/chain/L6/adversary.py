# Adversarial search for gap sets: fixed size K, local search (swap one element) to MINIMISE the two-sided depth A_s
# (fast primal proxy), optionally subject to Gamma_T >= Gamma*/sqrt3 (competitive even without floors).
# Then evaluate the best sets properly: polar depths, transfer, and the share-floored ratio at the discount delta(A_s).
# usage: adversary.py p K steps competitive(0/1)   (appends to adv_p{p}.jsonl)
import sys, json, time, numpy as np
sys.path.insert(0, '..'); sys.path.insert(0, '../L5')
from tools import *
from floor_test import value as floored_ratio

p, K, steps, comp = (int(a) for a in sys.argv[1:5]); rng = np.random.default_rng(1000 * p + K)
h = (p - 1) // 2
Gam, Sstar = gamma_star(p, rng)
A = cov_matrix(p)
delta = lambda a: max(1.0, np.sqrt(3) * (1 - np.sqrt(3) * a))

def proxy(T):
    d = depth_est(p, T, True, rng, starts=3)
    if comp:
        g = gamma_H(p, T, rng, starts=2, A=A)[0]
        if g < Gam / np.sqrt(3):
            d += 1.0 + (Gam / np.sqrt(3) - g)          # infeasible: push away
    return d

T = sorted(int(k) for k in rng.choice(np.arange(1, h + 1), K, replace=False))
cur = proxy(T); best = (cur, T)
for s in range(steps):
    i = rng.integers(K); new = [k for k in T if k != T[i]]
    cand = int(rng.choice([k for k in range(1, h + 1) if k not in T]))
    Tn = sorted(new + [cand]); v = proxy(Tn)
    if v < cur or rng.uniform() < np.exp(-(v - cur) / 0.01):
        T, cur = Tn, v
        if v < best[0]:
            best = (v, T)
T = best[1]
rec = dict(p=p, K=K, competitive_constraint=comp, T=T, proxy=best[0])
rec['A_star'] = depth_polar(p, T, False, rng, starts=8, iters=150)[0]
rec['A_s'] = depth_polar(p, T, True, rng, starts=8, iters=150)[0]
rec['c_lo'] = transfer(rec['A_s'], rec['A_star'])
rec['Gamma_T'] = gamma_H(p, T, rng, starts=6, A=A)[0]
rec['diss_dim'] = dissociated_dimension(p, T); rec['energy'] = energy(p, T)
for eps in (0.01, 0.03):
    if eps * K <= 0.9:
        rec[f'ratio_opt_eps{eps}'] = floored_ratio(p, T, delta(rec['A_s']), eps, Gam, starts=20)
open(f'adv_p{p}.jsonl', 'a').write(json.dumps(rec) + '\n')
print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
