# Depth-only scan (no floored programme) for large p: structural parameters + one-/two-sided depths + transfer.
# usage: depth_scan.py p  (writes depth_p{p}.jsonl)
import sys, json, time, numpy as np
from tools import *
p = int(sys.argv[1]); rng = np.random.default_rng(p + 5); h = (p - 1) // 2
try:
    Gam, Sstar = gamma_star(p, rng) if p < 500 else (None, None)
except Exception:
    Gam, Sstar = None, None
fams = []
if Sstar: fams.append(('Gamma*-support', sorted(Sstar)))
for L in (3, 5, 8, 12):
    fams.append((f'AP{{1..{L}}}', list(range(1, L + 1))))
K0 = len(Sstar) if Sstar else int(round(1.3 * np.log(p)))
for t in range(3):
    fams.append((f'random K={K0} #{t}', sorted(int(k) for k in rng.choice(np.arange(1, h + 1), K0, replace=False))))
for t in range(2):
    base = [int(b) for b in rng.choice(np.arange(1, h + 1), K0 - 2, replace=False)]
    fams.append((f'planted 2 sums #{t}', sorted(set(base) | {fold(base[0] + base[1], p), fold(base[0] + base[2], p)})))
g = next(x for x in range(2, p) if all(pow(x, (p - 1) // q, p) != 1 for q in {q for q in range(2, p) if (p - 1) % q == 0 and all(q % r for r in range(2, q))}))
for m in sorted(m for m in range(4, 40) if (p - 1) % m == 0):
    H = sorted({fold(pow(g, (p - 1) // m * i, p), p) for i in range(m)})
    if 4 <= len(H) <= 14:
        fams.append((f'mult subgroup order {m}', H))
out = open(f'depth_p{p}.jsonl', 'a')
for name, T in fams:
    t0 = time.time()
    rec = dict(p=p, name=name, T=T, K=len(T), diss_dim=dissociated_dimension(p, T), shortest_rel=shortest_relation(p, T, jmax=4),
               energy=energy(p, T))
    rec['A_star'] = depth_polar(p, T, False, rng, starts=8, iters=150)[0]
    rec['A_s'] = depth_polar(p, T, True, rng, starts=8, iters=150)[0]
    rec['c_lo'] = transfer(rec['A_s'], rec['A_star'])
    rec['l2_floor_1_over_sqrt_2K'] = 1 / np.sqrt(2 * len(T))
    if Gam: rec['Gamma_T'] = gamma_H(p, T, rng, starts=4)[0]; rec['Gamma_star'] = Gam
    rec['secs'] = time.time() - t0
    out.write(json.dumps(rec) + '\n'); out.flush()
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
