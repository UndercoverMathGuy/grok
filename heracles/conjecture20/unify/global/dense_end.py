# Is K*(1.5,p) < F (i.e. some transition lies in [1, 1.5] = the s-range of n=2) for every prime 11 <= p <= 199?
# Certificate: any feasible allocation with value > the dense value (p/2) F^{-s} (the flat allocation is the only
# dense candidate up to the KKT structure; we compare against the best dense-support value computed by LP as well).
import numpy as np
from lib import programme, kappa_mat
from transitions_fast import best_support_value
def isprime(q): return q > 1 and all(q % d for d in range(2, int(q ** .5) + 1))
s = 1.5; bad = []
for p in [q for q in range(11, 200) if isprime(q)]:
    F = (p - 1) // 2
    dense = best_support_value(p, s, tuple(range(1, F + 1)), starts=3)       # best allocation with FULL support
    G, y, _ = programme(p, s, restarts=4)                                      # any (sparse) allocation: lower bound on Gamma
    K = int((y > 1e-12).sum())
    ok = G > dense * (1 + 1e-9) and K < F
    if not ok: bad.append(p)
    print(f"p={p}: full-support best {dense:.5f}  sparse allocation {G:.5f} with {K} features  -> transition below 1.5: {ok}", flush=True)
print("primes without certificate:", bad)
