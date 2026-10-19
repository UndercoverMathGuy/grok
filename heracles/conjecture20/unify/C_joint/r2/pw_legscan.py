import numpy as np
from pw_scan import legendre_phases, ratio
def isprime(q): return q > 1 and all(q % d for d in range(2, int(q**0.5)+1))
for (n, s) in [(2, 1.5), (3, 2.0), (2, 1.4)]:
    first = None; rows = []
    for p in range(211, 1100):
        if not isprime(p) or p % 4 != 3: continue
        v = ratio(legendre_phases(p), p, n, s)
        if v > 1 and first is None: first = p
        if v > 1: rows.append((p, round(v, 4)))
    print(f"n={n} s={s}: first p=3 mod 4 with Legendre PW_C violation: {first}; violators: {rows[:12]}", flush=True)
