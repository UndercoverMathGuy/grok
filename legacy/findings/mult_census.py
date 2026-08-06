"""Proxy 2: multiplicative-relation census of the 31 observed committees.
Pair (i,j) is ratio-r if k_j = +-r*k_i or k_i = +-r*k_j (mod p, folded).
Primary: ratio 2 (ReLU squaring subsidy). Secondary: ratio 3.
Recheck additive avoidance (k_i+k_j or |k_i-k_j| in committee) at n=31."""
import sys
from pathlib import Path
import numpy as np

ROOT = Path("/Users/ruhaanrajadhyaksha/projects/grok")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).parent))
from mask_lottery import discover, final_coeffs_and_acc, committee_from_coeffs

def fold(x, p):
    x = x % p
    return min(x, p - x)

def n_ratio(comm, p, r):
    c = 0
    for i in range(len(comm)):
        for j in range(len(comm)):
            if i != j and fold(r * comm[i], p) == comm[j]:
                c += 1
    return c // 1  # ordered count; symmetric pairs counted once each direction? no:
                   # k_j=2k_i and k_i=2k_j can't both hold (p prime, r^2!=1)

def n_additive(comm, p):
    s = set(comm)
    c = 0
    for i in range(len(comm)):
        for j in range(i + 1, len(comm)):
            if fold(comm[i] + comm[j], p) in s or fold(comm[i] - comm[j], p) in s:
                c += 1
    return c

comms = []
for d, cfg in discover():
    coeffs, acc, _ = final_coeffs_and_acc(d, cfg)
    if acc < 0.99:
        continue
    comms.append((cfg.p, committee_from_coeffs(coeffs)))

rng = np.random.default_rng(0)
def census(fn, name, *args):
    obs = sum(fn(c, p, *args) for p, c in comms)
    null = np.empty(20000)
    for t in range(20000):
        tot = 0
        for p, c in comms:
            rnd = rng.choice(np.arange(1, p // 2 + 1), len(c), replace=False)
            tot += fn(rnd.tolist(), p, *args)
        null[t] = tot
    plo = (null <= obs).mean(); phi = (null >= obs).mean()
    print(f"{name}: observed {obs}, null {null.mean():.2f} +/- {null.std():.2f}, "
          f"p(depleted) = {plo:.4f}, p(enriched) = {phi:.4f}")

print(f"{len(comms)} committees")
census(n_ratio, "ratio-2 pairs", 2)
census(n_ratio, "ratio-3 pairs", 3)
census(n_additive, "additive pairs (sum or diff in committee)")
