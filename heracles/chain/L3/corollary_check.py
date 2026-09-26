# r = 1 corollary on explicit sets: B strongly dissociate, c chosen with 2c = signed sum of j elements of 2B.
# Predicted: mass patterns 2 of support j+1; moment patterns: support j (k in S_c or k = c), j+2 (k in B \ S_c).
# One-sided >= max_rho (rho - max(rho^j, 2 rho^{j+2})) / (1 + 2 rho^{j+1}); two-sided = 1/2 when j is even.
import itertools, numpy as np
from fractions import Fraction as Fr
from riesz_bound import bounds, histograms
p = 100003; h = (p - 1) // 2; inv2 = pow(2, -1, p); rng = np.random.default_rng(3)
def strongly_dissociate(B):
    s = [sum(e * 2 * b for e, b in zip(eps, B)) % p for eps in itertools.product((-1, 0, 1), repeat=len(B))]
    return len(set(s)) == len(s)
while True:
    B = sorted(rng.choice(np.arange(1, h + 1), 7, replace=False).tolist())
    if strongly_dissociate(B): break
rhos = [Fr(i, 200) for i in range(1, 101)]
for j in (2, 3, 4, 5, 6):
    S = B[:j]; eps = [1 if i % 2 == 0 else -1 for i in range(j)]
    c2 = sum(e * 2 * b for e, b in zip(eps, S)) % p; c = (c2 * inv2) % p; c = c if c <= h else p - c
    H = B + [c]
    (b1, r1), (b2, r2) = bounds(p, H, rhos)[:2]
    pred = max((r - max(r ** j, 2 * r ** (j + 2))) / (1 + 2 * r ** (j + 1)) for r in rhos)
    N0, Nk = histograms(p, H)
    print(f"j={j}: mass patterns by support {[(s, int(N0[s])) for s in range(len(N0)) if N0[s]]}; one-sided {float(b1):.4f} "
          f"(predicted {float(pred):.4f}); two-sided {float(b2):.4f}")
