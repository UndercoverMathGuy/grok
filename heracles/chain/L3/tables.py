# (1) Corollary table: H = B u C, B strongly dissociate, |C| = r => at most R = 3^r - 1 error patterns each for mass
#     and for every moment; if all have support >= j+1 (mass) / >= j (moments):  A* >= max_rho (rho - R rho^j)/(1 + R rho^{j+1}).
# (2) Random-set expectation bound: for a uniform random K-subset of {1..h},
#     E[E_0(rho)], E[E_k(rho)] <= Q(rho) = ((1+2rho)^K - 1 - 2K rho)/(h - K + 1); two-sided: odd / even parts.
#     With prob >= 1 - delta (Markov on the sum of K+1 errors) all errors <= (K+1) Q / delta.
import numpy as np, math
rhos = np.linspace(0.005, 0.5, 996)
def best(R0, R, j):
    return max((r - R * r ** j) / (1 + R0 * r ** (j + 1)) for r in rhos)
print("(1) corollary: minimal j with A* >= 0.3476 (then fable transfer gives l2-ball depth >= 0.272)")
for r in (1, 2, 3, 4):
    R = 3 ** r - 1
    row = {j: round(best(R, R, j), 3) for j in range(3, 14)}
    jmin = next(j for j in range(3, 40) if best(R, R, j) >= 0.3476)
    print(f"  r={r} (R={R}): " + " ".join(f"j={j}:{v}" for j, v in row.items()) + f"   -> j >= {jmin}")
def rand_bound(p, K, delta=0.5, two=False):
    h = (p - 1) // 2; out = -1
    for r in rhos:
        if two:
            Qm = (((1 + 2 * r) ** K - (1 - 2 * r) ** K) / 2 - 2 * K * r) / (h - K + 1)   # odd supports >= 3
            Q0 = (((1 + 2 * r) ** K + (1 - 2 * r) ** K) / 2 - 1) / (h - K + 1)             # even supports >= 2
        else:
            Qm = Q0 = ((1 + 2 * r) ** K - 1 - 2 * K * r) / (h - K + 1)
        e = (K + 1) / delta
        out = max(out, (r - e * Qm) / (1 + e * Q0))
    return out
print("(2) random K-subsets, K = round(c ln p), guaranteed with prob >= 1/2 (Markov):")
for c in (1.0, 1.3, 1.44):
    for lp in (7, 10, 14, 20, 30, 50, 80):
        p = math.exp(lp); K = round(c * lp)
        print(f"  c={c} ln p={lp:3d} K={K:3d}: one-sided >= {rand_bound(p, K):.3f}   two-sided >= {rand_bound(p, K, two=True):.3f}")
