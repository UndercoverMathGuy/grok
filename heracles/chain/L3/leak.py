# Aim moments OUTSIDE H (interface for L2): for the full-H Riesz aim w = R/(pM) (L3-A),
#   |E_w[w^{u x}]| <= F(u) / (1 - E_0)            for every residue u not in {0, +-2k : k in H}   (one-sided)
#   two-sided mixture (Riesz(lambda) with flipped Riesz(-lambda)):  |.| <= F_odd(u) / (1 - E_0^even)
# F(u) = sum_{eps: s(eps) = u} rho^{|eps|} (triangle inequality over the patterns landing on u; exact integer counts).
# tau_1 = max_u F(u)/(1 - E_0), tau_2 = max_u F_odd(u)/(1 - E_0^even). Also prints the matching depth bounds.
import math, itertools, sys, numpy as np
from fractions import Fraction as Fr
def enum(p, H):
    sums = np.zeros(1, np.int64); sup = np.zeros(1, np.int64)
    for k in H:
        mk = (2 * k) % p
        sums = np.concatenate([sums, (sums + mk) % p, (sums - mk) % p]); sup = np.concatenate([sup, sup + 1, sup + 1])
    return sums, sup
def report(p, H, rhos=(0.3, 0.35, 0.4, 0.45, 0.5)):
    sums, sup = enum(p, H); m = np.array([(2 * k) % p for k in H]); inH = np.zeros(p, bool); inH[0] = True
    inH[m] = True; inH[(-m) % p] = True
    out = []
    for rho in rhos:
        wt = rho ** sup.astype(float); odd = (sup % 2 == 1)
        F = np.bincount(sums, weights=wt, minlength=p); Fo = np.bincount(sums, weights=wt * odd, minlength=p)
        rel = (sums == 0) & (sup > 0)
        E0 = wt[rel].sum(); E0e = wt[rel & ~odd].sum()
        Ek = [wt[(sums == mk)].sum() - rho for mk in m]; Eko = [wt[(sums == mk) & odd].sum() - rho for mk in m]
        one = (rho - max(Ek)) / (1 + E0); two = (rho - max(Eko)) / (1 + E0e)
        tau1 = F[~inH].max() / (1 - E0) if E0 < 1 else math.inf
        tau2 = Fo[~inH].max() / (1 - E0e) if E0e < 1 else math.inf
        out.append((rho, one, two, tau1, tau2))
    return out
if __name__ == '__main__':
    rng = np.random.default_rng(7)
    cases = []
    p = 100003; h = (p - 1) // 2
    def sd(B): s = [sum(e * 2 * b for e, b in zip(eps, B)) % p for eps in itertools.product((-1, 0, 1), repeat=len(B))]; return len(set(s)) == len(s)
    while True:
        B = sorted(rng.choice(np.arange(1, h + 1), 7, replace=False).tolist())
        if sd(B): break
    cases.append(("strongly dissociate |B|=7", p, B))
    inv2 = pow(2, -1, p)
    for j in (2, 3, 4):
        c = (sum((1 if i % 2 == 0 else -1) * 2 * b for i, b in enumerate(B[:j])) * inv2) % p; c = min(c, p - c)
        cases.append((f"B + one element, j={j}", p, B + [c]))
    for pp, cc in ((10007, 1.0), (10007, 1.3), (100003, 1.3)):
        hh = (pp - 1) // 2; K = round(cc * math.log(pp))
        cases.append((f"random K={K} (c={cc})", pp, sorted(rng.choice(np.arange(1, hh + 1), K, replace=False).tolist())))
    for name, pp, H in cases:
        print(f"{name} (p={pp}):")
        for rho, one, two, t1, t2 in report(pp, H):
            print(f"   rho={rho:.2f}: depth one-sided {one:.3f} two-sided {two:.3f} | leak tau1 {t1:.3f} (R=(1+tau)^2 {(1 + t1) ** 2:.3f})  tau2 {t2:.3f} (R {(1 + t2) ** 2:.3f})")
