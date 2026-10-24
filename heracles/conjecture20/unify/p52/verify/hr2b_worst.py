"""hr2b: re-examine the tiny positive residues of hr2 (R7 (c) and (d)) with high-precision roots (mpmath)."""
import numpy as np
import mpmath as mp
mp.mp.dps = 50
rng = np.random.default_rng(777)


def S_hp(P, q, r):
    P = mp.mpf(P); q = mp.mpf(q)
    if P == 0:
        return mp.mpf(3)
    m = 2 * mp.sqrt(P / 3)
    arg = 3 * q / (P * m)
    arg = max(min(arg, mp.mpf(1)), mp.mpf(-1))
    th = mp.acos(arg) / 3
    roots = [m * mp.cos(th - 2 * mp.pi * k / 3) for k in range(3)]
    return sum(max(1 + x, 0) ** mp.mpf(r) for x in roots)


def lowK(P):
    return max(P - 1, -2 * (P / 3) ** 1.5)


def upK(P):
    return 2 * (P / 3) ** 1.5


# (c): closed form vs brute force, high precision on cases where float said closed > brute
worst = -1
for trial in range(250):
    r = float(rng.choice([2 / 3, 0.8, 1.0]))
    Pbar = rng.uniform(0, 3.5); qlow = rng.uniform(-1.2, 2.2)
    Pb = min(Pbar, 3.0)
    if qlow > upK(Pb):
        continue
    sc = S_hp(Pb, max(qlow, lowK(Pb)), r)
    best = mp.inf
    for P in np.linspace(0, Pb, 60):
        lo = max(qlow, lowK(P)); hi = upK(P)
        if lo > hi:
            continue
        for q in np.linspace(lo, hi, 6):
            best = min(best, S_hp(P, q, r))
    worst = max(worst, float(sc - best))
print(f"[hr2b (c)] high-precision max[closed - brute grid] = {worst:.3e} (<= 0 means closed form is the min)")

# (d): single-atom / aligned-atom systems: payment vs P1 at high precision (tight case)
for (K, s) in [(1, 1.5), (1, 1.25), (2, 1.5)]:
    beta = np.array([1.0, 0.3])[:K]; xi = np.ones(K); eta = np.ones(K)
    A = beta.sum(); B = C = D = A
    P1 = 3 ** (1 - s) * np.sum(beta ** (1 / s)) ** s
    # L = Lmin = |A| (aligned): rho = 0; P = 3, q = 2 -> eigenvalues 3, 0, 0
    Sv = S_hp(3, 2, 1 / s)
    pay = A * (Sv / 3) ** s
    print(f"[hr2b (d)] aligned K={K} s={s}: payment at L=|A| = {mp.nstr(pay, 15)}  P1 = {P1:.15f}  diff = {mp.nstr(pay - P1, 5)}")
