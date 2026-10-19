"""Closed-form solver for the (P,T) moment relaxation (floating point version, for exploration).

F(P,T) = min { M_r(1+lam) : lam in R^4, sum lam = 0, sum lam^2 <= 2P, sum lam^3 >= T, lam >= -1 }.
By Claim V (variance active at minimisers, P<6) and the KKT analysis, minimisers are among
  C1: (-1^k, x^j, y^(4-k-j)) with sum = 0, sum of squares = 2P;
  C2: (x,x,y,z):  12x^3 - 6Px - T = 0,  y,z = -x +- sqrt(P-2x^2);
      (-1,x,y,z): x,y,z roots of t^3 - t^2 + (1-P)t - (T+3-3P)/3.
Feasibility: sum lam^3 >= T, lam >= -1.
"""
import numpy as np


def candidates(P, T):
    out = []
    if P >= 6 - 1e-15:
        out.append(np.array([3.0, -1, -1, -1]))
        return out
    for k in range(0, 4):
        for j in range(1, 4 - k + 1):
            n2 = 4 - k - j
            if n2 == 0:
                x = k / (4 - k)
                out.append(np.array([-1.0] * k + [x] * j))
                continue
            # y = (k - j x)/n2 ;  k + j x^2 + (k - j x)^2/n2 = 2P
            qa = j + j * j / n2; qb = -2 * k * j / n2; qc = k + k * k / n2 - 2 * P
            disc = qb * qb - 4 * qa * qc
            if disc < 0:
                continue
            for x in [(-qb + np.sqrt(disc)) / (2 * qa), (-qb - np.sqrt(disc)) / (2 * qa)]:
                y = (k - j * x) / n2
                out.append(np.array([-1.0] * k + [x] * j + [y] * n2))
    # C2 (x,x,y,z)
    for x in np.roots([12.0, 0.0, -6 * P, -T]):
        if abs(x.imag) > 1e-12:
            continue
        x = x.real
        if P - 2 * x * x < 0:
            continue
        sq = np.sqrt(P - 2 * x * x)
        out.append(np.array([x, x, -x + sq, -x - sq]))
    # C2 (-1,x,y,z)
    rts = np.roots([1.0, -1.0, 1 - P, -(T + 3 - 3 * P) / 3])
    if np.all(np.abs(rts.imag) < 1e-10):
        out.append(np.concatenate([[-1.0], rts.real]))
    return out


def F(P, T, r, tol=1e-9):
    best = np.inf
    for lam in candidates(P, T):
        if lam.min() < -1 - tol or abs(lam.sum()) > 1e-7:
            continue
        if np.sum(lam ** 3) < T - tol:
            continue
        if np.sum(lam ** 2) > 2 * P + 1e-7:
            continue
        nu = np.clip(1 + lam, 0, None)
        best = min(best, np.mean(nu ** r) ** (1 / r))
    return best


if __name__ == '__main__':
    # compare with brute force
    from inner_brute import brute
    r = 1 / 1.8552
    rng = np.random.default_rng(0)
    for t in range(12):
        P = rng.uniform(0.05, 2.8); T = -rng.uniform(0, 3) * P ** 1.5
        fb, lb = brute(P, T, 121)
        fa = F(P, T, r)
        print(f"P={P:.3f} T={T:.3f}: closed-form {4*(fa**r)/4:.5f}^... F={fa:.5f}  brute F={(fb/4)**(1/r):.5f}")
