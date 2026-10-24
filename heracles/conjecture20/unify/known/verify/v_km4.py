"""Referee check of KM4 / LARGEN R3 (slice subordination) and the degree-k BH consequence.

Class C single-frequency single feature: P(z) = sum_i beta_i prod_t (1 + xi_it z_t), Phi(x) = P(z(x)), z_t ranges over mu_p.
c := min_x cm(Phi(x)) (largest real c with Phi(x) in c + K), A = P(0), beta_T = sum_i beta_i prod_{t in T} xi_it,
H_k(zeta) = sum_{|T|=k} beta_T zeta^T.

R3:   cos(pi/p)^k sup_{T^n} |H_k| <= (4/p) |A - c|.
KM4, k=1 (BH_1 = 1):  sum_t |beta_{t}| <= (4/p)|A-c| / cos(pi/p).
KM4, k=2 (tetrahedral BH_2 <= 2^{7/4} via Littlewood 4/3 with constant sqrt2 and polarisation):
      (sum_{|T|=2} |beta_T|^{4/3})^{3/4} <= 2^{7/4} (4/p)|A-c| / cos(pi/p)^2.
We maximise each LHS/RHS adversarially over (beta, xi) (random restarts + hill climbing), for p in {3,5,7}, n in {2,3},
width m in {n, n+1, n+2}.  Also report how many of the found extremisers have positive margin (c > 0).
"""
import itertools
import numpy as np
rng = np.random.default_rng(99)


def setup(p, n):
    mu = np.exp(2j * np.pi * np.arange(p) / p)
    Z = np.array(list(itertools.product(mu, repeat=n)))           # p^n x n
    subsets = [T for k in range(1, n + 1) for T in itertools.combinations(range(n), k)]
    return Z, subsets


def stats(beta, xi, p, n, Z, subsets, zeta_grid):
    Phi = np.prod(1 + xi[None, :, :] * Z[:, None, :], axis=2) @ beta     # p^n
    cot = 1 / np.tan(np.pi / p)
    c = np.min(Phi.real - cot * np.abs(Phi.imag))
    A = beta.sum()
    rhs = (4 / p) * abs(A - c)
    R = np.cos(np.pi / p)
    bT = {T: np.sum(beta * np.prod(xi[:, list(T)], axis=1)) for T in subsets}
    out = {}
    for k in range(1, n + 1):
        Tk = [T for T in subsets if len(T) == k]
        H = sum(bT[T] * np.prod(zeta_grid[:, list(T)], axis=1) for T in Tk)
        out[f"R3_k{k}"] = R ** k * np.max(np.abs(H)) / rhs
    out["BH1"] = sum(abs(bT[(t,)]) for t in range(n)) * R / rhs
    if n >= 2:
        T2 = [T for T in subsets if len(T) == 2]
        out["BH2"] = (sum(abs(bT[T]) ** (4 / 3) for T in T2)) ** 0.75 * R ** 2 / (2 ** 1.75 * rhs)
    out["c"] = c
    return out


def zeta_grid_for(n, M=4000):
    G = np.exp(1j * rng.uniform(0, 2 * np.pi, (M, n)))
    if n == 2:
        a = np.linspace(0, 2 * np.pi, 120, endpoint=False)
        G = np.concatenate([G, np.exp(1j * np.array(list(itertools.product(a, a))))])
    return G


summary = {}
for p in [3, 5, 7]:
    for n in [2, 3]:
        Z, subsets = setup(p, n)
        zg = zeta_grid_for(n)
        for m in [n, n + 1, n + 2]:
            best = {}
            for start in range(25):
                beta = rng.normal(size=m) + 1j * rng.normal(size=m)
                th = rng.uniform(0, 2 * np.pi, (m, n))
                # half the starts near the IF block (equal beta, xi = omega^l) to probe the positive-margin regime
                if start % 2 == 0 and m >= n + 1:
                    beta = np.ones(m) * (1 + 0.05 * rng.normal(size=m))
                    th = np.array([[2 * np.pi * l / (n + 1)] * n for l in range(m)]) + 0.1 * rng.normal(size=(m, n))
                key_list = None
                st = stats(beta, np.exp(1j * th), p, n, Z, subsets, zg)
                for key in st:
                    if key == "c":
                        continue
                    # hill-climb this key
                    b_, t_ = beta.copy(), th.copy(); cur = st[key]; step = 0.3
                    for it in range(300):
                        nb = b_ + step * (rng.normal(size=m) + 1j * rng.normal(size=m)) * np.abs(b_).mean()
                        nt = t_ + step * rng.normal(size=(m, n))
                        s2 = stats(nb, np.exp(1j * nt), p, n, Z, subsets, zg)
                        if s2[key] > cur:
                            b_, t_, cur = nb, nt, s2[key]
                        else:
                            step *= 0.98
                    if cur > best.get(key, (-1, 0))[0]:
                        cfinal = stats(b_, np.exp(1j * t_), p, n, Z, subsets, zg)["c"]
                        best[key] = (cur, cfinal)
            line = ", ".join(f"{k}={v[0]:.4f}(c{'>' if v[1] > 0 else '<='}0)" for k, v in sorted(best.items()))
            print(f"p={p} n={n} m={m}: max LHS/RHS: {line}", flush=True)
            for k, v in best.items():
                summary[k] = max(summary.get(k, 0), v[0])
print("OVERALL max LHS/RHS (must be <= 1):", {k: round(v, 4) for k, v in summary.items()})
