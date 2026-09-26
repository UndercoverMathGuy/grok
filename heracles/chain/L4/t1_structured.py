# L4 lemmas A-C checked against brute force: arithmetic progressions, generalised progressions (GAPs), cubes of generators.
import numpy as np, itertools, math
from l4lib import *
rng = np.random.default_rng(1)
out = []
# --- Lemma A: AP {s, 2s, ..., Ls}: D = s^{-1} gives sum_j 2 sin^2(pi j / p) <= (pi^2/3)(L+1)(2L+1) L / p^2 (total), exact.
for p in (1009, 10007, 100003):
    for L in (4, 8, 16):
        s = int(rng.integers(2, p - 1)); H = [(j * s) % p for j in range(1, L + 1)]
        D = pow(s, -1, p); C = cov_matrix(p, H)
        actual_at_D = C[D].mean(); bound = (math.pi ** 2 / 3) * (L + 1) * (2 * L + 1) / p ** 2
        m, Dm = min_coverage(p, H)
        out.append(f"AP p={p} L={L}: cov at D=s^-1 {actual_at_D:.2e} <= bound {bound:.2e}: {actual_at_D <= bound * (1 + 1e-12)}; global min {m:.2e}")
# --- Lemma B: GAP {n1 g1 + ... + nd gd : |n_i| <= N_i}, box Dirichlet with prod Q_i < p:
#     exists D with ||kD||_p < sum_i |n_i| / Q_i  for every k;   cov <= 2 sin^2(pi * that) (if <= 1/2).
def gap(p, g, N):
    pts = set()
    for n in itertools.product(*[range(-Ni, Ni + 1) for Ni in N]):
        k = sum(ni * gi for ni, gi in zip(n, g)) % p
        if k not in (0,):
            pts.add(min(k, p - k))
    return sorted(pts)
for p in (10007, 100003, 1000003):
    for N in ((4, 4), (8, 3), (3, 3, 3), (2, 2, 2, 2)):
        d = len(N); g = [int(x) for x in rng.integers(1, p, d)]
        H = gap(p, g, N)
        lam = ((p - 1) / np.prod(N)) ** (1 / d)
        Q = [max(1, int(Ni * lam)) for Ni in N]
        while np.prod(Q) >= p:
            Q[int(np.argmax(Q))] -= 1
        D = box_dirichlet(p, g, Q)
        rho = sum(Ni / Qi for Ni, Qi in zip(N, Q))
        C = cov_matrix(p, H)[D]
        bound = 2 * math.sin(math.pi * min(rho, 0.5)) ** 2
        m, _ = min_coverage(p, H)
        out.append(f"GAP p={p} N={N} |H|={len(H)}: Q={Q} rho={rho:.3f} -> max cov at D {C.max():.3f} <= {bound:.3f}: {C.max() <= bound + 1e-12}; mean cov at D {C.mean():.3f}; global min mean {m:.3f}")
# --- Lemma C: cube of d generators (all signed sums, support <= j): N_i = 1, Q = floor((p-1)^{1/d})
for p in (10007, 1000003):
    for d, jmax in ((3, 3), (4, 2), (5, 2), (6, 2)):
        g = [int(x) for x in rng.integers(1, p, d)]
        H = set()
        for j in range(1, jmax + 1):
            for idx in itertools.combinations(range(d), j):
                for sg in itertools.product((-1, 1), repeat=j):
                    k = sum(s * g[i] for s, i in zip(sg, idx)) % p
                    if k: H.add(min(k, p - k))
        H = sorted(H); Q = int((p - 1) ** (1 / d))
        while Q ** d >= p: Q -= 1
        D = box_dirichlet(p, g, [Q] * d)
        C = cov_matrix(p, H)[D]; bound = 2 * math.sin(math.pi * min(jmax / Q, 0.5)) ** 2
        m, _ = min_coverage(p, H)
        out.append(f"cube p={p} d={d} len<={jmax} |H|={len(H)}: Q={Q} -> max cov at D {C.max():.3f} <= {bound:.3f}: {C.max() <= bound + 1e-12}; mean {C.mean():.3f}; global min mean {m:.3f}")
print("\n".join(out))
