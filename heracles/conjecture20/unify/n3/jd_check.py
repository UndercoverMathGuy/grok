"""Numerical check of the n=3 joint-detection lemma (four plane slicings + line slicings).

For random mean-zero profiles D_T (T the 7 non-empty subsets of [3]) on Z_p, and every d != 0:
  A = -min_u Z(u),  A' = max_u Z^d(u),  Z(u)=sum_T D_T(s_T(u)),  Z^d(u)=sum_T D_T(s_T(u)+d).
Checks (all must hold):
  S_k  : A A' >= kap (E - V_k) + C_k            (k = 1,2,3)       [gibbs R9]
  S_123: A A' >= kap (E - V_123) + C_123                           [new slicing]
  L_k  : A A' >= sum_{T contains k} V_T + sum_{T not containing k} C_T            [line slicing]
  L_ij : A A' >= V_i+V_j+V_ik+V_jk + C_k + C_ij + C_ijk                           [diagonal line slicing]
  J    : A A' >= ((3 kap - zeta)/4) E  for class-C profiles (zeta = cos(pi/p))
Profiles: (i) arbitrary mean-zero, (ii) class-C type D_T(w) = sum_l Re(a_{lT} c_l(delta) rho^{l w}),
(iii) adversarial sawtooth / 4-circuit concentrated families.
"""
import itertools
import numpy as np

TS = [T for k in range(1, 4) for T in itertools.combinations(range(3), k)]
IX = {T: a for a, T in enumerate(TS)}


def setup(p):
    U = np.array(list(itertools.product(range(p), repeat=3)))
    S = np.stack([U[:, list(T)].sum(axis=1) % p for T in TS], axis=1)
    return S


def check(p, D, d, S):
    Z = sum(D[a][S[:, a]] for a in range(7))
    Zd = sum(D[a][(S[:, a] + d) % p] for a in range(7))
    A, Ap = -Z.min(), Zd.max()
    V = np.array([np.mean(D[a] ** 2) for a in range(7)])
    C = np.array([np.mean(D[a] * np.roll(D[a], -d)) for a in range(7)])
    E = V.sum(); kap = (p - 1) / (p + 1)
    prod = A * Ap
    out = {}
    for k in range(3):
        out['S%d' % (k + 1)] = kap * (E - V[IX[(k,)]]) + C[IX[(k,)]]
    out['S123'] = kap * (E - V[IX[(0, 1, 2)]]) + C[IX[(0, 1, 2)]]
    for k in range(3):
        out['L%d' % (k + 1)] = sum(V[a] if k in T else C[a] for a, T in enumerate(TS))
    for i, j in itertools.combinations(range(3), 2):
        k = 3 - i - j
        core = [(i,), (j,), tuple(sorted((i, k))), tuple(sorted((j, k)))]
        out['L%d%d' % (i + 1, j + 1)] = sum(V[IX[T]] for T in core) + C[IX[(k,)]] + C[IX[(i, j)]] + C[IX[(0, 1, 2)]]
    return prod, E, out


if __name__ == '__main__':
    rng = np.random.default_rng(5)
    worst = {}
    for p in [5, 7, 11]:
        S = setup(p)
        rho = np.exp(2j * np.pi / p)
        zeta = np.cos(np.pi / p); kap = (p - 1) / (p + 1)
        minJ = np.inf
        for trial in range(400):
            kind = trial % 3
            if kind == 0:   # arbitrary mean-zero
                D = rng.normal(size=(7, p)) * rng.exponential(1, size=(7, 1))
                D -= D.mean(axis=1, keepdims=True)
            elif kind == 1:  # class-C type profiles for a random delta
                delta = rng.integers(1, p)
                w = np.arange(p)
                D = np.zeros((7, p))
                for a in range(7):
                    for l in range(1, (p - 1) // 2 + 1):
                        coef = (rng.normal() + 1j * rng.normal()) * rng.exponential(1) * (rng.random() < 0.6)
                        D[a] += np.real(coef * (1 - rho ** (-l * delta)) * rho ** (l * w))
            else:  # energy concentrated on the 4-circuit {1,2,3,123} or a 3-circuit, sawtooth-like
                saw = np.array([((-x) % p) - (p - 1) / 2 for x in range(p)], dtype=float)
                D = np.zeros((7, p))
                sets = [IX[(0,)], IX[(1,)], IX[(2,)], IX[(0, 1, 2)]] if rng.random() < 0.5 else list(rng.choice(7, 3, replace=False))
                for a in sets:
                    D[a] = np.roll(saw, rng.integers(p)) * rng.choice([-1, 1]) * rng.uniform(0.5, 1.5)
                D += 0.05 * rng.normal(size=(7, p)); D -= D.mean(axis=1, keepdims=True)
            for d in range(1, p):
                prod, E, out = check(p, D, d, S)
                for key, val in out.items():
                    slack = (prod - val) / E
                    worst[key] = min(worst.get(key, np.inf), slack)
                if kind == 1:
                    minJ = min(minJ, prod / E)
        print(f"p={p}: min AA'/E over class-C-type profiles = {minJ:.4f}  vs guaranteed (3kap-zeta)/4 = {(3*kap-zeta)/4:.4f}")
    print("min normalised slack (must be >= 0):", {k: round(v, 5) for k, v in worst.items()})
