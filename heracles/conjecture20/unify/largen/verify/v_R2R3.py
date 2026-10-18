# Independent referee checks of largen R2 (higher-order circularity, cone moments) and R3 (slice subordination).
# Fields are computed from the LOGIT TABLE via the definition Phi_1(x) = (2/p) rho^{y(x)} sum_c f(x)[c] rho^{-c},
# not from any atom formula.  Coefficients b_T of Phi in z_t = rho^{2 x_t} are obtained by Fourier analysis of Phi.
# Part 1: random class-C and single-frequency-with-DC networks (no positive-margin requirement: c = min cm always gives
#         Phi in c+K, so R2(c) and R3 apply to EVERY network with that c).
# Part 2: adversarial maximisation of the R3 ratio and the R2(c) ratio over atom systems (scipy, many restarts).
import itertools, math, sys
import numpy as np
from scipy.optimize import minimize
rng = np.random.default_rng(12345)

def inputs(p, n):
    return np.array(list(itertools.product(range(p), repeat=n)), dtype=int)

def logits(U, W, X):
    # U: (m, n, p) input vectors, W: (m, p) outputs; f(x)[c] = sum_i W[i,c] prod_t U[i,t,x_t]
    m, n, p = U.shape
    prod = np.ones((len(X), m))
    for t in range(n):
        prod *= U[:, t, :][:, X[:, t]].T
    return prod @ W                                     # (p^n, p)

def field(U, W, p, X):
    F = logits(U, W, X)
    rho = np.exp(2j * np.pi / p)
    y = X.sum(1) % p
    return (2 / p) * rho ** y * (F @ rho ** (-np.arange(p)))

def cm(z, p):
    return z.real - np.abs(z.imag) / math.tan(math.pi / p)

def classC_net(p, n, m, dc=False):
    a = np.arange(p)
    U = np.empty((m, n, p)); W = np.empty((m, p))
    for i in range(m):
        for t in range(n):
            al, be = rng.standard_normal(2)
            U[i, t] = al * np.cos(2 * np.pi * a / p) + be * np.sin(2 * np.pi * a / p)
            if dc: U[i, t] += rng.standard_normal()
        al, be = rng.standard_normal(2)
        W[i] = al * np.cos(2 * np.pi * a / p) + be * np.sin(2 * np.pi * a / p) + rng.standard_normal()  # output DC is harmless
    return U, W

def coeffs_z(Phi, p, n, X):
    # b_e for e in {0..p-1}^n with Phi = sum_e b_e prod_t z_t^{e_t}, z_t = rho^{2x_t}
    rho = np.exp(2j * np.pi / p)
    out = {}
    for e in itertools.product(range(p), repeat=n):
        e = np.array(e)
        out[tuple(e)] = np.mean(Phi * rho ** (-2 * (X @ e)))
    return out

print("=== Part 1a: R2(a)/(b) moments, networks built from weights ===")
for (p, n, m) in [(3, 3, 3), (5, 2, 3), (5, 3, 2), (7, 2, 3), (7, 3, 3), (11, 2, 2)]:
    X = inputs(p, n)
    for dc in [False, True]:
        U, W = classC_net(p, n, m, dc)
        Phi = field(U, W, p, X); A = Phi.mean()
        kmax = p if not dc else (p + 1) // 2
        rel = [abs(np.mean((Phi - A) ** k)) / np.mean(np.abs(Phi - A) ** k) for k in range(1, kmax + 1)]
        pred = p - 1 if not dc else (p - 1) // 2
        print(f" p={p} n={n} {'DC ' if dc else 'C  '} predicted zero up to k={pred}: ", " ".join(f"{v:.0e}" for v in rel))

print("=== Part 1b: support of Phi in z-exponents (class C must be multi-affine: e in {0,1}^n) ===")
for (p, n, m) in [(5, 2, 3), (7, 2, 2), (3, 3, 2)]:
    X = inputs(p, n); U, W = classC_net(p, n, m, False)
    Phi = field(U, W, p, X); b = coeffs_z(Phi, p, n, X)
    off = max(abs(v) for e, v in b.items() if max(e) > 1)
    on = max(abs(v) for e, v in b.items() if max(e) <= 1)
    print(f" p={p} n={n}: max |b_e| with some e_t>=2: {off:.1e}   (on {{0,1}}^n: {on:.2e})")

def slice_coeffs(b, S, zeta, n):
    # c_k = sum_{T subset S, |T|=k} b_T zeta^T  (b keyed by 0/1 tuples)
    ck = np.zeros(len(S) + 1, complex)
    for k in range(len(S) + 1):
        for T in itertools.combinations(range(len(S)), k):
            e = [0] * n
            for j in T: e[S[j]] = 1
            ck[k] += b[tuple(e)] * np.prod([zeta[j] for j in T])
    return ck

print("=== Part 1c: R3 on random class-C networks (all networks, c = min cm) and R2(c) ===")
worst3 = 0; worst3s = 0; worst2 = 0; cnt = 0
for (p, n, m) in [(3, 2, 2), (3, 3, 3), (5, 2, 2), (5, 2, 4), (5, 3, 3), (7, 2, 3), (7, 3, 2), (11, 2, 3)]:
    X = inputs(p, n); R = math.cos(math.pi / p)
    for trial in range(40):
        U, W = classC_net(p, n, m, False)
        Phi = field(U, W, p, X); A = Phi.mean(); c = cm(Phi, p).min()
        b = coeffs_z(Phi, p, n, X) if p <= 7 else None
        if b is None:
            continue
        b01 = {e: v for e, v in b.items() if max(e) <= 1}
        cnt += 1
        ang = np.angle(A - c)
        sharp = (4 / p) * abs(A - c) * math.cos(p / 2 * ang)
        for S in [list(s) for r in range(1, n + 1) for s in itertools.combinations(range(n), r)]:
            for _ in range(15):
                zeta = np.exp(2j * np.pi * rng.random(len(S)))
                ck = slice_coeffs(b01, S, zeta, n)
                for k in range(1, len(S) + 1):
                    worst3 = max(worst3, R ** k * abs(ck[k]) / ((4 / p) * abs(A - c)))
                    worst3s = max(worst3s, R ** k * abs(ck[k]) / sharp)
        for k in range(1, (p - 1) // 2 + 1):
            worst2 = max(worst2, np.mean(np.abs(Phi - c) ** k) / (((A - c) ** k).real / math.cos(k * math.pi / p)))
print(f" {cnt} random class-C nets: max R3 ratio = {worst3:.4f}, vs sharp Rogosinski bound (4/p)|A-c|cos(p/2 arg) = {worst3s:.4f}; max R2(c) ratio = {worst2:.4f}  (all must be <= 1)")

print("=== Part 1d: R2(c) on random single-frequency networks WITH DC ===")
worst = 0
for (p, n, m) in [(5, 2, 3), (5, 3, 2), (7, 2, 3), (7, 3, 3), (11, 2, 2)]:
    X = inputs(p, n)
    for trial in range(60):
        U, W = classC_net(p, n, m, True)
        Phi = field(U, W, p, X); A = Phi.mean(); c = cm(Phi, p).min()
        for k in range(1, (p - 1) // 2 + 1):
            worst = max(worst, np.mean(np.abs(Phi - c) ** k) / (((A - c) ** k).real / math.cos(k * math.pi / p)))
print(f" max R2(c) ratio with DC = {worst:.4f} (must be <= 1)")

# ---------------- Part 2: adversarial optimisation over atoms -----------------
def atom_field(beta, xi, p):
    m, n = xi.shape
    Z = np.exp(2j * np.pi * np.array(list(itertools.product(range(p), repeat=n))) / p)   # all of mu_p^n
    return np.prod(1 + xi[None] * Z[:, None, :], axis=2) @ beta

def unpack(v, m, n, S):
    beta = v[:m] * np.exp(1j * v[m:2 * m])
    xi = np.exp(1j * v[2 * m:2 * m + m * n]).reshape(m, n)
    zeta = np.exp(1j * v[2 * m + m * n:2 * m + m * n + len(S)])
    return beta, xi, zeta

def r3_ratio(v, m, n, S, k, p):
    beta, xi, zeta = unpack(v, m, n, S)
    P = atom_field(beta, xi, p); A = beta.sum(); c = cm(P, p).min()
    # slice coefficient: polynomial in w of sum_i beta_i prod_{t in S}(1 + xi_it zeta_t w)
    poly = np.zeros(len(S) + 1, complex)
    for i in range(m):
        q = np.array([1.0 + 0j])
        for j, t in enumerate(S):
            q = np.convolve(q, [1.0, xi[i, t] * zeta[j]])
        poly += beta[i] * q
    R = math.cos(math.pi / p)
    return R ** k * abs(poly[k]) / ((4 / p) * abs(A - c) + 1e-300)

print("=== Part 2: adversarial maximisation of the R3 ratio over atom systems ===")
for (p, n, m, S, k) in [(3, 1, 2, [0], 1), (3, 2, 3, [0, 1], 1), (3, 2, 3, [0, 1], 2), (5, 1, 3, [0], 1), (5, 2, 3, [0, 1], 1),
                        (5, 2, 3, [0, 1], 2), (5, 2, 4, [0], 1), (7, 2, 3, [0, 1], 2), (5, 3, 4, [0, 1, 2], 3), (5, 3, 4, [0, 1, 2], 2)]:
    best = 0
    for start in range(12):
        v0 = np.concatenate([np.abs(rng.standard_normal(m)) + 0.1, rng.uniform(0, 2 * np.pi, m),
                             rng.uniform(0, 2 * np.pi, m * n), rng.uniform(0, 2 * np.pi, len(S))])
        res = minimize(lambda v: -r3_ratio(v, m, n, S, k, p), v0, method='Nelder-Mead',
                       options=dict(maxiter=6000, xatol=1e-10, fatol=1e-12))
        res = minimize(lambda v: -r3_ratio(v, m, n, S, k, p), res.x, method='Powell', options=dict(maxiter=20000))
        best = max(best, -res.fun)
    print(f" p={p} n={n} m={m} S={S} k={k}: max ratio found = {best:.5f}  (claim: <= 1)", flush=True)
