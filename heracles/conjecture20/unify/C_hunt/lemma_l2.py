# Sanity check of Lemma L2 (pair detection for consistent character families), any n:
#   class-C network, fields Phi_l = A_l + sum_T a_{l,T} chi_{l,T};  Lambda = {T : some a_{l,T} != 0}.
#   If some v in Z_p^n has 1_T . v = 1 (mod p) for all T in Lambda, then for every delta:
#     min( min_x g(x, delta), min_x g(x, -delta) ) <= sum_l Re(A_l) kappa_l(delta) - ( sum_{l,T} |a_{l,T}|^2 kappa_l(delta) )^{1/2}.
# Checks: (1) random consistent networks (several Lambda, n = 2, 3): the inequality at every delta, and the exact shift
#   identity D_{-delta}(x - t v) = -D_delta(x);  (2) inconsistent Lambda (all three characters at n = 2): search for
#   networks where the pair minimum EXCEEDS the bound (the lemma's hypothesis is necessary).
import itertools, json
import numpy as np
import hlib as H

rng = np.random.default_rng(7)


def random_net(p, n, S, Lam, mode):
    """mode 'atoms': genuine class-C atom systems whose a_T vanish off Lambda only approximately is hard;
    so we work directly at the field level (the lemma is a statement about fields): random A_l and a_{l,T}, T in Lambda."""
    A = {l: rng.uniform(0.2, 1.0) * np.exp(1j * rng.uniform(-0.3, 0.3)) for l in S}
    a = {(l, T): (rng.uniform(0, 1.2) * np.exp(1j * rng.uniform(0, 2 * np.pi))) for l in S for T in Lam}
    return A, a


def gaps(p, n, A, a):
    X = np.indices((p,) * n).reshape(n, -1).T
    d = np.arange(1, p)
    g = np.zeros((X.shape[0], p - 1))
    for l, Al in A.items():
        Phi = np.full(X.shape[0], Al, complex)
        for (ll, T), aT in a.items():
            if ll != l: continue
            Phi += aT * np.exp(2j * np.pi * 2 * l * (X @ np.array(T)) / p)
        c = 1 - np.exp(-2j * np.pi * l * d / p)
        g += np.real(Phi[:, None] * c[None, :])
    return X, g


def bound(p, A, a):
    d = np.arange(1, p)
    clean = sum(np.real(Al) * (1 - np.cos(2 * np.pi * l * d / p)) for l, Al in A.items())
    L2 = np.sqrt(sum(abs(aT) ** 2 * (1 - np.cos(2 * np.pi * l * d / p)) for (l, T), aT in a.items()))
    return clean - L2


def consistent(p, Lam, n):
    for v in itertools.product(range(p), repeat=n):
        if all(sum(v[t] for t in range(n) if T[t]) % p == 1 for T in Lam): return v
    return None


res = {}
for (p, n) in [(5, 2), (7, 2), (11, 2), (13, 2), (5, 3), (7, 3)]:
    F = (p - 1) // 2
    Ts = [T for T in itertools.product([0, 1], repeat=n) if any(T)]
    fams = [L for k in range(1, len(Ts) + 1) for L in itertools.combinations(Ts, k)]
    worst_c = -np.inf; worst_i = -np.inf; nc = ni = 0; shift_err = 0.0
    for trial in range(300):
        Lam = list(fams[int(rng.integers(len(fams)))])
        S = sorted(rng.choice(np.arange(1, F + 1), size=int(rng.integers(1, F + 1)), replace=False).tolist())
        A, a = random_net(p, n, S, Lam, 'field')
        X, g = gaps(p, n, A, a)
        pairmin = np.minimum(g.min(0), g.min(0)[::-1])           # delta and p - delta
        B = bound(p, A, a)
        excess = (pairmin - B) / (np.abs(B).max() + 1e-12)        # > 0 means the bound fails
        v = consistent(p, Lam, n)
        if v is not None:
            nc += 1; worst_c = max(worst_c, excess.max())
            # shift identity at delta = 1
            t = (1 * pow(2, -1, p)) % p
            Xs = (X - t * np.array(v)) % p
            idx = np.ravel_multi_index(Xs.T, (p,) * n)
            d = np.arange(1, p)
            cl = sum(np.real(Al * (1 - np.exp(-2j * np.pi * l * d / p))) for l, Al in A.items())
            Dp = g[:, 0] - cl[0]; Dm = g[idx, p - 2] - cl[p - 2]
            shift_err = max(shift_err, np.abs(Dm + Dp).max())
        else:
            ni += 1; worst_i = max(worst_i, excess.max())
    res[(p, n)] = (nc, worst_c, shift_err, ni, worst_i)
    print(f"p={p} n={n}: consistent families tested={nc}, max relative excess of pair-min over the bound={worst_c:.3e} "
          f"(must be <= 0), shift identity error={shift_err:.1e}; inconsistent tested={ni}, max excess={worst_i:.3e}", flush=True)

# (2) targeted: inconsistent family at n = 2 (all three characters) -- maximise the excess (pair-min minus bound),
#     relative to the L2 term, by random search + Nelder-Mead on the fluctuation coefficients (A fixed).
from scipy.optimize import minimize
for p in [7, 11, 13]:
    F = (p - 1) // 2; n = 2
    Ts = [(1, 0), (0, 1), (1, 1)]
    S = list(range(1, F + 1))
    best = np.inf
    for trial in range(12):
        A = {l: 1.0 for l in S}
        z0 = rng.standard_normal(2 * len(S) * 3) * 0.3

        def unpack(z):
            c = z[:len(S) * 3] + 1j * z[len(S) * 3:]
            return {(l, T): c[k * 3 + j] for k, l in enumerate(S) for j, T in enumerate(Ts)}

        def obj(z):
            a = unpack(z)
            X, g = gaps(p, n, A, a)
            pm = np.minimum(g.min(0), g.min(0)[::-1])
            d = np.arange(1, p)
            L2 = np.sqrt(sum(abs(aT) ** 2 * (1 - np.cos(2 * np.pi * l * d / p)) for (l, T), aT in a.items()))
            clean = sum((1 - np.cos(2 * np.pi * l * d / p)) for l in S)
            # ratio of detection to L2 at the worst offset: (clean - pairmin) / L2 ; < 1 means the L2 bound fails
            return np.min((clean - pm) / np.maximum(L2, 1e-12))
        r = minimize(obj, z0, method='Nelder-Mead', options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-10})
        best = min(best, r.fun)
    print(f"p={p} n=2 inconsistent (all three characters): min over designs of detection/L2 found = {best:.4f} "
          f"(< 1 means flat-bottom below the L2 aggregate is possible)", flush=True)
