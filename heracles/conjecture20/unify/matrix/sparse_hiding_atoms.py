"""Atomic version of the sparse-prime obstruction: own modes priced EXACTLY by atoms (Q = sum|beta_i|^r), all non-own
modes free.  We optimise a perturbation of the nu-atom IF block of feature l (at fixed size eps of the sign-flip vector)
to get a positive bonus  y - nu^{1-s} Q^s  at clean amplitude y, then hide its interference from the binding set exactly
(free modes) and evaluate the exact margin.  Output: ratio > 1 means the relaxation (free non-own modes) beats IF."""
import sys
import numpy as np
from scipy.optimize import minimize
sys.path.insert(0, '/home/user/grok/heracles/conjecture20/mechanism_scripts')
from prog import solve as prog_solve, kappa
from mrel import subsets
from sparse_hiding import build_fields, margin


def atoms_from(v, nu, n, y):
    om = np.exp(2j * np.pi / nu)
    beta = np.full(nu, y / nu, dtype=complex) * (1 + v[:nu] + 1j * v[nu:2 * nu])
    xi = np.array([[om ** k] * n for k in range(nu)], dtype=complex) * np.exp(1j * v[2 * nu:].reshape(nu, n))
    A = beta.sum(); beta = beta * (np.conj(A) / abs(A)) * y / abs(A)
    return beta, xi


def signflips(beta, xi, n):
    return {T: sum(b * np.prod([x[t] for t in T]) for b, x in zip(beta, xi)) for T in subsets(n)}


def best_perturbation(nu, n, s, y, eps, rng, tries=20):
    r = 1 / s
    dim = 2 * nu + nu * n
    best = (-np.inf, None)

    def neg_bonus(w):
        w = eps * w / np.linalg.norm(w)
        beta, xi = atoms_from(w, nu, n, y)
        Q = np.sum(np.abs(beta) ** r)
        return -(y - nu ** (1 - s) * Q ** s)

    for t in range(tries):
        w0 = rng.normal(size=dim)
        res = minimize(neg_bonus, w0, method='Nelder-Mead', options={'maxiter': 4000, 'xatol': 1e-10, 'fatol': 1e-14})
        if -res.fun > best[0]:
            best = (-res.fun, eps * res.x / np.linalg.norm(res.x))
    return best


def run(p, n, b, rng):
    nu = n + 1; s = nu / b; r = 1 / s
    G, ystar, _ = prog_solve(p, s, restarts=8)
    cov = kappa(p) @ ystar
    Bh = [d for d in range(1, (p - 1) // 2 + 1) if cov[d - 1] < G * (1 + 1e-7)]
    B = set(Bh) | set((-d) % p for d in Bh)
    S = [j for j in range(1, (p - 1) // 2 + 1) if ystar[j - 1] > 1e-9]
    IFcost = sum(nu ** (1 - r) * yy ** r for yy in ystar if yy > 0)
    print(f"p={p} n={n} b={b}: support {S}, binding {sorted(B)}")
    for l in S:
        y = ystar[l - 1]
        for eps in [0.08, 0.04, 0.02]:
            bonus, w = best_perturbation(nu, n, s, y, eps, rng)
            beta, xi = atoms_from(w, nu, n, y)
            sf = signflips(beta, xi, n)
            own = {T: np.conj(v) for T, v in sf.items()}
            Q = np.sum(np.abs(beta) ** r)
            phi = build_fields(p, n, ystar, l, own, B)
            h, _ = margin(p, n, phi)
            Qt = IFcost - nu ** (1 - r) * y ** r + Q
            ratio = h / (nu ** (1 - s) * G * Qt ** s)
            E = np.sqrt(sum(abs(v) ** 2 for T, v in sf.items() if T != ()))
            print(f"   l={l} eps={eps}: bonus {bonus:+.3e} (bonus*y/E^2 = {bonus*y/E**2:.4f}), |A|={abs(sf[()]):.4f}, ratio {ratio:.7f}")


if __name__ == '__main__':
    rng = np.random.default_rng(11)
    for (p, n, b) in [(11, 2, 2.0), (13, 2, 2.0), (11, 3, 2.0)]:
        run(p, n, b, rng)
