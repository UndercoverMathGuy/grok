"""Sanity check for R2 (entropic Lemma 7): for every ALIGNED interference-free atom configuration
(weights w_l > 0, unit xi_lt, sum_l w_l prod_{t in T} xi_lt = 0 for all nonempty T in [n]),
the escort entropy H(esc_r(w)) >= log nu for every r in (0,1], equivalently psi(r) = log sum w^r >= (1-r) log nu.
Configurations are found by least squares from random starts (exact zeros up to 1e-13), with m = nu .. nu+3 atoms.
Also probes NON-aligned IF configurations (complex weights) for the same entropy bound."""
import itertools, math
import numpy as np
from scipy.optimize import least_squares

rng = np.random.default_rng(7)
out = []


def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); out.append(s)


def subsets(n):
    return [T for r in range(1, n + 1) for T in itertools.combinations(range(n), r)]


def resid(x, m, n, aligned):
    if aligned:
        a = x[:m]; th = x[m:m + m * n].reshape(m, n); ph = np.zeros(m)
    else:
        a = x[:m]; th = x[m:m + m * n].reshape(m, n); ph = x[m + m * n:]
    w = np.exp(a) / np.exp(a).sum()
    beta = w * np.exp(1j * ph)
    R = []
    for T in subsets(n):
        v = (beta * np.exp(1j * th[:, list(T)].sum(1))).sum()
        R += [v.real, v.imag]
    if not aligned:   # normalise |sum beta| bounded away from 0 so the config is a genuine feature
        R += [max(0.0, 0.3 - abs(beta.sum()))]
    return np.array(R)


def H(w):
    w = w[w > 0]; return float(-(w * np.log(w)).sum())


def esc(w, r):
    e = w ** r; return e / e.sum()


rs = np.concatenate([np.linspace(0.02, 1.0, 50)])
for aligned in [True, False]:
    say(f"===== {'ALIGNED' if aligned else 'NON-ALIGNED'} IF configurations =====")
    for n in [1, 2, 3, 4]:
        nu = n + 1
        for m in range(nu, nu + 4):
            found = 0; worst = np.inf; worst_l7 = np.inf; minH = np.inf; maxdev = 0
            for trial in range(60):
                dim = m + m * n + (0 if aligned else m)
                x0 = np.concatenate([rng.normal(0, 1.0, m), rng.uniform(0, 2 * np.pi, m * n),
                                     ([] if aligned else rng.uniform(0, 2 * np.pi, m))])
                sol = least_squares(resid, x0, args=(m, n, aligned), xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=4000)
                if np.abs(sol.fun).max() > 1e-12:
                    continue
                a = sol.x[:m]; w = np.exp(a) / np.exp(a).sum()
                if not aligned:
                    ph = sol.x[m + m * n:]
                    alpha = abs((w * np.exp(1j * ph)).sum())   # |beta| / sum |beta_l|
                    if alpha > 1 - 1e-6:
                        continue   # effectively aligned
                found += 1
                maxdev = max(maxdev, w.max() * len(w) - 1)
                for r in rs:
                    e = esc(w, r)
                    worst = min(worst, H(e) - math.log(nu))
                    if aligned:
                        worst_l7 = min(worst_l7, math.log((w ** r).sum()) - (1 - r) * math.log(nu))
                minH = min(minH, H(w))
            if found:
                say(f"n={n} m={m}: {found} configs; non-uniformity max(w)*m-1 up to {maxdev:.3f}; "
                    f"min_r [H(esc_r)-log nu] = {worst:+.3e}" + (f"; min_r [psi(r)-(1-r)log nu] = {worst_l7:+.3e}" if aligned else "")
                    + f"; min Shannon H - log nu = {minH - math.log(nu):+.3e}")
            else:
                say(f"n={n} m={m}: no exact configuration found")
open("check_entropic_l7.out", "w").write("\n".join(out) + "\n")
