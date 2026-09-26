"""VERIFY link 6(e): a direct, fresh assault on P1 itself.

P1:  for every N and every exact (KL) system,  sum_l |c_l|^r >= (n+1)^{1-r} |A|^r.

Structural note this script respects (and which the earlier stress run stumbled on): (KL) is 2^n
complex equations, so for GENERIC phases a solution needs N >= 2^n.  At n >= 3 the range
n+2 <= N <= 2^n - 1 is therefore empty for generic phases and the real content of P1 starts at
N = 2^n.  So we sweep N from n+1 up to 2^n + 3 and, for each phase configuration, minimise
sum|c_l|^r over the whole affine solution set (SLSQP on the null-space coordinates), then also
minimise over the phases.
"""
import itertools, json, os
import numpy as np
from scipy.optimize import minimize

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "FINAL_RUNS", "VERIFY")
os.makedirs(OUT, exist_ok=True)


def moments(v, n):
    """X[l, T] = prod_{j in T} v_lj over all 2^n subsets T (empty first)."""
    Ts = [t for k in range(n + 1) for t in itertools.combinations(range(n), k)]
    Ts.sort(key=lambda t: (len(t), t))
    X = np.ones((v.shape[0], len(Ts)), dtype=complex)
    for i, t in enumerate(Ts):
        if t:
            X[:, i] = np.prod(v[:, list(t)], axis=1)
    return X


def solve_kl(v, n):
    X = moments(v, n)                     # (N, 2^n)
    M = X.T                               # (2^n, N)
    b = np.zeros(M.shape[0], dtype=complex); b[0] = 1.0
    c0, *_ = np.linalg.lstsq(M, b, rcond=None)
    if np.linalg.norm(M @ c0 - b) > 1e-9:
        return None, None
    u, s, vh = np.linalg.svd(M)
    rank = int((s > 1e-9 * max(s[0], 1e-300)).sum())
    ns = vh.conj().T[:, rank:]
    return c0, ns


def min_cost(v, n, r, restarts=6, rng=None):
    c0, ns = solve_kl(v, n)
    if c0 is None:
        return None
    if ns.shape[1] == 0:
        return float(np.sum(np.abs(c0) ** r))
    d = ns.shape[1]

    def f(t):
        c = c0 + ns @ (t[:d] + 1j * t[d:])
        return float(np.sum(np.abs(c) ** r + 1e-300))
    best = f(np.zeros(2 * d))
    for _ in range(restarts):
        t0 = rng.normal(0, 0.5, 2 * d) if rng is not None else np.zeros(2 * d)
        res = minimize(f, t0, method="Nelder-Mead",
                       options=dict(maxiter=4000, xatol=1e-11, fatol=1e-13))
        best = min(best, res.fun)
    return best


def families(n, N, rng, fam):
    if fam == 0:
        return np.exp(2j * np.pi * rng.random((N, n)))
    if fam == 1:                                   # block on the first n+1, rest free
        w = np.exp(-2j * np.pi * np.arange(n + 1) / (n + 1))
        v = np.zeros((N, n), dtype=complex)
        v[:min(N, n + 1)] = w[:min(N, n + 1), None] * np.ones((1, n))
        if N > n + 1:
            v[n + 1:] = np.exp(2j * np.pi * rng.random((N - n - 1, n)))
        return v
    if fam == 2:                                   # exact sign rule (the 2^n tensor grid)
        sg = np.array(list(itertools.product([1, -1], repeat=n)), dtype=complex)
        rot = np.exp(2j * np.pi * rng.random(n))
        base = sg * rot
        if N <= len(base):
            return base[:N]
        extra = np.exp(2j * np.pi * rng.random((N - len(base), n)))
        return np.vstack([base, extra])
    if fam == 3:                                   # tensor grid of m-th roots per slot
        m = int(rng.integers(2, 4))
        pts = np.array(list(itertools.product(range(m), repeat=n)))
        base = np.exp(2j * np.pi * pts / m) * np.exp(2j * np.pi * rng.random(n))
        if N <= len(base):
            return base[:N]
        return np.vstack([base, np.exp(2j * np.pi * rng.random((N - len(base), n)))])
    if fam == 4:                                   # coalescing atoms
        v = np.exp(2j * np.pi * rng.random((N, n)))
        v[1] = v[0] * np.exp(1j * rng.normal(0, 10.0 ** (-rng.integers(1, 7)), n))
        return v
    # perturbed sign rule
    sg = np.array(list(itertools.product([1, -1], repeat=n)), dtype=complex)
    base = sg[rng.choice(len(sg), size=N, replace=True)]
    return base * np.exp(1j * rng.normal(0, 0.15, (N, n)))


def main():
    rep = []
    print(f"{'cell':<12}{'N':>4}{'worst sum|c|^r / (n+1)^(1-r)':>32}{'fam':>5}{'residual':>12}")
    for n, r, lab in [(2, 2 / 3, "nu=3,q=2"), (3, 1 / 2, "nu=4,q=2"), (4, 2 / 5, "nu=5,q=2")]:
        tgt = (n + 1) ** (1 - r)
        rng = np.random.default_rng(11 + n)
        overall = np.inf
        for N in range(n + 1, 2 ** n + 4):
            worst = np.inf; wf = None
            trials = 600 if n == 2 else (250 if n == 3 else 60)
            for t in range(trials):
                fam = t % 6
                v = families(n, N, rng, fam)
                c = min_cost(v, n, r, restarts=(4 if n < 4 else 2), rng=rng)
                if c is None:
                    continue
                if c / tgt < worst:
                    worst = c / tgt; wf = fam
            if np.isfinite(worst):
                print(f"{lab:<12}{N:>4}{worst:>32.10f}{wf:>5}{'':>12}")
                overall = min(overall, worst)
                rep.append(dict(cell=lab, N=N, ratio=worst, fam=wf))
                if worst < 1 - 1e-8:
                    print("   *** P1 VIOLATED ***")
        print(f"  -> {lab}: overall min = {overall:.10f}\n")
    with open(os.path.join(OUT, "p1_assault.json"), "w") as f:
        json.dump(rep, f, indent=1, default=float)
    print("wrote", os.path.join(OUT, "p1_assault.json"))


if __name__ == "__main__":
    main()
