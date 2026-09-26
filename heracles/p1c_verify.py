"""P1C / step 5: verification of every identity and every proved inequality of P1C.md.

Blocks
  V1  the dictionary:  (KL)  <=>  h(y) = sum_l c_l prod_j (1 + y_j v_lj) == A
                             <=>  sum_l c_l prod_j (u_j - v_lj) == A prod_j u_j     (dagger)
  V2  the BIJECTION INEQUALITY WITH A TAIL (the new ingredient (a)):
        |c_l0| prod_j |v_{sigma(j),j} - v_{l0,j}|  >=  |A| - 2^n R
      and its consequence  sum_{l in S} |c_l|^r >= (n+1)^{1-r} max(|A| - 2^n R, 0)^r.
  V3  the assembled hybrid bound  F >= Val(R, s)  on random (KL) systems.
  V4  the LAGRANGE form at N = 2^n:  c_l = Lambda^{(l)}(0) with Lambda^{(l)} the
      multi-affine Lagrange basis at the atoms, and sum_l Lambda^{(l)} == 1.
  V5  the DIAGONAL RELAXATION counterexample, in exact rational/algebraic terms.
"""
import argparse, json, os
from itertools import permutations
import numpy as np
from p1c_core import Xmat, subsets, cost
from p1c_hybrid2 import Val, A_G

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "FINAL_RUNS", "P1C")


def random_system(n, N, rng):
    """Random phases; c = least-squares/null-space solution of (KL) with A = 1.
    Returns None if the phases admit no solution (N < 2^n generic case)."""
    phi = rng.uniform(0, 2 * np.pi, (N, n))
    X = Xmat(phi)
    b = np.zeros(1 << n, dtype=complex); b[0] = 1.0
    c, *_ = np.linalg.lstsq(X.T, b, rcond=None)
    if np.abs(X.T @ c - b).max() > 1e-9:
        return None
    if N > (1 << n):                       # add a random null-space component
        _, sv, Vh = np.linalg.svd(X.T)
        ns = Vh[(1 << n):].conj().T
        t = rng.normal(size=(ns.shape[1],)) + 1j * rng.normal(size=(ns.shape[1],))
        c = c + ns @ (0.4 * t)
    return phi, c


def v1(n, N, trials, rng):
    worst = 0.0
    for _ in range(trials):
        out = random_system(n, N, rng)
        if out is None:
            continue
        phi, c = out
        v = np.exp(1j * phi)
        for _ in range(4):
            y = rng.normal(size=n) + 1j * rng.normal(size=n)
            h = np.sum(c * np.prod(1.0 + y[None, :] * v, axis=1))
            worst = max(worst, abs(h - 1.0))
            u = np.exp(1j * rng.uniform(0, 2 * np.pi, n))
            lhs = np.sum(c * np.prod(u[None, :] - v, axis=1))
            worst = max(worst, abs(lhs - np.prod(u)))
    return worst


def v2(n, N, trials, r, rng):
    """Bijection inequality with a tail, and its Key-Lemma consequence."""
    bad_bij, bad_kl, nchk = -np.inf, -np.inf, 0
    for _ in range(trials):
        out = random_system(n, N, rng)
        if out is None:
            continue
        phi, c = out
        v = np.exp(1j * phi)
        order = np.argsort(np.abs(c))[::-1]
        S = order[:n + 1]
        R = float(np.abs(c[order[n + 1:]]).sum())
        Aprime = 1.0 - 2 ** n * R
        for l0 in S:
            others = [l for l in S if l != l0]
            for sig in permutations(others):
                u = np.array([v[sig[j], j] for j in range(n)])
                lhs = abs(c[l0]) * np.prod([abs(u[j] - v[l0, j]) for j in range(n)])
                bad_bij = max(bad_bij, Aprime - lhs)      # must be <= 0
                nchk += 1
        lhs = float(np.sum(np.abs(c[S]) ** r))
        bad_kl = max(bad_kl, A_G(R, n, r) - lhs)          # must be <= 0
    return bad_bij, bad_kl, nchk


def v3(n, N, trials, r, rng):
    worst = -np.inf
    for _ in range(trials):
        out = random_system(n, N, rng)
        if out is None:
            continue
        phi, c = out
        x = np.sort(np.abs(c))[::-1]
        R = float(x[n + 1:].sum())
        s = float(x[n])
        F = cost(c, r)
        vb = Val(R, s, n, r)
        if np.isfinite(vb):
            worst = max(worst, vb - F)                    # must be <= 0
    return worst


def v4(n, trials, rng):
    """N = 2^n:  c_l = Lambda^{(l)}(0)  and  sum_l Lambda^{(l)} == 1."""
    N = 1 << n
    worst_c, worst_pu = 0.0, 0.0
    for _ in range(trials):
        phi = rng.uniform(0, 2 * np.pi, (N, n))
        X = Xmat(phi)
        if abs(np.linalg.det(X)) < 1e-6:
            continue
        b = np.zeros(N, dtype=complex); b[0] = 1.0
        c = np.linalg.solve(X.T, b)
        Linv = np.linalg.inv(X)            # columns: coefficients of Lambda^{(l)}
        worst_c = max(worst_c, float(np.abs(Linv[0, :] - c).max()))
        pu = Linv.sum(axis=1)              # coefficients of sum_l Lambda^{(l)}
        e0 = np.zeros(N, dtype=complex); e0[0] = 1.0
        worst_pu = max(worst_pu, float(np.abs(pu - e0).max()))
    return worst_c, worst_pu


def v5(r):
    """The diagonal relaxation dies: p_1 = y^2 + 1, p_2 = y^2 - 1, d = (1/2, -1/2).
    Both are monic of degree n = 2 with all roots on the unit circle
    (roots -+ i and -+ 1), so they are legitimate diagonal data; and
    d_1 p_1 + d_2 p_2 == 1 identically.  Cost 2 * (1/2)^r = 2^{1-r} < 3^{1-r}."""
    p1 = np.array([1.0, 0.0, 1.0])
    p2 = np.array([1.0, 0.0, -1.0])
    comb = 0.5 * p1 - 0.5 * p2
    val = 2 * 0.5 ** r
    return dict(combination=comb.tolist(), identity_ok=bool(np.allclose(comb, [0, 0, 1])),
                cost=val, target=3 ** (1 - r), ratio=val / 3 ** (1 - r),
                roots_p1=np.abs(np.roots(p1)).tolist(),
                roots_p2=np.abs(np.roots(p2)).tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=120)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--tag", default="verify")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    rep = {}
    print("V1  dictionary (KL) <=> h == A <=> (dagger)")
    for n, N in [(1, 3), (2, 4), (2, 6), (3, 8), (3, 10)]:
        e = v1(n, N, a.trials, rng)
        rep[f"V1_n{n}_N{N}"] = e
        print(f"   n={n} N={N}: max |residual| = {e:.3e}")
    print("V2  bijection inequality with a tail, and the Key Lemma with a tail")
    for n, N, q in [(2, 5, 2.0), (2, 6, 2.0), (2, 7, 2.5), (3, 9, 2.0), (3, 10, 2.0)]:
        r = q / (n + 1)
        bb, bk, nc = v2(n, N, a.trials, r, rng)
        rep[f"V2_n{n}_N{N}_q{q}"] = dict(worst_bijection_violation=bb,
                                         worst_keylemma_violation=bk, checks=nc)
        print(f"   n={n} N={N} q={q}: worst bijection violation = {bb:.3e} "
              f"({nc} checks), worst Key-Lemma-with-tail violation = {bk:.3e} "
              f"(both must be <= 0)")
    print("V3  the assembled hybrid bound F >= Val(R, s)")
    for n, N, q in [(2, 5, 2.0), (2, 6, 2.0), (2, 7, 2.0), (3, 9, 2.0), (3, 10, 2.0)]:
        r = q / (n + 1)
        w = v3(n, N, a.trials, r, rng)
        rep[f"V3_n{n}_N{N}_q{q}"] = w
        print(f"   n={n} N={N} q={q}: worst (Val - F) = {w:.3e}  (must be <= 0)")
    print("V4  Lagrange form at N = 2^n")
    for n in (1, 2, 3):
        wc, wp = v4(n, a.trials, rng)
        rep[f"V4_n{n}"] = dict(c_vs_Lambda0=wc, partition_of_unity=wp)
        print(f"   n={n}: max |c_l - Lambda^(l)(0)| = {wc:.3e}, "
              f"max |sum_l Lambda^(l) - 1| = {wp:.3e}")
    print("V5  the diagonal relaxation counterexample")
    for q in (2.0, 2.5):
        d = v5(q / 3.0)
        rep[f"V5_q{q}"] = d
        print(f"   q={q}: identity ok={d['identity_ok']}, |roots|={d['roots_p1']}/"
              f"{d['roots_p2']}, cost={d['cost']:.9f} vs target {d['target']:.9f} "
              f"-> ratio {d['ratio']:.9f}")
    with open(os.path.join(OUT, f"{a.tag}.json"), "w") as f:
        json.dump(rep, f, indent=1, default=float)


if __name__ == "__main__":
    main()
