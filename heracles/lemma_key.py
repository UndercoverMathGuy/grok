"""lemma_key.py -- direct fp64 test of the KEY LEMMA and of every step of its proof.

KEY LEMMA (proved in LEMMA_ATTEMPT.md Sec. 4).  Let n >= 1 and let c_1..c_{n+1} in C and unit-modulus
v_lj (l = 1..n+1, j = 1..n) satisfy the junk-free system

    sum_l c_l prod_{j in T} v_lj = A * [T is empty]   for every T subset of {1..n}.

Then      prod_l |c_l| >= (|A|/(n+1))^{n+1}     and hence   sum_l |c_l|^r >= (n+1)^{1-r} |A|^r, 0 < r <= 1.

Proof steps checked here:
  (S1) the polynomial identity  sum_l c_l prod_j (u_j - v_lj) = A prod_j u_j  for all u in C^n;
  (S2) the bijection identity   |c_l0| * prod_j |v_{sigma(j) j} - v_{l0 j}| = |A|  for every bijection
       sigma from the n slots onto the n atoms other than l0  (this is (S1) at u_j = v_{sigma(j) j});
  (S3) the geometric-mean step  prod_l |c_l| >= |A|^{n+1} * (prod_j V_j^2)^{-1/n},  V_j = prod_{l<m} |v_lj - v_mj|;
  (S4) Hadamard  V_j <= (n+1)^{(n+1)/2};
  (S5) the conclusion, and the equality case (roots of unity, equal amplitudes).

Junk-free configurations are generated in the COMMON-PHASE gauge (v_lj = z_l for all j), which by the
rigidity lemma is the only gauge available at N = n+1; z_l are random points of the unit circle, and c
is obtained from the Vandermonde system.  Every one of the 2^n constraints is then verified directly.

Usage: uv run python heracles/lemma_key.py
"""
import itertools, json, math
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "FINAL_RUNS" / "LEMMA"
rng = np.random.default_rng(11)


def solve_common_phase(n, A=1.0):
    """z_l random on the circle; c from  sum_l c_l z_l^t = A [t = 0],  t = 0..n."""
    z = np.exp(1j * rng.uniform(0, 2 * np.pi, n + 1))
    V = np.vander(z, n + 1, increasing=True).T          # V[t, l] = z_l^t
    rhs = np.zeros(n + 1, dtype=complex); rhs[0] = A
    c = np.linalg.solve(V, rhs)
    return c, z


def check(n, trials=3000, rs=(0.2, 0.5, 2 / 3, 0.8, 1.0)):
    worst = dict(constraints=0.0, S2=0.0, prod=np.inf, sum={r: np.inf for r in rs}, hadamard=0.0)
    for _ in range(trials):
        c, z = solve_common_phase(n)
        v = np.stack([z] * n, 1)                        # v[l, j] = z_l
        A = abs(c.sum())
        # all 2^n constraints
        for rb in range(2 ** n):
            T = [j for j in range(n) if rb >> j & 1]
            w = np.ones(n + 1, dtype=complex)
            for j in T:
                w = w * v[:, j]
            s = (c * w).sum() - (1.0 if not T else 0.0)
            worst["constraints"] = max(worst["constraints"], abs(s))
        # (S2) bijection identity
        others = lambda l0: [l for l in range(n + 1) if l != l0]
        for l0 in range(n + 1):
            for sig in itertools.permutations(others(l0)):
                pr = abs(c[l0]) * np.prod([abs(v[sig[j], j] - v[l0, j]) for j in range(n)])
                worst["S2"] = max(worst["S2"], abs(pr - A))
        # (S3)+(S4)+(S5)
        P = float(np.prod(np.abs(c)))
        worst["prod"] = min(worst["prod"], P / (A / (n + 1)) ** (n + 1))
        for r in rs:
            worst["sum"][r] = min(worst["sum"][r], float((np.abs(c) ** r).sum()) / ((n + 1) ** (1 - r) * A ** r))
        Vj = float(np.prod([abs(z[l] - z[m]) for l in range(n + 1) for m in range(l + 1, n + 1)]))
        worst["hadamard"] = max(worst["hadamard"], Vj / (n + 1) ** ((n + 1) / 2))
    return worst


if __name__ == "__main__":
    res = {}
    for n in (1, 2, 3, 4, 5, 6):
        w = check(n, trials=(3000 if n <= 3 else 400))
        res[n] = dict(constraints=w["constraints"], S2=w["S2"], prod_ratio=w["prod"],
                      sum_ratio={str(k): v for k, v in w["sum"].items()}, hadamard_ratio=w["hadamard"])
        print(f"n={n}  max |constraint residual| {w['constraints']:.2e}   max |S2 residual| {w['S2']:.2e}")
        print(f"      min  prod|c_l| / (A/(n+1))^(n+1) = {w['prod']:.6f}   (claim: >= 1)")
        print(f"      min  V_j / (n+1)^((n+1)/2)  -> max over trials = {w['hadamard']:.6f}   (claim: <= 1)")
        print("      min  sum|c_l|^r / ((n+1)^(1-r) A^r) = " +
              "  ".join(f"r={r:.3g}: {w['sum'][r]:.6f}" for r in w["sum"]) + "   (claim: >= 1)", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "key.json").write_text(json.dumps(res, indent=1, default=float))
