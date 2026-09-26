"""P3C step 3: the PARTITION test-density family and the criterion (PFP_A).

See P3C.md SS3-SS4.  For a partition `P = {B_1,...,B_s}` of the operand slots
`{1..n}` and a probability density `nu_i` on `Z_p` for each block, the density

    w(a) = prod_i  nu_i( sum_{j in B_i} a_j ) / p^{|B_i| - 1}

is a probability density on `Z_p^n` whose Fourier coefficient at the junk mode
`2k * 1_T` is

    prod_{i : B_i subset T} nuhat_i(2k)     if every block meets T in nothing or
                                            in all of it,
    0                                       otherwise.

So the frequency-`k` contribution seen by this density is the multi-affine form

    g_P(t) = sum_{U subset {1..s}}  C_{B_U} * prod_{i in U} t_i ,   B_U = union of blocks in U,
             t_i = nuhat_i(2k),   |t_i| <= A*,  phases FREE per (block, frequency).

`P` = all singletons recovers Theorem S's product-of-Riesz density; `P` = one
block recovers P3A's fibre density (its F5-4).  This script measures

    R_P(pop, A) := kappa0 * ( min over phases of Re g_P )_+ / ( sum_l |c_l|^r )^{1/r}

("how well this population does against that density, in units of the junk-free
block of the same cost"), and then the mixture threshold

    A_lam(n,r) := inf { A : exists lam on partitions with
                             sum_P lam_P R_P(pop,A)^r <= 1 for EVERY population } ,

which by a minimax argument (P3C.md SS4.2) is exactly the condition under which
(T) follows for every K and every junk level.

  uv run python p3c_part.py --cells nu3q2 nu4q2 --mode scan
"""
import argparse, itertools, json, os
import numpy as np
from scipy.optimize import minimize, linprog

import p3c_core as C

CELLS = {"nu3q2": (3, 2.0), "nu3q2.5": (3, 2.5), "nu4q2": (4, 2.0),
         "nu4q3": (4, 3.0), "nu5q2": (5, 2.0), "nu5q4": (5, 4.0)}


# ------------------------------------------------------------------ partitions
def partitions(n):
    """All set partitions of {0..n-1}, each as a tuple of bitmask blocks."""
    if n == 0:
        return [()]
    out = []
    for prev in partitions(n - 1):
        j = n - 1
        for i in range(len(prev)):
            new = list(prev)
            new[i] = new[i] | (1 << j)
            out.append(tuple(sorted(new)))
        out.append(tuple(sorted(prev + ((1 << j),))))
    return sorted(set(out))


def part_coef(Ct, blocks, A):
    """Coefficients D_U of g_P: D_U = C_{union of blocks in U} * A^{|U|}."""
    s = len(blocks)
    D = np.zeros(1 << s, dtype=complex)
    for u in range(1 << s):
        t = 0
        k = 0
        for i in range(s):
            if (u >> i) & 1:
                t |= blocks[i]
                k += 1
        D[u] = Ct[t] * A ** k
    return D


# ------------------------------------------- exact min over phases (vectorised)
_BITS = {}


def bits(s):
    if s not in _BITS:
        _BITS[s] = np.array([[(t >> j) & 1 for j in range(s)]
                             for t in range(1 << s)], dtype=float)
    return _BITS[s]


def min_re(D, s, restarts=8, sweeps=30, rng=None):
    """min over phi in T^s of Re sum_U D_U exp(i sum_{i in U} phi_i).

    Exact cyclic coordinate descent: the form is affine in each e^{i phi_i}, so
    the per-coordinate minimum is Re P - |Q| at e^{i phi_i} = -conj(Q)/|Q|.
    All restarts are swept in parallel.  Verified against a brute-force grid.
    """
    if s == 0:
        return float(D[0].real), np.zeros(0)
    B = bits(s)
    if rng is None:
        rng = np.random.default_rng(0)
    phi = np.zeros((restarts, s))
    if restarts > 1:
        phi[1] = np.pi
    if restarts > 2:
        phi[2:] = rng.uniform(0, 2 * np.pi, (restarts - 2, s))
    masks = [B[:, j] == 1 for j in range(s)]
    for _ in range(sweeps):
        for j in range(s):
            ph = np.exp(1j * (phi @ B.T))                  # (R, 2^s)
            m = masks[j]
            Q = (ph[:, m] * D[m]).sum(axis=1) * np.exp(-1j * phi[:, j])
            ok = np.abs(Q) > 1e-300
            phi[ok, j] = np.angle(-np.conj(Q[ok]))
    vals = np.real(np.exp(1j * (phi @ B.T)) @ D)
    k = int(np.argmin(vals))
    return float(vals[k]), phi[k]


def R_all(c, V, A, parts, r, n, restarts=6, sweeps=25, rng=None):
    """R_P(pop,A) for every partition P in `parts`."""
    Ct = C.coeffs(c, V)
    cost = float(np.sum(np.abs(c) ** r))
    if cost <= 1e-300:
        return np.full(len(parts), np.inf)
    den = (cost ** (1.0 / r)) / C.kappa0(n, r)
    out = np.empty(len(parts))
    for i, blocks in enumerate(parts):
        D = part_coef(Ct, blocks, A)
        v, _ = min_re(D, len(blocks), restarts=restarts, sweeps=sweeps, rng=rng)
        out[i] = max(v, 0.0) / den
    return out


# --------------------------------------------------------------- the adversary
def worst_for_lambda(lam, n, r, A, parts, starts, rng, N_list=(1, 2, 3, 4, 5)):
    """max over populations of sum_P lam_P R_P(pop,A)^r, and the argmax."""
    best = (-np.inf, None, None)
    for N in N_list:
        for s in range(starts):
            if s == 0:
                c0 = np.zeros(N, dtype=complex); c0[0] = 1.0
                z0 = C.pack(c0, np.zeros((N, n)))
            elif s == 1 and N >= 2:
                # the "m-block": N atoms at N-th roots of unity in every slot
                w = np.exp(-2j * np.pi / N)
                c0 = np.full(N, 1.0 / N, dtype=complex)
                th0 = np.array([[-2 * np.pi * l / N] * n for l in range(N)])
                z0 = C.pack(c0, th0)
            else:
                c0 = rng.standard_normal(N) + 1j * rng.standard_normal(N)
                z0 = C.pack(c0, rng.uniform(0, 2 * np.pi, (N, n)))

            def neg(z):
                c, V = C.unpack(z, N, n)
                R = R_all(c, V, A, parts, r, n, restarts=3, sweeps=8, rng=rng)
                return -float(np.dot(lam, R ** r))

            res = minimize(neg, z0, method="Nelder-Mead",
                           options={"maxiter": 500, "xatol": 1e-8, "fatol": 1e-10})
            c, V = C.unpack(res.x, N, n)
            R = R_all(c, V, A, parts, r, n, restarts=16, sweeps=60, rng=rng)
            val = float(np.dot(lam, R ** r))
            if val > best[0]:
                best = (val, res.x.copy(), (N, R.copy()))
    return best


def seed_pool(n, r, A, parts, npop, rng, N_list=(1, 2, 3, 4, 5)):
    """A pool of R-vectors from random and structured populations, to start the
    LP from.  Structured seeds: the `m`-families (m atoms at m-th roots of unity
    in every slot), the block, and single atoms."""
    pool = []
    for m in range(1, n + 3):
        c0 = np.full(m, 1.0 / m, dtype=complex)
        V = np.exp(1j * np.array([[-2 * np.pi * l / m] * n for l in range(m)]))
        pool.append(R_all(c0, V, A, parts, r, n, restarts=10, sweeps=40, rng=rng))
    for _ in range(npop):
        N = int(rng.choice(N_list))
        c0 = rng.standard_normal(N) + 1j * rng.standard_normal(N)
        V = np.exp(1j * rng.uniform(0, 2 * np.pi, (N, n)))
        pool.append(R_all(c0, V, A, parts, r, n, restarts=3, sweeps=10, rng=rng))
    return pool


def cutting_plane(n, r, A, starts=40, rounds=8, seed=11, N_list=(1, 2, 3, 4, 5),
                  npop=0):
    """Is there a lam on partitions with sum_P lam_P R_P(pop,A)^r <= 1 for all pop?

    Cutting planes: keep a pool of populations, solve the LP
    min_{lam in simplex} max_{pool} sum_P lam_P R_P^r, then attack the winner.
    Returns (value, lam, pool of R-vectors).
    """
    rng = np.random.default_rng(seed)
    parts = partitions(n)
    P = len(parts)
    pool = [x ** r for x in seed_pool(n, r, A, parts, npop, rng, N_list)] if npop \
        else []
    lam = np.full(P, 1.0 / P)
    val = np.nan
    for it in range(rounds + (1 if pool else 0)):
        if it > 0 or not pool:
            v, z, info = worst_for_lambda(lam, n, r, A, parts, starts, rng, N_list)
            pool.append(info[1] ** r)
        else:
            v = float(np.max(np.array(pool) @ lam))
        # LP: minimise u s.t. sum_P lam_P Rrow_P <= u, lam >= 0, sum lam = 1
        M = np.array(pool)                      # (m, P)
        m = M.shape[0]
        # variables (lam_1..lam_P, u)
        cvec = np.zeros(P + 1); cvec[-1] = 1.0
        Aub = np.hstack([M, -np.ones((m, 1))])
        bub = np.zeros(m)
        Aeq = np.zeros((1, P + 1)); Aeq[0, :P] = 1.0
        beq = [1.0]
        res = linprog(cvec, A_ub=Aub, b_ub=bub, A_eq=Aeq, b_eq=beq,
                      bounds=[(0, 1)] * P + [(None, None)], method="highs")
        lam = res.x[:P]
        val = float(res.x[-1])
        print(f"   round {it}: adversary {v:.6f}  ->  LP value {val:.6f}  "
              f"lam={np.round(lam, 4)}", flush=True)
    return val, lam, np.array(pool), parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="*", default=["nu3q2", "nu4q2"])
    ap.add_argument("--A", type=float, nargs="*", default=None)
    ap.add_argument("--starts", type=int, default=30)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--npop", type=int, default=3000)
    ap.add_argument("--Nmax", type=int, default=5)
    ap.add_argument("--out", default="FINAL_RUNS/P3C/part.json")
    a = ap.parse_args()

    out = {}
    for key in a.cells:
        nu, q = CELLS[key]
        n, r = nu - 1, q / nu
        As = a.A if a.A else [0.30, 0.40, 0.50, 0.60]
        rows = []
        for A in As:
            print(f"[{key}] A = {A}")
            val, lam, pool, parts = cutting_plane(
                n, r, A, starts=a.starts, rounds=a.rounds, npop=a.npop,
                N_list=tuple(range(1, a.Nmax + 1)))
            rows.append({"A": float(A), "lp_value": val,
                         "lam": lam.tolist(),
                         "parts": [list(b) for b in parts],
                         "holds": bool(val <= 1 + 1e-9)})
            print(f"== {key} A={A}: LP value {val:.9f} "
                  f"{'OK' if val <= 1 + 1e-9 else 'FAILS'}\n", flush=True)
        out[key] = {"nu": nu, "q": q, "r": r, "n": n,
                    "kappa0": C.kappa0(n, r), "rows": rows}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
