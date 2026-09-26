"""eleg_aligned.py -- the aligned (probability) case: the tight-frame corollary.

If all c_l share a phase -- equivalently sum_l |c_l| = |A|, the "probability cubature rule"
of P1B section 7 -- then M = sum_l c_l m_l m_l^* is Hermitian and positive semi-definite,
so Lemma 1's triangularity forces M = A I exactly (a TIGHT FRAME).  Evaluating that identity
at m_l gives, in one line,

        |c_l| <= |A| / (n+1)      for every atom l,

which is P1B's Proposition A1 conjecture ("max_l x_l <= 1/(n+1) for probability cubature
rules"), open there for n >= 3, and which implies P1A's entropy statement (E0)
(H >= H_inf >= log(n+1)).

Dual form: the same bound comes from the test polynomial

    Lambda_w(z) = 1 + (2/(n+1)) * sum over intervals [s,t] of prod_{j=s..t} z_j conj(w_j),

whose real part is (1/(n+1)) |sum_{k=0..n} prod_{j<=k} z_j conj(w_j)|^2 >= 0 -- the FEJER
kernel along the chain -- with Haar mean 1 and value n+1 at z = w.

Blocks:
  frame   aligned (KL) systems: check M = A I and max_l |c_l| <= |A|/(n+1)
  kernel  check Re Lambda_w >= 0 on a torus sample and Re Lambda_w(w) = n+1
  lp      LP over probability cubature rules on random node sets: max weight vs 1/(n+1)

Run:  uv run --with scipy python eleg_aligned.py
"""

import argparse
import itertools
import json
import os

import numpy as np
from scipy.optimize import linprog

from eleg_chain import chain_vectors, intervals, moment_matrix, subsets

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "FINAL_RUNS", "ELEG")
os.makedirs(OUT, exist_ok=True)


def rotated_block(n, zeta):
    """nodes v_l = (w^{-l} zeta_1, ..., w^{-l} zeta_n), w = exp(2 pi i/(n+1)), weights 1/(n+1).
    A probability cubature rule for every zeta in T^n (checked below)."""
    w = np.exp(2j * np.pi / (n + 1))
    v = np.array([[w ** (-l) * zeta[j] for j in range(n)] for l in range(n + 1)])
    return v, np.full(n + 1, 1.0 / (n + 1), dtype=complex)


def aligned_nodes(n, nblocks, rng, pin_one=False):
    """node set = union of `nblocks` rotated blocks (so a feasible rule always exists)."""
    vs = []
    for b in range(nblocks):
        zeta = (np.ones(n, dtype=complex) if (b == 0 and pin_one)
                else np.exp(2j * np.pi * rng.random(n)))
        vs.append(rotated_block(n, zeta)[0])
    return np.vstack(vs)


def aligned_systems(n, nblocks, rng, trials):
    """probability cubature rules: p >= 0, sum p = 1, all non-constant multi-affine
    moments zero.  LP-extremal rules on node sets built from rotated blocks."""
    out = []
    sets = subsets(n)[1:]
    for _ in range(trials):
        v = aligned_nodes(n, nblocks, rng)
        N = len(v)
        Mc = moment_matrix(v, sets)
        Aeq = np.vstack([np.ones((1, N)), Mc.real, Mc.imag])
        beq = np.concatenate([[1.0], np.zeros(2 * len(sets))])
        cobj = rng.normal(size=N)
        res = linprog(cobj, A_eq=Aeq, b_eq=beq, bounds=[(0, None)] * N, method="highs")
        if res.status == 0 and np.max(np.abs(Aeq @ res.x - beq)) < 1e-9:
            out.append((v, res.x.astype(complex)))
    return out


def block_frame(rng):
    rows = []
    for n in (1, 2, 3, 4, 5):
        for nb in (1, 2, 3, 5):
            sysl = aligned_systems(n, nb, rng, 40)
            if not sysl:
                continue
            worst_frame, worst_w = 0.0, 0.0
            for v, p in sysl:
                m = chain_vectors(v)
                M = np.einsum("l,lk,lj->kj", p, m, m.conj())
                worst_frame = max(worst_frame,
                                  float(np.max(np.abs(M - p.sum() * np.eye(n + 1)))))
                worst_w = max(worst_w, float(np.max(np.abs(p)) * (n + 1)))
            rows.append({"n": n, "nblocks": nb, "N": nb * (n + 1), "systems": len(sysl),
                         "max_dev_from_AI": worst_frame,
                         "max_weight_times_np1": worst_w})
            print(f"[frame] n={n} blocks={nb} N={nb*(n+1):2d} k={len(sysl):3d} | "
                  f"max|M - A I| = {worst_frame:.3e} | "
                  f"(n+1)*max_l p_l = {worst_w:.9f}  (must be <= 1)", flush=True)
    return rows


def fejer_lambda_coeffs(n):
    """Lambda(z) = 1 + (2/(n+1)) sum_{intervals} prod z_j : return {T: coeff}."""
    lam = {(): 1.0}
    for T in intervals(n):
        lam[T] = lam.get(T, 0.0) + 2.0 / (n + 1)
    return lam


def block_kernel(rng):
    rows = []
    for n in (1, 2, 3, 4, 5, 6):
        lam = fejer_lambda_coeffs(n)
        pts = np.exp(2j * np.pi * rng.random((200000, n)))
        val = np.zeros(len(pts), dtype=complex)
        for T, co in lam.items():
            term = np.full(len(pts), co, dtype=complex)
            for j in T:
                term = term * pts[:, j]
            val += term
        re = val.real
        # exact identity: Re Lambda(z) = |sum_k prod_{j<=k} z_j|^2 / (n+1)
        mm = chain_vectors(pts)
        exact = np.abs(mm.sum(axis=1)) ** 2 / (n + 1)
        rows.append({"n": n, "min_Re": float(re.min()),
                     "identity_err": float(np.max(np.abs(re - exact))),
                     "value_at_one": 1.0 + 2.0 / (n + 1) * (n * (n + 1) // 2)})
        print(f"[kernel] n={n}: min Re Lambda = {re.min():+.3e}  "
              f"|Re Lambda - |chain sum|^2/(n+1)| = {np.max(np.abs(re - exact)):.2e}  "
              f"Lambda(1..1) = {rows[-1]['value_at_one']:.1f} (= n+1)", flush=True)
    return rows


def block_lp(rng):
    """direct LP maximisation of one weight over probability cubature rules,
    then a crude search over node sets: reproduces P1B's 0.5 / 0.3333 / 0.25 and extends."""
    rows = []
    for n in (1, 2, 3, 4):
        best = 0.0
        sets = subsets(n)[1:]
        for _ in range(200):
            v = aligned_nodes(n, int(rng.integers(1, 6)), rng, pin_one=True)
            N = len(v)
            Mc = moment_matrix(v, sets)
            Aeq = np.vstack([np.ones((1, N)), Mc.real, Mc.imag])
            beq = np.concatenate([[1.0], np.zeros(2 * len(sets))])
            cobj = np.zeros(N)
            cobj[0] = -1.0
            res = linprog(cobj, A_eq=Aeq, b_eq=beq, bounds=[(0, None)] * N, method="highs")
            if res.status == 0:
                best = max(best, float(res.x[0]))
        rows.append({"n": n, "max_weight": best, "bound": 1.0 / (n + 1)})
        print(f"[lp] n={n}: max weight found = {best:.9f}   bound 1/(n+1) = "
              f"{1/(n+1):.9f}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", nargs="*", default=["kernel", "frame", "lp"])
    a = ap.parse_args()
    rng = np.random.default_rng(4242)
    rep = {}
    if "kernel" in a.blocks:
        rep["kernel"] = block_kernel(rng)
    if "frame" in a.blocks:
        rep["frame"] = block_frame(rng)
    if "lp" in a.blocks:
        rep["lp"] = block_lp(rng)
    with open(os.path.join(OUT, "aligned.json"), "w") as f:
        json.dump(rep, f, indent=1, default=float)
    print("wrote aligned.json", flush=True)


if __name__ == "__main__":
    main()
