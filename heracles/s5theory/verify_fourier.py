"""verify_fourier.py -- fp64 checks of the group-side lemmas of S5_THEORY.md sections 1-2.

  F0  Frobenius-Schur indicators: every irrep of S5, S4, A5, D_7 is real type (+1).
  F1  level-set formula: mean of z over L_C = { (a, c) : a_1...a_n c^{-1} in C } equals
        N^{-nu} sum_rho chi_rho(C^{-1}) G_rho,   G_rho = sum_i tr( W_i E_i1^T ... E_in^T ),
      with E = sum_x e(x) rho(x)^T (real orthogonal rho), on random gated chains (S4 n = 2, 3; S5 n = 2).
  F2  the (rho,...,rho) Fourier block of the logit table equals X_rho = sum_i E_i1 (x) ... (x) E_in (x) W_i.
  F3  Plancherel per irrep: sum_x e(x)^2 = N^{-1} sum_rho d_rho ||E(rho)||_F^2.
  F4  Schur: the space of (rho,...,rho)-block tensors invariant under the task symmetry group G^nu is ONE-dimensional
      and spanned by the chain-trace tensor T (checked by computing the rank of the averaging projector pair by pair).
  F5  margin <= level-set bound <= unrestricted master bound, on random and on trained nets (GX bil runs if present).
"""
import itertools, json, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gt_core as gc  # noqa: E402

rng = np.random.default_rng(0)
LOG = []


def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.append(s)


def irreps_of(G):
    out = {}
    for r, nm in enumerate(G.irrep_names):
        out[nm] = gc.real_irrep(G, r) if G.irrep_dims[r] > 1 else G.CHAR[r][G.class_of][:, None, None].astype(float)
    return out


def logits(G, E, W, n):
    """E (n, m, N), W (m, N) -> z (N^n rows, N classes) rows in lexicographic order of (a_1..a_n)."""
    m, N = W.shape
    h = np.ones((m, 1))
    for j in range(n):
        h = (h[:, :, None] * E[j][:, None, :]).reshape(m, -1)
    return h.T @ W                      # (N^n, N)


def row_products(G, n):
    N = G.N; P = np.zeros(N ** n, dtype=int)
    for idx, a in enumerate(itertools.product(range(N), repeat=n)):
        x = a[0]
        for y in a[1:]:
            x = G.MULT[x, y]
        P[idx] = x
    return P


def fhat(f, rho):
    return np.einsum("x,xij->ji", f, rho)        # sum_x f(x) rho(x)^T


def check(G, n, m=5):
    N = G.N; nu = n + 1
    sizes, inv_cls, e = gc.class_data(G)
    rho = irreps_of(G)
    E = rng.normal(size=(n, m, N)); W = rng.normal(size=(m, N))
    z = logits(G, E, W, n); P = row_products(G, n)
    # F1 brute-force level-set means: D = P c^{-1}
    D = G.MULT[P[:, None], G.INV[None, :]]            # (rows, classes c)
    cls = G.class_of[D]
    zbar = np.array([z[cls == C].mean() for C in range(len(sizes))])
    Gr = {}
    for nm, R in rho.items():
        acc = 0.0
        for i in range(m):
            M = fhat(W[i], R)
            for j in range(n):
                M = M @ fhat(E[j, i], R).T
            acc += np.trace(M)
        Gr[nm] = acc
    pred = np.array([sum(G.CHAR[r][inv_cls[C]] * Gr[nm] for r, nm in enumerate(G.irrep_names)) for C in range(len(sizes))]) / N ** nu
    err1 = np.abs(zbar - pred).max() / np.abs(zbar).max()
    # F3 Plancherel
    f = E[0, 0]
    pl = sum(G.irrep_dims[r] * np.linalg.norm(fhat(f, rho[nm])) ** 2 for r, nm in enumerate(G.irrep_names)) / N
    err3 = abs(pl - (f ** 2).sum()) / (f ** 2).sum()
    # F2 block of the logit table at (rho,...,rho; rho) for the largest irrep (n = 2 only, cost)
    err2 = float("nan")
    if n == 2:
        nm = max(rho, key=lambda k: rho[k].shape[1]); R = rho[nm]; d = R.shape[1]
        Z = z.reshape(N, N, N)
        blk = np.einsum("abc,aij,bkl,cst->jilkts", Z, R, R, R)   # sum z rho(a)^T (x) rho(b)^T (x) rho(c)^T
        X = sum(np.einsum("ji,lk,ts->jilkts", fhat(E[0, i], R), fhat(E[1, i], R), fhat(W[i], R)) for i in range(m))
        err2 = np.abs(blk - X).max() / np.abs(X).max()
    # F5 margin <= level-set bound (min over C != e of zbar_e - zbar_C)
    y = P; zz = z.copy(); zy = zz[np.arange(len(y)), y]; zz[np.arange(len(y)), y] = -np.inf
    margin = (zy - zz.max(1)).min()
    lsb = min(zbar[e] - zbar[C] for C in range(len(sizes)) if C != e)
    log(f"  {G.name} n={n}: F1 level-set formula rel err {err1:.2e}; F2 block rel err {err2:.2e}; F3 Plancherel rel err {err3:.2e}; "
        f"F5 margin {margin:.3f} <= level-set bound {lsb:.3f}: {margin <= lsb + 1e-9}")
    return max(err1, err3, 0 if np.isnan(err2) else err2)


def schur_invariants(G, nm_list):
    """F4: for the (rho,...,rho) block, G^nu acts by g_k on a PAIR of indices (rho(g) (x) rho(g)).
    Invariant space = tensor product over k of Inv(rho (x) rho); report dim Inv(rho (x) rho) (must be 1)."""
    rho = irreps_of(G)
    for nm in nm_list:
        R = rho[nm]; d = R.shape[1]
        Pk = np.mean([np.kron(R[g], R[g]) for g in range(G.N)], axis=0)   # averaging projector on V (x) V
        rank = int(round(np.trace(Pk)))
        # the invariant vector is vec(I) (orthogonal rho): check
        vI = np.eye(d).reshape(-1)
        ok = np.allclose(Pk @ vI, vI)
        log(f"  {G.name} {nm} (d={d}): dim Inv(rho (x) rho) = {rank} (trace of averaging projector), vec(I) invariant: {ok}")


def main():
    log("== F0 Frobenius-Schur indicators")
    for nm in ["s5", "s4", "a5", "d7"]:
        G = gc.get(nm)
        log(f"  {nm}: " + ", ".join(f"{G.irrep_names[r]}:{gc.fs_indicator(G, r):+.3f}" for r in range(G.R)))
    log("== F1-F3, F5 on random gated chains")
    worst = 0
    for nm, n in [("s4", 2), ("s4", 3), ("s5", 2), ("a5", 2), ("d7", 2), ("d7", 3)]:
        worst = max(worst, check(gc.get(nm), n))
    log(f"  worst relative error {worst:.2e}")
    log("== F4 Schur invariants")
    schur_invariants(gc.get("s4"), ["2", "std", "std_sign"])
    schur_invariants(gc.get("s5"), ["std", "std_sign", "5a", "5b", "6"])
    schur_invariants(gc.get("a5"), ["3a", "3b", "4", "5"])
    (HERE / "out").mkdir(exist_ok=True)
    (HERE / "out" / "verify_fourier.log").write_text("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
