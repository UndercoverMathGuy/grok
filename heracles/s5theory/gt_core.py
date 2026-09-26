"""gt_core.py -- shared machinery for the finite-group feature-count theory (heracles/s5theory, 2026-09-23).

Objects (see S5_THEORY.md section 0 for the symbols):
  * groups from ../groups.py (S5, S4, A5, D_n, abelian), class data, Frobenius-Schur indicators;
  * explicit REAL ORTHOGONAL models of the real-type irreps (right-regular commutant trick);
  * the programme  Gamma(c) = max { min_{C != e} sum_pi y_pi cov_pi(C) : sum_pi c_pi y_pi^r <= 1 },
    solved EXACTLY by enumerating the vertices of P = { y >= 0 : sum_pi y_pi cov_pi(C) >= 1 for all C != e }
    (a concave, coordinatewise non-decreasing cost attains its minimum over P at a vertex; Lemma V in the write-up).
Everything in float64 unless stated; vertex enumeration in exact rationals where the character table is integral.
"""
import itertools, sys
from fractions import Fraction
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import groups as groups_mod  # noqa: E402


def get(name):
    if name.startswith("d") and name[1:].isdigit():
        n = int(name[1:])
        if name not in groups_mod._CACHE:
            groups_mod._CACHE[name] = groups_mod.D(n)
        return groups_mod._CACHE[name]
    return groups_mod.get_group(name)


# ----------------------------------------------------------------------------------------------- class data
def class_data(G):
    """sizes (C,), inverse-class map inv_cls (C,), identity class index."""
    C = len(G.class_names)
    sizes = np.bincount(G.class_of, minlength=C)
    rep = [int(np.flatnonzero(G.class_of == c)[0]) for c in range(C)]
    inv_cls = np.array([G.class_of[G.INV[x]] for x in rep])
    return sizes, inv_cls, int(G.class_of[G.e])


def fs_indicator(G, r):
    """Frobenius-Schur indicator of the COMPLEX irreducible constituent of real irrep r:
    groups.py stores real irreps; for a real-type irrep CHAR is the complex character, for a complex-type
    (schur 2) irrep CHAR = chi + conj(chi) and we return 0."""
    if G.schur[r] == 2:
        return 0
    sq = G.MULT[np.arange(G.N), np.arange(G.N)]          # x^2
    return float(np.mean(G.CHAR[r][G.class_of[sq]]))


# ----------------------------------------------------------------------------------------------- real irreps
def real_irrep(G, r, seed=0):
    """explicit real orthogonal model rho: (N, d, d) of real-type irrep r (schur 1).
    The isotypic component of R^G under the LEFT regular action is d copies of rho; a generic symmetric element of
    the RIGHT group algebra commutes with the left action, acts on the multiplicity space, and its eigenspaces are
    single copies. rho(g) = U^T L_g U with U an orthonormal basis of one eigenspace."""
    assert G.schur[r] == 1
    d = int(G.irrep_dims[r]); N = G.N
    P = G.PROJ[r]
    rng = np.random.default_rng(seed)
    # right translation R_h f(x) = f(x h): matrix R_h[x, y] = 1 if y = x h
    coef = rng.normal(size=N)
    M = np.zeros((N, N))
    for h in range(N):
        M[np.arange(N), G.MULT[np.arange(N), h]] += coef[h]
    M = M + M.T
    Mi = P @ M @ P
    w, V = np.linalg.eigh(Mi)
    # isotypic subspace has dim d^2; pick eigenvalues belonging to it (eigvecs inside range of P)
    inside = np.array([np.linalg.norm(P @ V[:, i]) > 0.999 for i in range(N)])
    wi, Vi = w[inside], V[:, inside]
    assert len(wi) == d * d, (len(wi), d)
    order = np.argsort(wi); wi, Vi = wi[order], Vi[:, order]
    # group into clusters of size d
    U = Vi[:, :d]
    assert np.ptp(wi[:d]) < 1e-8 and (d * d == d or wi[d] - wi[d - 1] > 1e-6), wi
    # left translation L_g f(x) = f(g^{-1} x): L_g[x, y] = 1 if y = g^{-1} x
    rho = np.zeros((N, d, d))
    for g in range(N):
        gi = G.INV[g]
        idx = G.MULT[gi, np.arange(N)]          # y = g^{-1} x
        Lu = U[idx, :]                           # (L_g U)[x] = U[g^{-1} x]
        rho[g] = U.T @ Lu
    # checks: homomorphism, orthogonal, character
    for _ in range(20):
        a, b = rng.integers(N, size=2)
        assert np.allclose(rho[G.MULT[a, b]], rho[a] @ rho[b], atol=1e-9)
    assert np.allclose(np.einsum("gij,gkj->gik", rho, rho), np.eye(d)[None], atol=1e-9)
    assert np.allclose(np.trace(rho, axis1=1, axis2=2), G.CHAR[r][G.class_of], atol=1e-8)
    return rho


# ----------------------------------------------------------------------------------------------- programme data
def programme_data(G):
    """list of nontrivial real irreps with (name, d, eps, cov over non-identity classes).
    d = COMPLEX dimension of a constituent, eps = 1 (self-dual) or 2 (conjugate pair).
    cov_pi(C) = d - Re chi(C) for one complex constituent chi (for a complex pair groups.py stores chi + conj chi)."""
    sizes, inv_cls, e = class_data(G)
    classes = [c for c in range(len(G.class_names)) if c != e]
    out = []
    for r, nm in enumerate(G.irrep_names):
        if nm == "trivial":
            continue
        if G.schur[r] == 2:
            d, eps, rechi = int(G.irrep_dims[r]) // 2, 2, G.CHAR[r] / 2.0
        else:
            d, eps, rechi = int(G.irrep_dims[r]), 1, G.CHAR[r]
        cov = np.array([d - rechi[c] for c in classes], dtype=float)
        out.append(dict(r=r, name=nm, d=d, eps=eps, cov=cov))
    return out, classes


def prices(irr, nu, q, kappa):
    """c_pi = kappa_pi * eps^{q/2 - r} * d^{q/2}; kappa: callable (irrep dict) -> kappa value."""
    r = q / nu
    return np.array([kappa(p) * p["eps"] ** (q / 2 - r) * p["d"] ** (q / 2) for p in irr])


# ----------------------------------------------------------------------------------------------- vertex enumeration
def _solve_frac(A, b):
    """exact Gaussian elimination over Fractions; returns x or None if singular."""
    n = len(A); M = [list(map(Fraction, A[i])) + [Fraction(b[i])] for i in range(n)]
    for c in range(n):
        piv = next((i for i in range(c, n) if M[i][c] != 0), None)
        if piv is None:
            return None
        M[c], M[piv] = M[piv], M[c]
        for i in range(n):
            if i != c and M[i][c] != 0:
                f = M[i][c] / M[c][c]
                M[i] = [x - f * y for x, y in zip(M[i], M[c])]
    return [M[i][n] / M[i][i] for i in range(n)]


def vertices(cov, exact=True, tol=1e-10):
    """vertices of P = {y >= 0, cov^T y >= 1}. cov: (R, m) array (irreps x non-identity classes).
    exact=True requires integer (or rational) cov; returns list of tuples (Fractions or floats)."""
    R, m = cov.shape
    rows = [("cov", j) for j in range(m)] + [("nn", i) for i in range(R)]
    verts = set()
    for act in itertools.combinations(range(len(rows)), R):
        A, b = [], []
        for k in act:
            kind, j = rows[k]
            if kind == "cov":
                A.append([cov[i, j] for i in range(R)]); b.append(1)
            else:
                A.append([1 if i == j else 0 for i in range(R)]); b.append(0)
        if exact:
            A = [[Fraction(x).limit_denominator(10 ** 9) if not isinstance(x, Fraction) else x for x in row] for row in A]
            x = _solve_frac(A, b)
            if x is None:
                continue
            if any(v < 0 for v in x):
                continue
            if any(sum(cov[i, j] * x[i] for i in range(R)) < 1 for j in range(m)):
                continue
            verts.add(tuple(x))
        else:
            A = np.array(A, float); b = np.array(b, float)
            if abs(np.linalg.det(A)) < 1e-12:
                continue
            x = np.linalg.solve(A, b)
            if (x < -tol).any() or (cov.T @ x < 1 - tol).any():
                continue
            verts.add(tuple(np.round(np.clip(x, 0, None), 12)))
    return sorted(verts)


def solve(verts, c, r):
    """Gamma = (min_v sum c v^r)^{-1/r}; returns Gamma, argmin vertex index, all costs."""
    V = np.array([[float(x) for x in v] for v in verts])
    costs = (c[None, :] * np.where(V > 0, V, 0) ** r).sum(1)
    k = int(costs.argmin())
    return costs[k] ** (-1 / r), k, costs


def support_names(irr, v):
    return [irr[i]["name"] for i, x in enumerate(v) if float(x) > 0]
