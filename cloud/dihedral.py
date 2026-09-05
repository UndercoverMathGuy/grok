"""Dihedral group D_n machinery (n odd), same interface as groups.py (S5).

Elements: index i in [0, 2n): i < n is the rotation r^i, i >= n is the
reflection s r^(i-n). Product rule: r^a r^b = r^(a+b); r^a (s r^b) = s r^(b-a);
(s r^a) r^b = s r^(a+b); (s r^a)(s r^b) = r^(b-a).
Classes (n odd): {e}, {r^k, r^-k} for k = 1..(n-1)/2, {all reflections}.
Irreps: trivial, sign (s -> -1), and (n-1)/2 two-dim irreps rho_j with
chi_j(r^k) = 2 cos(2 pi j k / n), chi_j(s r^k) = 0.
Coverage of class g by irrep rho = 1 - chi_rho(g)/d_rho (the S5 note's
quantity): rho_j covers rotation k by 1 - cos(2 pi j k/n) (EXACTLY the Z_n
single-frequency coverage) and every reflection by 1; sign covers rotations
by 0 and reflections by 2.
"""
import sys
import numpy as np


def build(n):
    assert n % 2 == 1
    N = 2 * n
    idx = np.arange(N)
    rot = idx < n
    k = np.where(rot, idx, idx - n)
    MULT = np.zeros((N, N), dtype=np.int64)
    for i in range(N):
        for j in range(N):
            a, b = int(k[i]), int(k[j])
            if rot[i] and rot[j]:
                MULT[i, j] = (a + b) % n
            elif rot[i] and not rot[j]:
                MULT[i, j] = n + (b - a) % n
            elif not rot[i] and rot[j]:
                MULT[i, j] = n + (a + b) % n
            else:
                MULT[i, j] = (b - a) % n
    INV = np.array([int(np.flatnonzero(MULT[i] == 0)[0]) for i in range(N)])
    # classes: 0 = e, 1..(n-1)/2 = rotation pairs, (n+1)/2 = reflections
    h = (n - 1) // 2
    CLASS_OF = np.where(rot, np.minimum(k, (n - k) % n), h + 1)
    nC = h + 2
    names = ["trivial", "sign"] + [f"rho{j}" for j in range(1, h + 1)]
    dims = np.array([1, 1] + [2] * h)
    CHAR = np.zeros((len(names), nC))
    CHAR[0] = 1
    CHAR[1, :h + 1] = 1; CHAR[1, h + 1] = -1
    for j in range(1, h + 1):
        CHAR[1 + j, 0] = 2
        for c in range(1, h + 1):
            CHAR[1 + j, c] = 2 * np.cos(2 * np.pi * j * c / n)
        CHAR[1 + j, h + 1] = 0
    xyinv = MULT[np.arange(N)[:, None], INV[None, :]]
    PROJ = np.stack([(dims[r] / N) * CHAR[r][CLASS_OF[xyinv]] for r in range(len(names))])
    assert np.allclose(PROJ.sum(0), np.eye(N), atol=1e-9)
    for r in range(len(names)):
        assert np.allclose(PROJ[r] @ PROJ[r], PROJ[r], atol=1e-9)
    class_sizes = np.bincount(CLASS_OF, minlength=nC)
    return dict(N=N, MULT=MULT, INV=INV, CLASS_OF=CLASS_OF, IRREP_NAMES=names,
                IRREP_DIMS=dims, CHAR=CHAR, PROJ=PROJ, CLASS_SIZES=class_sizes)


def coverage(G):
    """cov[r, c] = 1 - chi_r(c)/d_r for nontrivial classes c >= 1."""
    return 1 - G["CHAR"][:, 1:] / G["IRREP_DIMS"][:, None]


def prediction(n, nu=6, Kmax=12):
    """Closed form. All two-dim irreps share d = 2 and hence the same cost, so
    the max-min program over supports of rho's is EXACTLY the Z_n program:
    rotation class k is covered by sum_{j in S} (1 - cos(2 pi j k / n)) = the
    Z_n coverage of offset k, and every reflection is covered by |S| (each
    rho_j gives 1). The sign irrep covers only reflections (by 2). Hence
      K*(D_n) = argmax_K M_K(n) / K^(nu/2)  (= K*(Z_n) at the same nu),
      sign amplitude s* = 0 as long as the binding class is a rotation, i.e.
      as long as K > M_K(n) (reflection coverage K exceeds rotation coverage
      M_K), which holds for every K at every n (M_K = K - c, c > 0).
    Returns the ladder r(K) and the reflection/rotation coverage ratio."""
    import sys as _s; _s.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "nudial"))
    from kstar_table import best_subset
    M = {K: best_subset(n, K)[0] for K in range(1, Kmax + 1)}
    g = {K: M[K] / K ** (nu / 2) for K in M}
    gm = max(g.values())
    return {K: dict(M=M[K], r=g[K] / gm, refl_over_rot=K / M[K] if M[K] > 0 else float("inf")) for K in M}


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 113
    G = build(n)
    cov = coverage(G)
    print(f"D_{n}: |G|={G['N']}, {len(G['IRREP_NAMES'])} irreps; class sizes {G['CLASS_SIZES'].tolist()[:3]}...{G['CLASS_SIZES'].tolist()[-1]}")
    print("worst-class coverage: sign", float(cov[1].min()), " rho1", round(float(cov[2].min()), 5), " (Z_n M_1 =", round(1 - np.cos(2 * np.pi / n), 5), ")")
    for nu in (6, 4):
        lad = prediction(n, nu=nu, Kmax=8)
        ks = max(lad, key=lambda K: lad[K]["r"])
        print(f"nu={nu}: K*={ks}; ladder r(K) " + " ".join(f"{K}:{v['r']:.3f}" for K, v in lad.items())
              + "; reflections/rotations coverage at K*: %.1f" % lad[ks]["refl_over_rot"])
