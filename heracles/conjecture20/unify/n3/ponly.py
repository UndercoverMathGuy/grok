"""P-only spectral bounds for 4x4 correlation matrices: how lossy at the n=3 extremals?

For a correlation matrix Hhat (unit diagonal, 4x4) with P = sum_{j<k} |H_jk|^2, the spectrum nu has
sum nu = 4, sum (nu-1)^2 = 2P.  Worst case for sum nu^r (r<1) over such spectra is computed by brute force
over a fine parametrisation (exact for this 2-constraint problem: the extremal spectra have <= 2 distinct
non-zero values plus zeros; we simply scan).
"""
import numpy as np
from chain_tight import three_atom, two_atom, chain_vectors, Mr
from n3lib import coeffs


def worst_Mr_given_P(P, r, grid=4001):
    # spectra of the form (z zeros, then values): scan families (a, b, b, b), (0, c, e, e), (0, 0, f, g)
    best = np.inf
    cands = []
    for k_zero in range(0, 3):
        m = 4 - k_zero  # number of non-zero eigenvalues
        S2 = 2 * P - k_zero  # sum over nonzero of (nu-1)^2
        if S2 < -1e-12:
            continue
        # nonzero part: one value a, (m-1) values b ; a + (m-1) b = 4 ; (a-1)^2 + (m-1)(b-1)^2 = S2
        for a in np.linspace(0, 4, grid):
            b = (4 - a) / (m - 1) if m > 1 else None
            if m == 1:
                if abs((a - 1) ** 2 - S2) < 1e-3 and abs(a - 4) < 1e-9:
                    cands.append([a])
                continue
            if b < 0:
                continue
            val = (a - 1) ** 2 + (m - 1) * (b - 1) ** 2
            cands.append(([a] + [b] * (m - 1), val - S2))
    # choose root crossings
    out = []
    for k in range(len(cands) - 1):
        c0, c1 = cands[k], cands[k + 1]
        if isinstance(c0, tuple) and isinstance(c1, tuple) and len(c0[0]) == len(c1[0]) and c0[1] * c1[1] <= 0:
            nu = np.array(c0[0])
            out.append(np.mean(np.concatenate([nu ** r, np.zeros(4 - len(nu))])) ** (1 / r))
    return min(out) if out else np.nan


if __name__ == '__main__':
    s = 2.0; r = 1 / s
    for label, (beta, ang) in [("3-atom", three_atom(0.37174, 0.3266, 0.2773, 1.432)),
                               ("2-atom", two_atom(0.3158))]:
        A, bT = coeffs(beta, ang)
        rho = np.abs(beta); L = rho.sum()
        for e in [(0, 1, 1, 2), (0, 0, 1, 1)]:
            rho, U = chain_vectors(beta, ang, e, (0, 1, 2))
            H = (U.T * rho) @ U.conj() / L
            nu = np.linalg.eigvalsh(H)
            P = (np.sum(np.abs(H) ** 2) - 4) / 2
            print(label, e, "spectrum", np.round(nu, 4), "P=%.4f" % P, "actual M_r=%.5f" % Mr(nu, r),
                  "P-only worst M_r=%.5f" % worst_Mr_given_P(P, r), " a=%.5f" % (abs(A) / L))
