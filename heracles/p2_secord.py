"""P2: the sharp constant c_n of the Quadratic Product Lemma, as an eigenvalue.

At the (n+1)-block, c_l = (1 + eps_l)/(n+1), eps_l = x_l + i y_l, and
v_lj = w^{-l} e^{i delta_lj}, w = exp(2 pi i/(n+1)).  To second order

    -log[ (n+1)^{n+1} prod_l |c_l| ]  =  (1/2)( sum_l x_l^2 - sum_l y_l^2 ) + O(3)
    C_empty = 1                       <=>  sum_l eps_l = 0
    C_T (T != empty)                  =  (1/(n+1)) sum_l w^{-l|T|}( eps_l + i sum_{j in T} delta_lj )

so the largest junk-normalised product deficit is

    c_n := max { (1/2)(sum x^2 - sum y^2) : sum_l eps_l = 0, sum_{T != empty}|C_T|^2 <= 1 } .

CLAIM (hand-derived exactly at n = 2, giving 9/8):  c_n = (n+1)^2/(4n).
This script computes c_n as a generalised eigenvalue and checks the closed form.
"""
import json, os
import numpy as np
import p2_core as P


def build(n):
    """Real quadratic forms Pm (objective) and Qm (junk energy) in z = (x, y, delta)."""
    nb = n + 1
    M = P.masks(n)
    w = np.exp(2j * np.pi / nb)
    dim = 2 * nb + nb * n

    def cT(z):
        x, y = z[:nb], z[nb:2 * nb]
        d = z[2 * nb:].reshape(nb, n)
        eps = x + 1j * y
        out = []
        for t in range(1, M.shape[0]):
            tt = int(M[t].sum())
            dl = d[:, M[t]].sum(axis=1)
            out.append(np.sum(w ** (-np.arange(nb) * tt) * (eps + 1j * dl)) / nb)
        return np.array(out)

    # constraint sum eps = 0 -> project onto its null space
    A = np.zeros((2, dim))
    A[0, :nb] = 1.0
    A[1, nb:2 * nb] = 1.0
    _, _, Vt = np.linalg.svd(A)
    B = Vt[2:].T                       # dim x (dim-2) basis of the constraint subspace

    k = B.shape[1]
    Pm = np.zeros((k, k))
    Qm = np.zeros((k, k))
    for i in range(k):
        for j in range(i, k):
            zi, zj = B[:, i], B[:, j]
            pv = 0.5 * (np.dot(zi[:nb], zj[:nb]) - np.dot(zi[nb:2 * nb], zj[nb:2 * nb]))
            qv = float(np.real(np.vdot(cT(zi), cT(zj))))
            Pm[i, j] = Pm[j, i] = pv
            Qm[i, j] = Qm[j, i] = qv
    return Pm, Qm


def cn(n, tol=1e-12):
    """c_n = max{P(z) : Q(z) <= 1} = min{lam >= 0 : lam Q - P is PSD}.

    Q is singular (its kernel = the junk-free perturbations, on which P <= 0 by the Key
    Lemma), so one may NOT restrict to range(Q): kernel/range cross terms matter.
    Bisection on the smallest eigenvalue of lam Q - P.
    """
    Pm, Qm = build(n)
    lo, hi = 0.0, 1.0
    while np.linalg.eigvalsh(hi * Qm - Pm).min() < -tol:
        hi *= 2
        if hi > 1e8:
            return np.inf
    while hi - lo > 1e-11 * max(1.0, hi):
        mid = 0.5 * (lo + hi)
        if np.linalg.eigvalsh(mid * Qm - Pm).min() < -tol:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


if __name__ == "__main__":
    res = {}
    print(f"{'n':>3} {'c_n measured':>16} {'(n+1)^2/(4n)':>16} {'ratio':>12}")
    for n in range(1, 7):
        v = cn(n)
        pred = (n + 1) ** 2 / (4.0 * n)
        res[str(n)] = {"c_n": v, "pred": pred, "ratio": v / pred}
        print(f"{n:>3} {v:>16.9f} {pred:>16.9f} {v/pred:>12.9f}")
    os.makedirs("FINAL_RUNS/P2", exist_ok=True)
    json.dump(res, open("FINAL_RUNS/P2/secord.json", "w"), indent=1)
    print("wrote FINAL_RUNS/P2/secord.json")
