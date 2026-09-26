"""gopt4_secord.py -- the junk-relief coefficient beta as an exact eigenvalue.

gopt4_relief.py measures  Cl2_r(eta)/Cl2_r(0) = 1 - beta eta^2 + O(eta^4).  This
script derives beta in closed form as the value of a small quadratic programme
at the (n+1)-block, and checks it against the measured number.

Set-up.  Block: c_l = (1 + eps_l)/(n+1), v_lj = w^{-l} with w = exp(2 pi i/(n+1)),
l = 0..n, and perturb the phases v_lj -> w^{-l} exp(i delta_lj).  Write
eps_l = x_l + i y_l.  Then

  (cost)   sum_l |c_l|^r = (n+1)^{1-r} [ 1 + (1/(n+1)) ( (r(r-1)/2) sum_l x_l^2
                                        + (r/2) sum_l y_l^2 ) + O(|eps|^3) ],
           and delta does not enter the cost AT ALL (phases of v are free);
  (empty)  C_empty = 1  <=>  sum_l eps_l = 0 + O(2);
  (junk)   C_T = (1/(n+1)) sum_l w^{-l|T|} ( eps_l + i sum_{j in T} delta_lj ) + O(2).

Since r < 1 the coefficient r(r-1)/2 is NEGATIVE: spreading the moduli unevenly
is what buys cost, and the phases delta are the free currency used to pay the
junk constraint.  Hence

  beta = (1/(n+1)) * max { (r(1-r)/2) sum x^2 - (r/2) sum y^2 :
                           sum_l eps_l = 0,  sum_{T != empty} |C_T|^2 <= 1 },

a homogeneous quadratic-over-quadratic problem = the least lambda >= 0 with
P - lambda Q <= 0.  (Kernel directions of Q are the junk-free perturbations;
the Key Lemma says P <= 0 there, so the value is finite.)

Plain reading: junk cannot be bought at first order because every junk
coefficient is charged a phase that the cost cannot see, so the leading effect
is a genuine curvature.
"""
import json
import itertools
import argparse
import numpy as np


def subsets(n):
    return [T for k in range(n + 1) for T in itertools.combinations(range(n), k)]


def build(n, r):
    """Return (P, Q) on the real coordinate vector u = (x_0..x_n, y_0..y_n, delta)."""
    nb = n + 1
    Ts = [T for T in subsets(n) if len(T) > 0]
    nd = nb * n
    dim = 2 * nb + nd
    w = np.exp(2j * np.pi / nb)

    # linear map u -> (C_T)_{T != empty},  complex, as a (len(Ts), dim) matrix
    L = np.zeros((len(Ts), dim), dtype=complex)
    for t, T in enumerate(Ts):
        m = len(T)
        for l in range(nb):
            ph = w ** (-l * m)
            L[t, l] += ph / nb                       # x_l
            L[t, nb + l] += 1j * ph / nb             # y_l
            for j in T:
                L[t, 2 * nb + l * n + j] += 1j * ph / nb   # delta_lj

    # eliminate the C_empty constraint sum_l eps_l = 0 by restricting to its kernel
    E = np.zeros((2, dim))
    E[0, :nb] = 1.0
    E[1, nb:2 * nb] = 1.0
    _, s, vt = np.linalg.svd(E)
    Z = vt[int(np.sum(s > 1e-12)):].T                # (dim, dim-2) orthonormal basis

    P = np.zeros((dim, dim))
    for l in range(nb):
        P[l, l] = r * (1 - r) / 2.0
        P[nb + l, nb + l] = -r / 2.0
    LZ = L @ Z
    Q = (LZ.conj().T @ LZ).real                      # |C|^2 as a real quadratic form
    Pz = Z.T @ P @ Z
    return Pz, Q


def value(P, Q, lo=0.0, hi=1e4, iters=200):
    """least lambda >= 0 with P - lambda Q <= 0  (bisection on max eigenvalue)."""
    def ok(lam):
        return np.linalg.eigvalsh(P - lam * Q).max() <= 1e-11
    if ok(lo):
        return 0.0
    while not ok(hi):
        hi *= 2
        if hi > 1e12:
            return np.inf
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if ok(mid):
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=str, default='FINAL_RUNS/GOPT4/secord.json')
    args = ap.parse_args()
    measured = {(2, 2.0): 0.076946, (2, 2.5): 0.047181,
                (3, 2.0): 0.071429, (3, 3.0): 0.050000}
    out = []
    print(f"{'n':>2} {'q':>5} {'r':>8}   {'beta (eigen)':>16} "
          f"{'beta (measured)':>16}  ratio")
    for (n, q) in [(1, 2.0), (2, 2.0), (2, 2.5), (2, 3.0),
                   (3, 2.0), (3, 3.0), (3, 4.0), (4, 2.0)]:
        r = q / (n + 1)
        P, Q = build(n, r)
        lam = value(P, Q)
        beta = lam / (n + 1)
        mm = measured.get((n, q))
        print(f"{n:>2} {q:>5} {r:>8.5f}   {beta:>16.9f} "
              f"{(f'{mm:.6f}' if mm else '-'):>16}"
              f"  {(beta/mm if mm else float('nan')):.6f}")
        out.append({'n': n, 'q': q, 'r': r, 'beta_eigen': beta,
                    'beta_measured': mm})
    with open(args.out, 'w') as fh:
        json.dump(out, fh, indent=1, default=float)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
