r"""Check of Theorem KM2 (tight case of Paper Lemma 7 is the diagonal Fourier configuration).
1) Random-start solutions of sum_l prod_{t in T} xi_lt = 0 (all nonempty T) with nu=n+1 points:
   test diagonal structure xi_lt / xi_{0t} = zeta_l independent of t, and {zeta_l} = nu-th roots of unity.
2) Test the key identity of the proof: for every l0 and bijection sigma:[n]->[nu]\{l0},
   prod_t (1 - xi_{l0 t}/xi_{sigma(t) t}) = nu.
3) Complex Hadamard counter-check at nu=4: the Craigen/Haagerup family F4(a) gives orthogonal chain vectors
   but violates the non-interval condition T={1,3} unless a is a 4th-root case."""
import numpy as np, itertools
from scipy.optimize import least_squares
rng = np.random.default_rng(1)
def subsets(n):
    for k in range(1, n+1):
        for T in itertools.combinations(range(n), k): yield T
def resid(phi, n):
    th = phi.reshape(n+1, n); out = []
    for T in subsets(n):
        z = np.exp(1j*th[:, list(T)].sum(axis=1)).sum(); out += [z.real, z.imag]
    return np.array(out)
for n in [2, 3, 4, 5]:
    nu = n+1; worst_diag = 0; worst_root = 0; worst_id = 0; cnt = 0
    for trial in range(60):
        r = least_squares(resid, rng.uniform(0, 2*np.pi, nu*n), args=(n,), xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=4000)
        if np.max(np.abs(r.fun)) > 1e-10: continue
        cnt += 1
        xi = np.exp(1j*r.x.reshape(nu, n))
        ratio = xi / xi[0:1, :]
        worst_diag = max(worst_diag, np.max(np.abs(ratio - ratio[:, 0:1])))
        zeta = ratio[:, 0]
        worst_root = max(worst_root, np.max(np.abs(zeta**nu - 1)))
        for l0 in range(nu):
            others = [l for l in range(nu) if l != l0]
            for sig in itertools.permutations(others):
                val = np.prod([1 - xi[l0, t]/xi[sig[t], t] for t in range(n)])
                worst_id = max(worst_id, abs(val - nu))
    print(f"n={n}: {cnt} solutions; max |xi_lt/xi_0t - xi_l1/xi_01| = {worst_diag:.1e}; max |zeta^nu-1| = {worst_root:.1e}; max |prod(1-xi_l0/xi_sigma)-nu| = {worst_id:.1e}")
# 3) Craigen family, nu=4: rows m_l = chain vectors (1, xi_l1, xi_l1 xi_l2, xi_l1 xi_l2 xi_l3)
for a in [0.0, 0.3, 1.0, np.pi/2]:
    e = np.exp(1j*a)
    M = np.array([[1, 1, 1, 1], [1, 1j*e, -1, -1j*e], [1, -1, 1, -1], [1, -1j*e, -1, 1j*e]])
    orth = np.max(np.abs(M.conj() @ M.T - 4*np.eye(4)))
    xi = M[:, 1:] / M[:, :-1]
    viol = max(abs(xi[:, list(T)].prod(axis=1).sum()) for T in subsets(3))
    viol13 = abs((xi[:, 0]*xi[:, 2]).sum())
    print(f"Craigen F4(a={a:.3f}): chain orthogonality defect {orth:.1e}; max JF violation {viol:.3f} (T={{1,3}}: {viol13:.3f})")
