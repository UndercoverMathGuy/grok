"""Referee check of KM2 (equality case of Paper Lemma 7).

The agent's scripts ASSUME beta_l = beta/nu (equal phases) and then solve for the xi.  Step (i) of the proof (equal
phases) is therefore untested numerically.  Here:
(A) Unknowns: unit d_l = nu*beta_l/beta (free phases) and unit xi_lt, nu = n+1 terms.  Solve (JF):
      sum_l d_l prod_{t in T} xi_lt = nu [T = empty]   for all T subset [n]
    by least squares from random starts.  Any nu-term solution is automatically an equality case of Lemma 7
    (sum |beta_l|^r = nu^{1-r}|beta|^r).  Test: d_l == 1 and xi_lt = c_t * omega^{pi(l)} (same root in every slot).
(B) Over-complete search: N = nu+1 terms with |d_l| free (not unit), minimise sum |d_l|^r subject to JF.  The minimum must
    be nu (normalised), approached only by nu-term Fourier configurations.
(C) Craigen F4(a): closed form of the (JF) violation at T = {1,3}: |2 - 2 e^{2ia}| = 4|sin a| (zero iff a = 0 mod pi).
(D) Ingredient checks: normal + upper-triangular => diagonal (random unitary-diagonal test), and Newton step.
"""
import itertools
import numpy as np
from scipy.optimize import least_squares, minimize
rng = np.random.default_rng(777)


def subsets(n, include_empty=True):
    for k in range(0 if include_empty else 1, n + 1):
        for T in itertools.combinations(range(n), k):
            yield T


def resid_A(x, n):
    nu = n + 1
    ph_d = x[:nu]; th = x[nu:].reshape(nu, n)
    d = np.exp(1j * ph_d)
    out = []
    for T in subsets(n):
        z = np.sum(d * np.exp(1j * th[:, list(T)].sum(axis=1))) - (nu if len(T) == 0 else 0)
        out += [z.real, z.imag]
    return np.array(out)


print("(A) nu-term solutions of (JF) with FREE unit phases d_l")
for n in [1, 2, 3, 4, 5]:
    nu = n + 1
    ntr = 200 if n <= 3 else (100 if n == 4 else 50)
    sols = 0; worst_d = 0; worst_diag = 0; worst_root = 0
    for _ in range(ntr):
        x0 = rng.uniform(0, 2 * np.pi, nu + nu * n)
        r = least_squares(resid_A, x0, args=(n,), xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=6000)
        if np.max(np.abs(r.fun)) > 1e-10:
            continue
        sols += 1
        d = np.exp(1j * r.x[:nu]); xi = np.exp(1j * r.x[nu:].reshape(nu, n))
        worst_d = max(worst_d, np.max(np.abs(d - 1)))
        ratio = xi / xi[0:1, :]
        worst_diag = max(worst_diag, np.max(np.abs(ratio - ratio[:, 0:1])))
        zeta = ratio[:, 0]
        # zeta must be all nu-th roots of unity, each once
        k = np.angle(zeta) * nu / (2 * np.pi)
        dev = np.max(np.abs(k - np.round(k)))
        if sorted(np.mod(np.round(k).astype(int), nu)) != list(range(nu)):
            dev = max(dev, 1.0)   # not each root exactly once
        worst_root = max(worst_root, dev)
    print(f"  n={n}: {sols}/{ntr} solutions; max|d_l - 1| = {worst_d:.1e}; max slot-dependence of xi_l/xi_0 = {worst_diag:.1e}; "
          f"max deviation of {{zeta_l}} from mu_nu (each once) = {worst_root:.1e}")

print("(B) over-complete: N = nu+1 free complex d_l, minimise sum |d_l|^r subject to (JF) (penalty continuation)")


def cost_B(x, n, N, r, mu):
    nu = n + 1
    d = x[:N] * np.exp(1j * x[N:2 * N]); th = x[2 * N:].reshape(N, n)
    res = 0.0
    for T in subsets(n):
        z = np.sum(d * np.exp(1j * th[:, list(T)].sum(axis=1))) - (nu if len(T) == 0 else 0)
        res += abs(z) ** 2
    return np.sum((x[:N] ** 2 + 1e-12) ** (r / 2)) + mu * res, res


for n in [2, 3]:
    nu = n + 1; N = nu + 1
    for r in [0.5, 0.8]:
        best = np.inf; bestx = None
        for _ in range(40):
            x = np.concatenate([rng.uniform(0.2, 1.5, N), rng.uniform(0, 2 * np.pi, N + N * n)])
            for mu in [1, 10, 100, 1e3, 1e4, 1e5]:
                x = minimize(lambda y: cost_B(y, n, N, r, mu)[0], x, method="BFGS", options={"maxiter": 3000, "gtol": 1e-10}).x
            c, res = cost_B(x, n, N, r, 0)
            if res < 1e-8 and c < best:
                best = c; bestx = x
        mods = np.sort(np.abs(bestx[:N]))[::-1]
        print(f"  n={n} r={r}: min feasible sum|d|^r = {best:.6f} (bound nu = {nu}); moduli of best = {np.round(mods, 4)}")

print("(C) Craigen F4(a), T={1,3}: numerical vs 4|sin a|")
for a in [0.0, 0.3, 1.0, np.pi / 2, 2.5, np.pi]:
    e = np.exp(1j * a)
    M = np.array([[1, 1, 1, 1], [1, 1j * e, -1, -1j * e], [1, -1, 1, -1], [1, -1j * e, -1, 1j * e]])
    xi = M[:, 1:] / M[:, :-1]
    v = abs((xi[:, 0] * xi[:, 2]).sum())
    orth = np.max(np.abs(M.conj() @ M.T - 4 * np.eye(4)))
    print(f"  a={a:.3f}: orth defect {orth:.1e}; |sum xi1 xi3| = {v:.6f}; 4|sin a| = {4 * abs(np.sin(a)):.6f}")

print("(D) normal upper-triangular with unit diagonal is I: random U D U^* never upper triangular unless D = I")
worst = 0
for _ in range(200):
    nu = rng.integers(2, 7)
    Q, _ = np.linalg.qr(rng.normal(size=(nu, nu)) + 1j * rng.normal(size=(nu, nu)))
    D = np.diag(np.exp(1j * rng.uniform(0, 2 * np.pi, nu)))
    H = Q @ D @ Q.conj().T
    off = np.linalg.norm(H, 'fro') ** 2 - np.sum(np.abs(np.diag(H)) ** 2)
    # Frobenius identity for normal matrices: ||H||_F^2 = sum |eig|^2 = nu
    worst = max(worst, abs(np.linalg.norm(H, 'fro') ** 2 - nu))
print(f"  ||UDU*||_F^2 = nu to {worst:.1e} (so triangular with unimodular diagonal forces zero off-diagonal)")
