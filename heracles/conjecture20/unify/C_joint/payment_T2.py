# Single-T payment for GENERAL (non-aligned) atoms: penalty continuation.
# maximise  (|A| - c_n |B_T|)/(nu^{1-s} Q^s)  subject to B_{T'} = 0 (T' != empty, T).
# Report the max ratio (<= 1 iff P_T holds on the sample) and the residual constraint violation.
import numpy as np, itertools, sys
from scipy.optimize import minimize
rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
def subsets(n): return [T for k in range(1, n+1) for T in itertools.combinations(range(n), k)]
def setup(n, T):
    others = [Tp for Tp in subsets(n) if Tp != T]
    Mo = np.array([[1 if t in Tp else 0 for t in range(n)] for Tp in others], float)
    mT = np.array([1 if t in T else 0 for t in range(n)], float)
    return Mo, mT
def run(n, s, T, m, starts, aligned):
    nu = n+1; r = 1/s; c = 1-(n/nu)**(s-1); Mo, mT = setup(n, T)
    def parts(x):
        rho = np.exp(x[:m]); psi = np.zeros(m) if aligned else x[m:2*m]; th = x[(m if aligned else 2*m):].reshape(m, n)
        cpx = rho*np.exp(1j*psi)
        Bo = (cpx[None, :]*np.exp(1j*(Mo@th.T))).sum(1)
        BT = np.sum(cpx*np.exp(1j*(th@mT))); A = np.sum(cpx)
        return rho, A, BT, Bo
    def obj(x, mu):
        rho, A, BT, Bo = parts(x); L = rho.sum(); Q = np.sum(rho**r)
        val = (abs(A) - c*abs(BT))/(nu**(1-s)*Q**s)
        return -(val - mu*np.sum(np.abs(Bo)**2)/L**2) + 1e-4*np.log(L)**2
    best = (-np.inf, None, None)
    for k in range(starts):
        x = np.concatenate([rng.normal(0, 0.7, m), ([] if aligned else rng.uniform(-np.pi, np.pi, m)), rng.uniform(-np.pi, np.pi, m*n)])
        for mu in [1, 10, 100, 1e3, 1e4, 1e5, 1e6]:
            res = minimize(obj, x, args=(mu,), method='BFGS', options={'maxiter': 3000, 'gtol': 1e-10})
            x = res.x
        rho, A, BT, Bo = parts(x); L = rho.sum(); Q = np.sum(rho**r)
        val = (abs(A) - c*abs(BT))/(nu**(1-s)*Q**s); viol = np.abs(Bo).max()/L
        if viol < 1e-5 and val > best[0]: best = (val, viol, (abs(A)/L, abs(BT)/L))
    return best, c
if __name__ == '__main__':
    n = int(sys.argv[2]); starts = int(sys.argv[3]); nu = n+1
    ms = sorted(set([n, n+1, n+2, 2*n]))
    for s in ([1.5, (nu/2+1.5)/2, nu/2] if n > 2 else [1.25, 1.5]):
        for T in ([tuple(range(n)), (0,)] + ([(0, 1)] if n >= 3 else [])):
            for aligned in [False, True]:
                res = [run(n, s, T, m, starts, aligned) for m in ms]
                print(f"n={n} s={s:.3f} T={T} aligned={aligned} c_n={res[0][1]:.4f} | " + "  ".join(
                    (f"m={mm}: {b[0]:.6f} (viol {b[1]:.0e}, |A|/L={b[2][0]:.3f}, |B_T|/L={b[2][1]:.3f})" if b[1] is not None else f"m={mm}: none")
                    for mm, (b, cc) in zip(ms, res)), flush=True)
