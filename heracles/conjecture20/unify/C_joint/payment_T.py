# Single-T payment P_T: for atoms (c_i, xi_it) with B_{T'}=0 for every T' not in {empty, T},
# test  |A| - c_n |B_T| <= nu^{1-s} Q^s,  c_n = 1-(n/nu)^{s-1}, Q = sum |c_i|^{1/s}.
# SLSQP with equality constraints, many random starts; aligned and general atoms.
import numpy as np, itertools, sys
from scipy.optimize import minimize
rng = np.random.default_rng(1)

def subsets(n):
    return [T for k in range(1, n+1) for T in itertools.combinations(range(n), k)]

def run(n, s, T, m, aligned, starts=40):
    nu = n+1; r = 1/s; c = 1-(n/nu)**(s-1)
    others = [Tp for Tp in subsets(n) if Tp != T]
    def unpackx(x):
        rho = np.exp(x[:m]); psi = np.zeros(m) if aligned else x[m:2*m]
        th = x[(m if aligned else 2*m):].reshape(m, n)
        return rho, psi, th
    def coeff(x, Tp):
        rho, psi, th = unpackx(x)
        return np.sum(rho*np.exp(1j*(psi + th[:, list(Tp)].sum(1))))
    def obj(x):
        rho, psi, th = unpackx(x)
        Q = np.sum(rho**r); A = np.abs(np.sum(rho*np.exp(1j*psi))); D = np.abs(coeff(x, T))
        return -((A - c*D)/(nu**(1-s)*Q**s))
    cons = []
    for Tp in others:
        cons.append({'type': 'eq', 'fun': (lambda x, Tp=Tp: np.real(coeff(x, Tp)))})
        cons.append({'type': 'eq', 'fun': (lambda x, Tp=Tp: np.imag(coeff(x, Tp)))})
    best = -np.inf; bx = None
    dim = m + (0 if aligned else m) + m*n
    for k in range(starts):
        x0 = np.concatenate([rng.normal(0, 0.5, m), (np.zeros(0) if aligned else rng.uniform(-np.pi, np.pi, m)), rng.uniform(-np.pi, np.pi, m*n)])
        res = minimize(obj, x0, constraints=cons, method='SLSQP', options={'maxiter': 2000, 'ftol': 1e-13})
        viol = max(abs(cc['fun'](res.x)) for cc in cons) / np.sum(np.exp(res.x[:m]))
        if viol < 1e-8 and -res.fun > best:
            best = -res.fun; bx = res.x
    return best, c

if __name__ == '__main__':
    for n in [2, 3, 4]:
        nu = n+1
        for s in sorted(set([1.25, 1.5, nu/2])):
            if s > nu/2: continue
            for T in [tuple(range(n)), (0,)] + ([(0, 1)] if n >= 3 else []):
                for aligned in [True, False]:
                    out = []
                    for m in [n, n+1, n+2, 2*n+1]:
                        b, c = run(n, s, T, m, aligned, starts=25 if n < 4 else 12)
                        out.append(b)
                    print(f"n={n} s={s:.3f} T={T} aligned={aligned} c_n={c:.4f} max ratio (|A|-c|B_T|)/(nu^(1-s)Q^s) over m={[n,n+1,n+2,2*n+1]}: " + " ".join(f"{v:.6f}" for v in out), flush=True)
