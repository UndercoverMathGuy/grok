# Single-feature Re-covering test: maximise min_{z in mu_p^n} Re P(z) / (nu^{1-s} Q^s)
# over class-C atom systems P(z) = sum_i beta_i prod_t (1 + xi_it z_t), Q = sum |beta_i|^r.
# Also report the cone ratio min cm(P) / (nu^{1-s} Q^s), cm(z) = Re z - cot(pi/p)|Im z|.
import numpy as np, itertools, sys
from scipy.optimize import minimize
def grid(p, n):
    rt = np.exp(2j*np.pi*np.arange(p)/p)
    return np.array(list(itertools.product(rt, repeat=n)))      # (p^n, n)
def field(Z, b, X):   # b: (m,) complex; X: (m,n) unimodular
    return (1 + Z[:, None, :]*X[None, :, :]).prod(2) @ b
def run(p, n, s, m, restarts, seed, mode='re'):
    r = 1/s; nu = n+1; Z = grid(p, n); cot = 1/np.tan(np.pi/p)
    rng = np.random.default_rng(seed); best = (-np.inf, None)
    def unpack(x):
        lb = x[:m]; pb = x[m:2*m]; th = x[2*m:].reshape(m, n)
        return np.exp(lb)*np.exp(1j*pb), np.exp(1j*th)
    def obj(x, tau):
        b, X = unpack(x); P = field(Z, b, X)
        v = P.real if mode == 're' else P.real - cot*np.abs(P.imag)
        Q = np.sum(np.abs(b)**r); den = nu**(1-s)*Q**s
        g = v/den; mn = g.min(); return -(mn - tau*np.log(np.exp(-(g-mn)/tau).sum()))
    for t in range(restarts):
        x = np.concatenate([rng.normal(0, 0.5, m), rng.uniform(0, 2*np.pi, m), rng.uniform(0, 2*np.pi, m*n)])
        for tau in [3e-2, 1e-2, 3e-3, 1e-3, 3e-4]:
            x = minimize(obj, x, args=(tau,), method='L-BFGS-B', options={'maxiter': 400}).x
        b, X = unpack(x); P = field(Z, b, X)
        Q = np.sum(np.abs(b)**r); den = nu**(1-s)*Q**s
        val = (P.real.min() if mode == 're' else (P.real - cot*np.abs(P.imag)).min())/den
        if val > best[0]: best = (val, (b, X))
    return best
if __name__ == '__main__':
    p, n, s, m, R, mode = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
    val, (b, X) = run(p, n, s, m, R, 1, mode)
    A = b.sum(); L = np.abs(b).sum()
    Pf = field(grid(p, n), b, X)
    print(f"p={p} n={n} s={s:.3f} m={m} mode={mode}: best ratio {val:.6f}  |A|/L={abs(A)/L:.4f} "
          f"minRe/|A|={Pf.real.min()/abs(A):.4f}  E|P-A|^2/|A|^2={np.mean(np.abs(Pf-A)**2)/abs(A)**2:.4f}", flush=True)
    np.save(f"sf_{mode}_p{p}_n{n}_s{s:.3f}_m{m}.npy", np.concatenate([b, X.ravel()]))
