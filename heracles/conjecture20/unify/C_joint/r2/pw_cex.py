# Explicit refutation of PW_C (J12 of the first C_joint report) at (p,n,b) = (17,2,2), and (19,3,2).
# Network: for every l in F, n balanced class-C neurons at frequency l with atoms beta/n,
# xi_{k,t} = e^{2 pi i k/n} (times e^{i phi_l} in slot 1)  =>  Phi_l(x) = beta (1 + e^{i phi_l} rho^{2 l y(x)}).
# Phases phi_l optimised to make min_x sum_l (Re Phi_l)_+^r large.  Verification from raw tables.
import numpy as np, itertools, sys, mpmath as mp
sys.path.insert(0, '/home/user/grok/heracles/conjecture20/unify/C_joint')
from cjlib import gamma_exact
from scipy.optimize import minimize
mp.mp.dps = 40
def opt_phases(p, n, s, seeds=6):
    r = 1/s; nu = n+1; F = (p-1)//2; l = np.arange(1, F+1); w = np.arange(p); M = 2*np.pi*np.outer(w, l)/p
    den = (n/nu)**(1-r)*F; best = (-1, None)
    for sd in range(seeds):
        phi = np.random.default_rng(sd).uniform(0, 2*np.pi, F)
        for tau in [0.05, 0.02, 0.01, 0.005, 0.002, 0.001]:
            def f(ph):
                c = np.maximum(1 + np.cos(ph[None, :] + M), 1e-12)
                v = (c**r).sum(1)/den; mn = v.min(); e = np.exp(-(v-mn)/tau); q = e/e.sum()
                return -(mn - tau*np.log(e.sum())), -(q[:, None]*(r*c**(r-1)*(-np.sin(ph[None, :] + M)))).sum(0)/den
            phi = minimize(f, phi, jac=True, method='L-BFGS-B', options={'maxiter': 1000}).x
        v = ((1 + np.cos(phi[None, :] + M))**r).sum(1).min()/den
        if v > best[0]: best = (v, phi.copy())
    return best
def tables(p, n, phi, beta=1.0):
    """class-C neurons; returns U (m,n,p), W (m,p) with field c_i prod_t(1 + xi_it rho^{2 l x_t})"""
    F = (p-1)//2; a = np.arange(p); U = []; W = []; feat = []
    for l in range(1, F+1):
        for k in range(n):
            xi = np.array([np.exp(2j*np.pi*k/n)]*n); xi[0] *= np.exp(1j*phi[l-1]); c = beta/n
            th = np.angle(xi)/2
            U.append([np.cos(2*np.pi*l*a/p + th[t]) for t in range(n)])
            psi = np.angle(c) + th.sum(); A = 2**n*abs(c)
            W.append(A*np.cos(2*np.pi*l*a/p + psi)); feat.append(l)
    return np.array(U), np.array(W), np.array(feat)
def verify(p, n, s, phi):
    r = 1/s; nu = n+1; F = (p-1)//2; rho = np.exp(2j*np.pi/p)
    U, W, feat = tables(p, n, phi); m = len(W)
    # table f(x)[c]
    X = np.array(list(itertools.product(range(p), repeat=n)))           # (p^n, n)
    prodU = np.ones((m, len(X)))
    for t in range(n): prodU *= U[:, t, X[:, t]]
    f = prodU.T @ W                                                    # (p^n, p)
    y = X.sum(1) % p
    # fields from the table
    fc = np.fft.fft(f, axis=1)                                         # sum_c f[c] rho^{-kc}
    Phi = np.array([(2.0/p)*rho**(l*y)*fc[:, l] for l in range(1, F+1)])   # (F, p^n)
    pred = np.array([1 + np.exp(1j*phi[l-1])*rho**(2*l*y) for l in range(1, F+1)])
    ferr = np.max(np.abs(Phi - pred))
    # clean atoms beta_il = 2 p^{-nu} conj(what_i(l)) prod_t uhat_it(l)
    Uh = np.fft.fft(U, axis=2); Wh = np.fft.fft(W, axis=1)
    bet = np.array([2*p**(-nu)*np.conj(Wh[i, feat[i]])*np.prod(Uh[i, :, feat[i]]) for i in range(m)])
    aerr = np.max(np.abs(np.abs(bet) - 1.0/n))
    Qtot = np.sum(np.abs(bet)**r)
    # PW_C in exact (40-digit) arithmetic using the analytic field (validated above to ferr)
    best = None
    for w in range(p):
        S = mp.mpf(0)
        for l in range(1, F+1):
            S += (1 + mp.cos(mp.mpf(phi[l-1]) + 2*mp.pi*l*2*w/p))**(mp.mpf(1)/mp.mpf(s))
        best = S if best is None or S < best else best
    rhs = mp.mpf(nu)**(mp.mpf(1)/s - 1)*mp.mpf(Qtot)
    # margin h from the table
    h = min((f[ix, y[ix]] - f[ix, (y[ix]-d) % p]).min() for ix in [slice(None)] for d in range(1, p))
    # gamma: balance each neuron (product of norms fixed), cost sum ||omega||^b, gamma = h/(sum ||omega||^b)^{nu/b}
    norms = np.concatenate([np.linalg.norm(U, axis=2), np.linalg.norm(W, axis=1)[:, None]], axis=1)   # (m, nu)
    om2 = nu*np.prod(norms, axis=1)**(2.0/nu); b = nu/s
    gam = h/np.sum(om2**(b/2))**(nu/b)
    Bp = 2*(2*p*nu)**(-nu/2.0)
    G, _ = gamma_exact(p, s)
    gIF = nu**(1-s)*Bp*G
    return ferr, aerr, float(best), float(rhs), float(best/rhs), h, gam/gIF, G
for (p, n, s) in [(17, 2, 1.5), (19, 2, 1.5), (19, 3, 2.0), (23, 2, 1.5)]:
    v, phi = opt_phases(p, n, s)
    ferr, aerr, lhs, rhs, ratio, h, g, G = verify(p, n, s, phi)
    np.save(f"pwcex_p{p}_n{n}_s{s}.npy", phi)
    print(f"(p,n,s)=({p},{n},{s}): field err {ferr:.1e}, atom err {aerr:.1e} | PW_C: min_x sum psi = {lhs:.10f} > "
          f"nu^(r-1) sum Q = {rhs:.10f}  ratio {ratio:.6f} | margin h = {h:.4f}  gamma/gamma_IF = {g:.4f} (Gamma_s exact {G:.6f})", flush=True)
