# Referee: adversarial gamma maximisation in class C at p = 3, 5 (n = 2), compared with the IF value (R7 covers p = 3).
import numpy as np, sys
from scipy.optimize import minimize
from netlib import table, gaps, Bp
sys.path.insert(0, '/home/user/grok/heracles/conjecture20/mechanism_scripts'); import prog
rng = np.random.default_rng(61)
def make(z, fr, p):
    m = len(fr); z = z.reshape(m, 6); a = np.arange(p); ang = 2*np.pi*np.outer(fr, a)/p
    return z[:, [0]]*np.cos(ang + z[:, [1]]), z[:, [2]]*np.cos(ang + z[:, [3]]), z[:, [4]]*np.cos(ang + z[:, [5]])
def neg_gamma(z, fr, p, b, tau):
    U1, U2, W = make(z, fr, p); g = gaps(table(U1, U2, W))[:, :, 1:].ravel()
    nrm = (np.sum((p/2)*(z.reshape(-1, 6)[:, [0, 2, 4]]**2), 1)**(b/2)).sum()**(3/b)
    if tau is None: return -g.min()/nrm
    sc = np.abs(g).max() + 1e-300; gm = g.min()
    return -(gm - np.log(np.mean(np.exp(-tau*(g - gm)/sc)))*sc/tau)/nrm
for p in [3, 5]:
    for b in [2.0, 2.5]:
        G = prog.solve(p, 3/b)[0] if p > 3 else 1.5**(3/b - 1)*0 + 1.5   # p=3: Gamma_s = kappa_1(1) = 3/2 for y = 1
        gIF = 3**(1 - 3/b)*Bp(p)*G; best = -np.inf
        for run in range(12):
            width = [1, 2, 3, 4, 6, 9][run % 6]
            fr = np.repeat(np.arange(1, (p - 1)//2 + 1), width)
            z = np.column_stack([rng.uniform(0.5, 1.5, len(fr)), rng.uniform(0, 2*np.pi, len(fr))]*3).ravel()
            for tau in [20.0, 80.0, 300.0, 1000.0, 3000.0]:
                z = minimize(neg_gamma, z, args=(fr, p, b, tau), method='L-BFGS-B', options=dict(maxiter=400)).x
            best = max(best, -neg_gamma(z, fr, p, b, None))
        print(f"class C, n=2, p={p}, b={b}: best gamma/gamma_IF over 12 runs = {best/gIF:.10f} (R7: <= 1)")
