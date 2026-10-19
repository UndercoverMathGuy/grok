# Maximise FRE/IF over single-T networks (alpha_l IF + beta_l n-block, phases), where
#   FRE = min_w min_delta sum_l (alpha_l + beta_l + beta_l cos(phi_l + 2 pi l w/p)) kappa_l(delta)  (>= h)
#   IF  = Gamma_s (sum alpha^r + lam sum beta^r)^s, lam = (n/nu)^{1-r}.
# Also report the true ratio h/IF of the optimiser's network.
import numpy as np, sys
sys.path.insert(0, '/home/user/grok/heracles/conjecture20/unify/C_joint')
from cjlib import gamma_exact, gamma_lp
from scipy.optimize import minimize
def run(p, n, s, Gam, seeds=8):
    r = 1/s; nu = n+1; lam = (n/nu)**(1-r); F = (p-1)//2; l = np.arange(1, F+1)
    TH = 2*np.pi*np.outer(np.arange(p), l)/p; KAP = 1 - np.cos(2*np.pi*np.outer(np.arange(1, p), l)/p)
    WD = (np.arange(p)[:, None] - np.arange(1, p)[None, :]) % p
    def parts(x):
        al = np.exp(np.clip(x[:F], -30, 30)); be = np.exp(np.clip(x[F:2*F], -30, 30)); ph = x[2*F:]
        return al, be, ph
    def f(x, tau):
        al, be, ph = parts(x); C = np.cos(ph[None, :] + TH); S = np.sin(ph[None, :] + TH)
        Y = (al + be)[None, :] + be[None, :]*C                    # field rows (p, F)
        cost = np.sum(al**r) + lam*np.sum(be**r); sc = Gam*cost**s
        Psi = (Y @ KAP.T)/sc                                       # (p, p-1)
        m = Psi.min(); E = np.exp(-(Psi - m)/tau); Z = E.sum(); val = m - tau*np.log(Z); Q = E/Z
        G = Q.T @ np.ones(p)                                       # unused
        dY = Q @ KAP                                               # (p, F): dval/dY / sc
        g_al = dY.sum(0)/sc - val*s*r*al**(r-1)/cost
        g_be = (dY*(1 + C)).sum(0)/sc - val*s*lam*r*be**(r-1)/cost
        g_ph = (dY*(-be[None, :]*S)).sum(0)/sc
        pen = 1e-3*np.log(cost)**2; dp = 2e-3*np.log(cost)/cost
        return -(val - pen), -np.concatenate([(g_al - dp*r*al**(r-1))*al, (g_be - dp*lam*r*be**(r-1))*be, g_ph])
    best = (-1, None)
    for sd in range(seeds):
        rng = np.random.default_rng(sd)
        x = np.concatenate([rng.normal(-1, 1, F), rng.normal(-1, 1, F), rng.uniform(0, 2*np.pi, F)])
        for tau in [3e-2, 1e-2, 3e-3, 1e-3, 3e-4]:
            x = minimize(f, x, args=(tau,), jac=True, method='L-BFGS-B', options={'maxiter': 1500}).x
        al, be, ph = parts(x); C = np.cos(ph[None, :] + TH)
        cost = np.sum(al**r) + lam*np.sum(be**r); sc = Gam*cost**s
        Y = (al + be)[None, :] + be[None, :]*C
        fre = (Y @ KAP.T).min()/sc
        V = C @ be; G = (KAP @ (al+be))[None, :] + V[:, None] - V[WD]
        h = G.min()/sc
        if fre > best[0]: best = (fre, h, np.sum(be)/np.sum(al+be))
    return best
for (p, n, s) in [(7, 2, 1.5), (11, 2, 1.5), (13, 2, 1.5), (17, 2, 1.5), (19, 2, 1.5), (23, 2, 1.5), (13, 3, 2.0), (17, 3, 2.0), (13, 5, 3.0), (19, 2, 1.2)]:
    Gam = gamma_exact(p, s)[0] if p <= 19 else gamma_lp(p, s, restarts=30)[0]
    fre, h, frac = run(p, n, s, Gam)
    print(f"(p,n,s)=({p},{n},{s}): max FRE/IF = {fre:.5f}  (its h/IF = {h:.5f}, interference share {frac:.3f})", flush=True)
