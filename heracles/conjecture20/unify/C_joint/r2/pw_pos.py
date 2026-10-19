# Maximise the PW_C ratio over single-T networks (alpha_l, beta_l, phi_l on all of F) subject to margin h >= 0.
import numpy as np, sys
from scipy.optimize import minimize
def setup(p):
    F = (p-1)//2; l = np.arange(1, F+1); w = np.arange(p); d = np.arange(1, p)
    TH = 2*np.pi*np.outer(w, l)/p                       # (p, F)
    KAP = 1 - np.cos(2*np.pi*np.outer(d, l)/p)          # (p-1, F)
    WD = (w[:, None] - d[None, :]) % p                  # (p, p-1)
    return F, l, TH, KAP, WD
def evaluate(x, p, n, s, S):
    F, l, TH, KAP, WD = S; r = 1/s; nu = n+1; lam = (n/nu)**(1-r)
    al = np.exp(x[:F]); be = np.exp(x[F:2*F]); ph = x[2*F:]
    C = np.cos(ph[None, :] + TH)                        # (p, F) cos(phi + theta w)
    Rows = al[None, :] + be[None, :]*(1 + C)
    pw = (Rows**(1/s)).sum(1).min()/(np.sum(al**r) + lam*np.sum(be**r))
    V = C @ be
    G = (KAP @ (al+be))[None, :] + V[:, None] - V[WD]  # (p, p-1)
    return pw, G.min()/np.sum(al+be)
def obj(x, p, n, s, S, tau, mu):
    F, l, TH, KAP, WD = S; r = 1/s; nu = n+1; lam = (n/nu)**(1-r)
    al = np.exp(x[:F]); be = np.exp(x[F:2*F]); ph = x[2*F:]
    C = np.cos(ph[None, :] + TH); Sn = np.sin(ph[None, :] + TH)
    Rows = al[None, :] + be[None, :]*(1 + C); cost = np.sum(al**r) + lam*np.sum(be**r)
    rv = (Rows**r).sum(1)/cost; m = rv.min(); e = np.exp(-(rv-m)/tau); q = e/e.sum(); pw = m - tau*np.log(e.sum())
    # grads of pw
    dRow = r*Rows**(r-1)                               # (p,F)
    g_al = (q[:, None]*dRow).sum(0)/cost - pw*r*al**(r-1)/cost
    g_be = (q[:, None]*dRow*(1+C)).sum(0)/cost - pw*lam*r*be**(r-1)/cost
    g_ph = (q[:, None]*dRow*(-be[None, :]*Sn)).sum(0)/cost
    # margin (normalised by sum(al+be)) soft-min
    V = C @ be; tot = np.sum(al+be)
    G = ((KAP @ (al+be))[None, :] + V[:, None] - V[WD])/tot
    gm = G.min(); E = np.exp(-(G-gm)/tau); Q = E/E.sum(); hs = gm - tau*np.log(E.sum())
    pen = np.log1p(np.exp(-hs/0.01))*0.01             # softplus(-h)
    dpen = -1/(1+np.exp(hs/0.01))                      # d pen / d hs
    # dhs/d(al,be,ph)
    Qd = Q.sum(0)                                      # weights per delta
    dh_ab = (Qd @ KAP)/tot - hs/tot                    # common part for al and be
    # V terms: sum_{w,d} Q[w,d] (V[w] - V[w-d]);  dV[w]/dbe_l = C[w,l], dV/dph_l = -be_l Sn[w,l]
    Qw = Q.sum(1); Qwd = np.zeros(len(V)); np.add.at(Qwd, WD.ravel(), Q.ravel())
    coef = (Qw - Qwd)                                  # (p,)
    dh_be = dh_ab + (coef @ C)/tot
    dh_ph = (coef @ (-be[None, :]*Sn))/tot
    val = pw - mu*pen
    grad = np.concatenate([(g_al - mu*dpen*dh_ab)*al, (g_be - mu*dpen*dh_be)*be, g_ph - mu*dpen*dh_ph])
    return -val, -grad
if __name__ == '__main__':
    n, s = int(sys.argv[1]), float(sys.argv[2]); ps = [int(a) for a in sys.argv[3:]]
    for p in ps:
        S = setup(p); F = S[0]; best = (-1, None)
        for sd in range(8):
            rng = np.random.default_rng(sd)
            x = np.concatenate([np.full(F, -4.0), rng.normal(0, 0.1, F), rng.uniform(0, 2*np.pi, F)])
            for tau, mu in [(0.05, 5), (0.02, 20), (0.01, 50), (0.005, 100), (0.002, 300), (0.001, 1000)]:
                x = minimize(obj, x, args=(p, n, s, S, tau, mu), jac=True, method='L-BFGS-B', options={'maxiter': 800}).x
            pw, h = evaluate(x, p, n, s, S)
            if h >= 0 and pw > best[0]: best = (pw, x.copy(), h)
        if best[1] is not None:
            np.save(f"pwpos_p{p}_n{n}_s{s}.npy", best[1])
            print(f"n={n} s={s} p={p}: best PW_C ratio with h>=0: {best[0]:.5f} (normalised margin {best[2]:.2e})", flush=True)
        else: print(f"n={n} s={s} p={p}: no positive-margin point found", flush=True)
