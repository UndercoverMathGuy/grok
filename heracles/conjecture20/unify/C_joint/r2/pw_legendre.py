# PW_C test on the J6 Legendre single-T networks: alpha=0, beta_l e^{i phi_l} = 2 chihat(l), l in F.
# Feature l: Re Phi_l(w) = beta_l (1 + cos(phi_l + 2 pi l w/p)) (w = 2 x_T), cost Q_l = n^{1-r} beta_l^r (n-block).
# PW_C asks for some w with  sum_l (Re Phi_l(w))_+^r <= nu^{r-1} sum_l Q_l = (n/nu)^{1-r} sum_l beta_l^r.
import numpy as np, sys
def legendre(p):
    x = np.arange(p); chi = np.zeros(p)
    qr = set((x[1:]**2) % p); chi[1:] = [1 if a in qr else -1 for a in x[1:]]
    return chi
def run(p, n, s):
    r = 1/s; nu = n+1; F = (p-1)//2; l = np.arange(1, F+1)
    chi = legendre(p); w = np.arange(p)
    chihat = np.array([np.sum(chi*np.exp(-2j*np.pi*k*w/p))/p for k in l])
    beta = 2*np.abs(chihat); phi = np.angle(chihat)
    # check potential: sum_l beta cos(phi + 2 pi l w/p) == chi(w)
    V = (beta[None, :]*np.cos(phi[None, :] + 2*np.pi*np.outer(w, l)/p)).sum(1)
    err = np.max(np.abs(V - chi))
    RePhi = beta[None, :]*(1 + np.cos(phi[None, :] + 2*np.pi*np.outer(w, l)/p))
    lhs = (np.maximum(RePhi, 0)**r).sum(1)
    rhs = (n/nu)**(1-r)*np.sum(beta**r)
    return err, lhs.min()/rhs, lhs.mean()/rhs
for p in [7, 11, 13, 19, 23, 29, 31, 43, 47, 59, 61, 101, 103, 199, 211, 401, 499, 1009, 1019]:
    out = []
    for (n, s) in [(2, 1.5), (2, 1.25), (3, 2.0), (5, 3.0), (10, 5.5)]:
        err, mn, me = run(p, n, s); out.append(f"(n={n},s={s}) min/rhs={mn:.4f} mean/rhs={me:.4f}")
    print(f"p={p} (p mod 4 = {p%4}) potential err {err:.1e} | " + " | ".join(out), flush=True)
