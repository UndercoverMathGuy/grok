# J6/J8: arithmetic flat interference at ALL offsets, and its efficiency cost.
# (a) Legendre symbol on all of F: phi = chi, |W_l| = 2/sqrt p: depth_delta = 2, sum_l v_l kappa_l(delta) = sqrt p (every delta).
# (b) Cyclotomic class H_d (index d subgroup of Z_p^*, symmetric) with interference from a character psi of order 2d:
#     phi(w) = Re(e^{i th} sum_{l in H_d} psi(l) rho^{lw});  compare depth with coverage and the network ratio R
#     R = tau_n(s) * (min_delta sum v kappa - max depth) / (Gamma_ref * (sum v^r)^s)   (pure single-T features, alpha = 0)
#     Gamma_ref = max(reweighted-LP value, uniform values on F and on the H_e, e | (p-1)/2 small) <= Gamma_s  =>  R is an UPPER bound.
import numpy as np, sys
from cjlib import kappa_full, gamma_lp
def prim_root(p):
    for g in range(2, p):
        if all(pow(g, (p-1)//q, p) != 1 for q in set(f for f in range(2, p) if (p-1) % f == 0 and all(f % d for d in range(2, int(f**0.5)+1)))):
            return g
def dlog_table(p):
    g = prim_root(p); t = np.zeros(p, int); x = 1
    for k in range(p-1): t[x] = k; x = x*g % p
    return t
def tl_depths(p, W):
    # W: complex amplitudes for l=1..F ; returns per-delta depth = max_w [phi(w-delta) - phi(w)] and phi
    F = (p-1)//2; w = np.arange(p); l = np.arange(1, F+1)
    phi = np.real((W[None, :]*np.exp(2j*np.pi*np.outer(w, l)/p)).sum(1))
    dep = np.array([np.max(np.roll(phi, d) - phi) for d in range(1, p)])   # roll(phi,d)[w] = phi[w-d]
    return dep, phi
print("(a) Legendre on all of F")
for p in [11, 13, 101, 103, 401, 1009, 1013]:
    F = (p-1)//2; w = np.arange(p)
    chi = np.array([0] + [1 if pow(int(x), (p-1)//2, p) == 1 else -1 for x in range(1, p)], float)
    chat = np.array([np.sum(chi*np.exp(-2j*np.pi*l*w/p))/p for l in range(1, F+1)])
    W = 2*chat; v = np.abs(W)
    dep, phi = tl_depths(p, W)
    K = kappa_full(p)                       # (p-1, F)
    cov = K@v
    print(f"  p={p}: max|phi-chi|={np.max(np.abs(phi-chi)):.1e}  v_l in [{v.min():.5f},{v.max():.5f}] (2/sqrt p={2/np.sqrt(p):.5f})  "
          f"max_delta depth/sum v kappa = {np.max(dep/cov):.5f}  (2/sqrt p = {2/np.sqrt(p):.5f}), min = {np.min(dep/cov):.5f}")
print("(b) cyclotomic classes")
for p in [101, 197, 401, 409, 1009]:
    F = (p-1)//2; t = dlog_table(p); K = kappa_full(p)
    for d in [2, 4]:
        if (p-1) % (2*d) or (((p-1)//d) % 2): continue      # need psi of order 2d and H_d symmetric (-1 in H_d)
        H = [l for l in range(1, p) if t[l] % d == 0]
        if (p-1-0) % 2 == 0 and t[p-1] % d != 0: continue
        HF = [l for l in H if l <= F]
        best = None
        for th in np.linspace(0, np.pi, 13):
            # psi(l) = exp(2 pi i t[l]/(2d)); W_l for l in F: coefficient of rho^{lw} plus conj of coefficient of rho^{-lw}
            W = np.zeros(F, complex)
            for l in HF:
                c1 = np.exp(1j*th)*np.exp(2j*np.pi*t[l]/(2*d)); c2 = np.exp(1j*th)*np.exp(2j*np.pi*t[p-l]/(2*d))
                W[l-1] = c1 + np.conj(c2)
            v = np.abs(W)
            if v.max() < 1e-9: continue
            dep, phi = tl_depths(p, W)
            cov = K@v
            ratio = np.max(dep/cov)
            if best is None or ratio < best[0]: best = (ratio, W, v, dep, cov, th)
        ratio, W, v, dep, cov, th = best
        print(f"  p={p} d={d}: |H_d cap F|={len(HF)} v in [{v[v>0].min():.3f},{v.max():.3f}] max_delta depth/coverage={ratio:.4f} (2d/sqrt p={2*d/np.sqrt(p):.4f})")
        for n in [2, 3]:
            for s in [1.02, 1.05, 1.1, 1.2]:
                if s > (n+1)/2: continue
                r = 1/s; tau = (1+1/n)**(s-1)
                hmin = np.min(cov - dep)              # min over delta of (clean - depth); alpha=0, beta_l=v_l (pure single-T)
                G, y = gamma_lp(p, s, restarts=3)
                # uniform allocations on F and on cyclotomic classes as extra lower bounds for Gamma_s
                cands = [G]
                for e in [1, 2, 3, 4, 5, 6, 8]:
                    if (p-1) % e: continue
                    He = [l for l in range(1, F+1) if t[l] % e == 0]
                    if not He: continue
                    yy = np.zeros(F); yy[np.array(He)-1] = 1.0
                    cands.append(np.min(K@yy)/np.sum(yy**r)**s)
                Gref = max(cands)
                R = tau*hmin/(Gref*np.sum(v[v > 0]**r)**s)
                print(f"     n={n} s={s:.2f}: tau={tau:.4f} Gamma_ref={Gref:.5f} (lp {G:.5f}) R<= {R:.5f}")
