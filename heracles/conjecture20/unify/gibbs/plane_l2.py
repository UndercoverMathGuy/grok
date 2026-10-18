# Plane geometry (class C, n=2): how much of the full L^2 interference energy does the two-sided detection see?
#   Z_d(x)  = sum_{l,T} Re(a_{l,T} c_l(d) rho^{2 l s_T(x)}),   x in Z_p^2,  T in {1},{2},{1,2}
#   det(d)  = (1/2)(-min_x Z_d - min_x Z_{-d})   (lower bound for the mean slack mbar_d of any class-C network)
#   E(d)    = sum_{l,T} |a_{l,T}|^2 kappa_l(d)    (= Var Z_d)
# ratio r(d) = det(d)^2 / E(d).  Theorem loc (per-T Bhatia-Davis) proves r >= 1/3.  Closing class C at n=2 for all p <= p0
# with Theorem A needs r >= (p/2)/12.  We minimise r adversarially over moduli and phases.
import numpy as np, sys
from scipy.optimize import minimize
p = int(sys.argv[1]); seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
rng = np.random.default_rng(seed)
F = np.arange(1, (p-1)//2 + 1); K = len(F)
x1, x2 = np.meshgrid(np.arange(p), np.arange(p), indexing='ij'); x1 = x1.ravel(); x2 = x2.ravel()
S = [x1, x2, (x1 + x2) % p]
ph = [2*np.pi*np.outer(s, 2*F)/p for s in S]            # (p^2, K) phases of rho^{2 l s_T(x)}
def parts(d):
    c = 1 - np.exp(-2j*np.pi*F*d/p); return np.abs(c), np.angle(c), 1 - np.cos(2*np.pi*F*d/p)
def smin(v, tau):
    m = v.min(); w = np.exp(-tau*(v - m)); return m - np.log(w.mean())/tau, w/w.sum()
def objective(z, d, tau):
    mod = np.exp(z[:3*K]).reshape(3, K); ang = z[3*K:].reshape(3, K)
    cabs, carg, kap = parts(d)
    E = (mod**2 * kap).sum()
    det = 0; g_mod = np.zeros((3, K)); g_ang = np.zeros((3, K))
    for sgn in (+1, -1):
        Zv = np.zeros(p*p); args = []
        for t in range(3):
            a = ph[t] + ang[t] + sgn*carg; args.append(a)
            Zv += (np.cos(a)*(mod[t]*cabs)).sum(1)
        if tau is None:
            det += -0.5*Zv.min(); continue
        m, w = smin(Zv, tau); det += -0.5*m
        for t in range(3):
            g_mod[t] += -0.5*(w @ np.cos(args[t]))*cabs
            g_ang[t] += -0.5*(w @ (-np.sin(args[t])))*mod[t]*cabs
    if tau is None: return det**2/E
    r = det**2/E
    gm = (2*det*g_mod/E - det**2*(2*mod*kap)/E**2)*mod   # chain rule through exp
    ga = 2*det*g_ang/E
    return r, np.concatenate([gm.ravel(), ga.ravel()])
best = {}
for d in range(1, (p-1)//2 + 1):
    bd = np.inf
    for rep in range(int(sys.argv[3]) if len(sys.argv)>3 else 12):
        z0 = np.concatenate([rng.normal(scale=1.0, size=3*K), rng.uniform(0, 2*np.pi, 3*K)])
        if rep % 4 == 0:   # concentrate on one slot-set (line case)
            z0[:3*K] -= 8*np.repeat(np.arange(3) != (rep//4) % 3, K)
        if rep % 4 == 1:   # one feature only
            z0[:3*K] -= 8*np.tile(np.arange(K) != rng.integers(K), 3)
        z = z0
        for tau in (5.0, 20.0, 80.0, 300.0, 1000.0):
            z[:3*K] -= z[:3*K].max()          # scale invariance: renormalise moduli
            z[:3*K] = np.maximum(z[:3*K], -30)
            res = minimize(objective, z, args=(d, tau*np.exp(-z[:3*K].max())), jac=True, method='L-BFGS-B', options={'maxiter': 400})
            z = res.x
        r = objective(z, d, None)
        bd = min(bd, r)
    best[d] = bd
    print(f"p={p} d={d}: min det^2/E = {bd:.4f}   (per-T bound 1/3; needed for closure with Thm A: {(p/2)/12:.3f})", flush=True)
print("overall min", min(best.values()))
