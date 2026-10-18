# Joint (all slot-sets) two-sided L^2 detection in class C, general n.
#   Z_d(x) = sum_{l, T != empty} Re(a_{l,T} c_l(d) rho^{2 l s_T(x)}),  x in Z_p^n
#   A = -min_x Z_d,  A' = -min_x Z_{-d},  E = sum |a_{l,T}|^2 kappa_l(d) = Var Z_d.
# Report min over (moduli, phases) of  AA'/E (product form) and ((A+A')/2)^2/E (mean form).
import numpy as np, sys, itertools
from scipy.optimize import minimize
p = int(sys.argv[1]); n = int(sys.argv[2]); seed = int(sys.argv[3]); reps = int(sys.argv[4]); mode = sys.argv[5]
rng = np.random.default_rng(seed)
F = np.arange(1, (p-1)//2 + 1); K = len(F)
Ts = [T for T in itertools.product([0,1], repeat=n) if any(T)]; NT = len(Ts)
X = np.indices((p,)*n).reshape(n, -1)
ph = [2*np.pi*np.outer((np.array(T) @ X) % p, 2*F)/p for T in Ts]
def parts(d):
    c = 1 - np.exp(-2j*np.pi*F*d/p); return np.abs(c), np.angle(c), 1 - np.cos(2*np.pi*F*d/p)
def smin(v, tau):
    m = v.min(); w = np.exp(-tau*(v - m)); return m - np.log(w.mean())/tau, w/w.sum()
def obj(z, d, tau):
    mod = np.exp(z[:NT*K]).reshape(NT, K); ang = z[NT*K:].reshape(NT, K)
    cabs, carg, kap = parts(d); E = (mod**2*kap).sum()
    vals = []; grads = []
    for sgn in (+1, -1):
        Zv = np.zeros(X.shape[1]); args = []
        for t in range(NT):
            a = ph[t] + ang[t] + sgn*carg; args.append(a); Zv += (np.cos(a)*(mod[t]*cabs)).sum(1)
        if tau is None: vals.append(-Zv.min()); continue
        m, w = smin(Zv, tau); vals.append(-m)
        gm = np.zeros((NT, K)); ga = np.zeros((NT, K))
        for t in range(NT):
            gm[t] = -(w @ np.cos(args[t]))*cabs; ga[t] = -(w @ (-np.sin(args[t])))*mod[t]*cabs
        grads.append((gm, ga))
    A, Ap = vals
    if tau is None: return (A*Ap/E, ((A+Ap)/2)**2/E)
    if mode == 'prod':
        r = A*Ap/E; dA = [Ap/E, A/E]
    else:
        r = ((A+Ap)/2)**2/E; dA = [(A+Ap)/2/E, (A+Ap)/2/E]
    gm = dA[0]*grads[0][0] + dA[1]*grads[1][0] - r*(2*mod*kap)/E
    ga = dA[0]*grads[0][1] + dA[1]*grads[1][1]
    return r, np.concatenate([(gm*mod).ravel(), ga.ravel()])
res_all = []
for d in range(1, (p-1)//2 + 1):
    best = (np.inf, None)
    for rep in range(reps):
        z = np.concatenate([rng.normal(size=NT*K), rng.uniform(0, 2*np.pi, NT*K)])
        if rep % 3 == 0: z[:NT*K] -= 8*np.repeat(np.arange(NT) != rng.integers(NT), K)
        if rep % 3 == 1: z[:NT*K] -= 8*np.tile(np.arange(K) != rng.integers(K), NT)
        for tau in (4.0, 16.0, 64.0, 256.0, 1024.0):
            z[:NT*K] -= z[:NT*K].max(); z[:NT*K] = np.maximum(z[:NT*K], -30)
            z = minimize(obj, z, args=(d, tau), jac=True, method='L-BFGS-B', options={'maxiter': 300}).x
        r = obj(z, d, None)
        val = r[0] if mode == 'prod' else r[1]
        if val < best[0]: best = (val, r)
    res_all.append(best[0])
    print(f"p={p} n={n} d={d} [{mode}]: min = {best[0]:.4f}   (prod {best[1][0]:.4f}, mean^2 {best[1][1]:.4f})", flush=True)
print(f"p={p} n={n} [{mode}] overall min {min(res_all):.4f}")
