# Where could the L2 relaxation become inconclusive?  Near the convex end (s -> 1, b -> nu) the programme optimum is dense
# (K* ~ F), so K_eff is large, while the payment constant theta(s) = (1 - 3^{1-s})/sqrt3 is small.  Uniform-eta safety factor
#   sigma(p, s) = theta(s) sqrt(K_eff) / rho_p,   K_eff = Gamma^2 / max_{delta in B} sum y*^2 kappa,   rho_p = sqrt((p-1)/(p+1)).
# sigma >= 1 means the relaxation admits profitable uniform cheapening of the optimum (the relaxation is then inconclusive,
# NOT a violation).  Also the full relaxation maximum UB over (y, eta) by L-BFGS (keff2 objective) for moderate F.
# Usage: python keff_dense.py s1,s2,... p1,p2,...
import sys, json, time, warnings
import numpy as np
from scipy.optimize import minimize, linprog
import hlib2 as H2
warnings.filterwarnings('ignore')
slist = [float(v) for v in sys.argv[1].split(',')]; plist = [int(v) for v in sys.argv[2].split(',')]
rng = np.random.default_rng(3)


def prog_rw(p, s, restarts):
    K = H2.kmat(p); F = K.shape[1]; r = 1 / s
    best = (np.inf, None)
    for t in range(restarts):
        w = np.ones(F) if t == 0 else rng.uniform(0.3, 1.7, F)
        for it in range(300):
            res = linprog(w, A_ub=-K, b_ub=-np.ones(F), bounds=(0, None), method='highs')
            v = res.x; wn = (v + 1e-12) ** (r - 1)
            if np.allclose(wn, w, rtol=1e-11): break
            w = wn
        v = v * (v > 1e-11); Q = np.sum(v[v > 0] ** r)
        if Q < best[0]: best = (Q, v.copy())
    Q, v = best
    return Q ** (-s), v / Q ** s


for s in slist:
    theta = (1 - 3 ** (1 - s)) / np.sqrt(3); Gcap = 3 ** (s - 1); r = 1 / s
    for p in plist:
        t0 = time.time()
        F = (p - 1) // 2
        G, y = prog_rw(p, s, 8 if F > 60 else 30)
        K = H2.kmat(p); cov = K @ y
        B = np.nonzero(cov < cov.min() * (1 + 1e-7))[0]
        L2 = np.sqrt((K[B] * y[None, :] ** 2).sum(1))
        ell = L2.max() / G; rho = np.sqrt((p - 1) / (p + 1))
        sigma = theta / (rho * ell)
        # best uniform eta at y*: max_eta G(eta) (1 - eta ell rho)  with G = min(Gcap, 1/(1-theta eta))
        etas = np.linspace(0, (1 - 1 / Gcap) / theta, 2001)
        Gs = np.minimum(Gcap, 1 / (1 - theta * etas))
        uni = np.max(Gs * (cov.min() - rho * etas * L2.max()) / G)
        print(json.dumps(dict(s=s, b=round(3 / s, 4), p=p, Kstar=int((y > 0).sum()), F=F, Gamma=G, K_eff=1 / ell ** 2, sigma=sigma,
                              uniform_eta_UB_at_ystar=uni, time=round(time.time() - t0, 1))), flush=True)
