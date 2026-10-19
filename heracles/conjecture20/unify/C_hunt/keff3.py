# Relaxation maximum UB(p, s, F) for LARGER p (class-C IF value at n = 2), with the programme value improved on the fly:
# Gamma := max(stored b=2 optimum (relations3/*.json), reweighted-LP incumbent, best eta=0 coverage found by the UB optimiser).
# If the UB optimiser's best point has eta = 0 and beats Gamma, Gamma is raised and the optimisation repeated, so UB > 1 can only
# survive with eta > 0 (a genuinely profitable cheapening inside the relaxation).
# Usage: python keff3.py s p1,p2,...
import sys, json, glob, time, warnings
import numpy as np
from scipy.optimize import minimize, linprog
import hlib2 as H2
warnings.filterwarnings('ignore')
s = float(sys.argv[1]); plist = [int(v) for v in sys.argv[2].split(',')]
r = 1 / s; theta = (1 - 3 ** (1 - s)) / np.sqrt(3); Gcap = 3 ** (s - 1); eta_cap = (1 - 1 / Gcap) / theta
rng = np.random.default_rng(9)
D = '/home/user/grok/heracles/conjecture20/mechanism_scripts/relations3/'
stored = {}
for f in [D + 'supports.json'] + glob.glob(D + 'large_*.json'):
    for k, v in json.load(open(f)).items():
        if k.isdigit() and isinstance(v, dict) and 'y' in v: stored[int(k)] = v


def rwlp(p, restarts):
    K = H2.kmat(p); F = K.shape[1]; best = (np.inf, None)
    for t in range(restarts):
        w = np.ones(F) if t == 0 else rng.uniform(0.05, 2.0, F)
        for it in range(150):
            res = linprog(w, A_ub=-K, b_ub=-np.ones(F), bounds=(0, None), method='highs')
            v = res.x; wn = (v + 1e-12) ** (r - 1)
            if np.allclose(wn, w, rtol=1e-10): break
            w = wn
        v = v * (v > 1e-11); Q = np.sum(v[v > 0] ** r)
        if Q < best[0]: best = (Q, v.copy())
    return best[0] ** (-s), best[1] / best[0] ** s


for p in plist:
    t0 = time.time(); F = (p - 1) // 2; K = H2.kmat(p); rho = np.sqrt((p - 1) / (p + 1))
    G, y = rwlp(p, 60 if F < 40 else 25)
    if abs(s - 1.5) < 1e-12 and p in stored:
        ys = np.zeros(F); ys[np.array(stored[p]['S']) - 1] = stored[p]['y']
        Gs = (K @ ys).min() / np.sum(ys[ys > 0] ** r) ** s
        if Gs > G: G, y = Gs, ys / np.sum(ys[ys > 0] ** r) ** s

    def fvals(z):
        yy = np.exp(z[:F]); sg = 1 / (1 + np.exp(-z[F:])); eta = eta_cap * sg
        Gl = 1 / (1 - theta * eta); e = eta * Gl * yy
        c = K @ (Gl * yy); q = np.sqrt(K @ (e ** 2) + 1e-300); N = np.sum(yy ** r) ** s
        return (c - rho * q) / N, (yy, eta, Gl, sg, e, q, N)

    def obj(z, tau):
        f, (yy, eta, Gl, sg, e, q, N) = fvals(z)
        fm = f.min(); w = np.exp(-(f - fm) / tau); lam = w / w.sum(); val = fm - tau * np.log(w.sum())
        Ky = K * (Gl * yy)[None, :] - rho * K * ((eta * Gl) ** 2 * yy ** 2)[None, :] / q[:, None]
        gy = (lam @ Ky) / N - (lam @ f) * (yy ** r / np.sum(yy ** r))
        deta = eta_cap * sg * (1 - sg); dGde = theta * Gl ** 2
        Ke = K * (dGde * deta * yy)[None, :] - rho * K * (e * yy * (Gl * deta + eta * dGde * deta))[None, :] / q[:, None]
        return -val, -np.r_[gy, (lam @ Ke) / N]

    for rnd in range(3):
        best = (-np.inf, None); yb = np.maximum(y, 1e-7 * y.max())
        starts = []
        for et in (0.02, 0.3, 0.7, 0.95):
            lg = np.log(et / (1 - et))
            starts += [np.r_[np.log(yb), np.full(F, lg)], np.r_[np.log(yb) + rng.normal(0, 0.7, F), rng.normal(lg, 1.0, F)],
                       np.r_[np.zeros(F), np.full(F, lg)]]
        for z0 in starts:
            z = z0.copy()
            for tau in (3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4):
                z = minimize(obj, z, args=(tau * G,), jac=True, method='L-BFGS-B', options={'maxiter': 1500}).x
            f, _ = fvals(z)
            if f.min() > best[0]: best = (f.min(), z.copy())
        f, (yy, eta, Gl, sg, e, q, N) = fvals(best[1])
        sig = yy / N > 1e-3 * (yy / N).max(); etamax = float(eta[sig].max())
        # eta = 0 coverage of the best point's allocation: an admissible programme value
        cov0 = (K @ (yy / N)).min()
        if best[0] / G > 1 + 1e-9 and cov0 > G * (1 + 1e-12):
            G, y = cov0, yy / N; continue
        break
    print(json.dumps(dict(p=p, s=s, Gamma=G, UB=best[0] / G, eta_max_at_UB=etamax, UB_feats=int(sig.sum()), rounds=rnd + 1,
                          time=round(time.time() - t0))), flush=True)
