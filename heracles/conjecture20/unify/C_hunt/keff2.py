# RELAXATION of the class-C compensation problem (n = 2 unless stated), used as a map of the margin of safety.
# Per feature l: clean amplitude A_l = G_l * 3^{1-s} Q_l^s (gain G_l), relative L2 fluctuation eta_l.
# Payment (numerical constants, ptest2.py):  G_l <= min(Gcap, 1/(1 - theta*eta_l)).
# Detection (Lemma L2 for consistent families: rho = 1; numerically rho = sqrt((p-1)/(p+1)) for all class-C designs, detl2.py):
#   h <= min_delta [ sum_l Re(A_l) kappa_l(delta) - rho * (sum_l |a_l|^2 kappa_l(delta))^{1/2} ].
# Hence  gamma / D_S <= UB(p,s,S) := max_{y in r-ball(S), eta >= 0} min_delta [ sum G_l y_l kappa_l - rho (sum eta_l^2 G_l^2 y_l^2 kappa_l)^{1/2} ] / Gamma^{(S)}.
# UB >= 1 always (eta = 0, y = y*_S).  UB = 1 means: within the relaxation, no compensation pays (RD_S, conditional on theta, rho).
# Also: ell = max_{delta in B} (sum y*^2 kappa)^{1/2} / Gamma^{(S)} (K_eff = 1/ell^2) and the uniform-eta safety factor sigma = theta/(rho*ell).
# Usage: python keff2.py theta Gcap rho s p1,p2,... Smode1,Smode2,...   rho: number or 'auto' (= sqrt((p-1)/(p+1)))
#   Smode: F | Sstar | AP<K> | sub<d> (multiplicative subgroup of order d, d | p-1, d even) | rand<K>
import sys, json, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
from scipy.optimize import minimize
import hlib2 as H2

theta = float(sys.argv[1]); Gcap = float(sys.argv[2]); rho_arg = sys.argv[3]; s = float(sys.argv[4])
plist = [int(v) for v in sys.argv[5].split(',')]; Smodes = sys.argv[6].split(',')
r = 1 / s; eta_cap = (1 - 1 / Gcap) / theta
rng = np.random.default_rng(13)


def red(l, p):
    l %= p
    return min(l, p - l)


def prim_root(p):
    for g in range(2, p):
        if all(pow(g, (p - 1) // q, p) != 1 for q in set(f for f in range(2, p) if (p - 1) % f == 0 and all(f % d for d in range(2, f)))):
            return g


def make_S(p, mode):
    F = (p - 1) // 2
    if mode == 'F': return list(range(1, F + 1))
    if mode == 'Sstar': return None
    if mode.startswith('AP'): return list(range(1, min(F, int(mode[2:])) + 1))
    if mode.startswith('sub'):
        d = int(mode[3:])
        if (p - 1) % d or d % 2: return 'skip'
        g = prim_root(p); h = pow(g, (p - 1) // d, p)
        return sorted(set(red(pow(h, k, p), p) for k in range(d)))
    if mode.startswith('rand'):
        K = min(int(mode[4:]), F); return sorted(rng.choice(np.arange(1, F + 1), size=K, replace=False).tolist())
    raise SystemExit('bad mode')


def ub(p, S):
    F = (p - 1) // 2
    rho = np.sqrt((p - 1) / (p + 1)) if rho_arg == 'auto' else float(rho_arg)
    t0 = time.time()
    R = H2.prog_best(p, s, S=S, restarts=200 if len(S) < 60 else 40)
    G, ystar, bind = R['G'], R['y'], R['bind']
    Ks = H2.kmat(p)[:, [l - 1 for l in S]]
    ys = ystar[[l - 1 for l in S]]
    L2b = np.sqrt((Ks[[d - 1 for d in bind]] * ys[None, :] ** 2).sum(1))
    ell = L2b.max() / G
    k = len(S)

    def fvals(z):
        y = np.exp(z[:k]); sg = 1 / (1 + np.exp(-z[k:])); eta = eta_cap * sg
        Gl = 1 / (1 - theta * eta)
        u = Gl * y; e = eta * Gl * y
        c = Ks @ u; q = np.sqrt(Ks @ (e ** 2) + 1e-300)
        N = np.sum(y ** r) ** s
        return (c - rho * q) / N, (y, eta, Gl, sg, e, q, N)

    def obj(z, tau):
        f, (y, eta, Gl, sg, e, q, N) = fvals(z)
        fm = f.min(); w = np.exp(-(f - fm) / tau); lam = w / w.sum()
        val = fm - tau * np.log(w.sum())
        Ky = Ks * (Gl * y)[None, :] - rho * Ks * ((eta * Gl) ** 2 * y ** 2)[None, :] / q[:, None]
        sy = y ** r / np.sum(y ** r)
        gy = (lam @ Ky) / N - (lam @ f) * sy
        deta = eta_cap * sg * (1 - sg); dGde = theta * Gl ** 2
        Ke = Ks * (dGde * deta * y)[None, :] - rho * Ks * (e * y * (Gl * deta + eta * dGde * deta))[None, :] / q[:, None]
        ge = (lam @ Ke) / N
        return -val, -np.r_[gy, ge]

    best = (-np.inf, None)
    yb = np.maximum(ys, 1e-7 * ys.max())
    starts = []
    for et in (0.02, 0.2, 0.5, 0.9):
        lg = np.log(et / (1 - et))
        starts += [np.r_[np.log(yb), np.full(k, lg)], np.r_[np.zeros(k), np.full(k, lg)],
                   np.r_[np.log(yb + ys.max() * rng.uniform(0, 0.5, k)), rng.normal(lg, 1.5, k)]]
    for z0 in starts:
        z = z0.copy()
        for tau in (3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4):
            z = minimize(obj, z, args=(tau * G,), jac=True, method='L-BFGS-B', options={'maxiter': 3000}).x
        f, _ = fvals(z)
        if f.min() > best[0]: best = (f.min(), z.copy())
    f, (y, eta, Gl, sg, e, q, N) = fvals(best[1])
    yn = y / N; sig = yn > 1e-3 * yn.max()
    return dict(p=p, s=s, S=S if len(S) <= 24 else f'|S|={len(S)}', theta=round(theta, 6), Gcap=round(Gcap, 6), rho=round(rho, 6),
                Gamma=G, Gamma_gap=R['gap'], Kstar=len(R['supp']), ell=ell, K_eff=1 / ell ** 2,
                sigma=theta / (rho * ell), UB=best[0] / G, UB_feats=int(sig.sum()),
                UB_eta=[round(float(eta[sig].min()), 4), round(float(eta[sig].max()), 4)], time=round(time.time() - t0, 1))


for p in plist:
    for mode in Smodes:
        S = make_S(p, mode)
        if S == 'skip': continue
        if S is None:
            S = H2.prog_best(p, s, restarts=200)['supp']
        out = ub(p, S); out['Smode'] = mode
        print(json.dumps(out), flush=True)
