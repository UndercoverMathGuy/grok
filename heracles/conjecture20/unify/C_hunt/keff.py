# Consistent-family (e.g. label-only) networks at LARGE p: how close can they get to D, given Lemma L2 and an L2 payment
# constant theta (G <= 1/(1 - theta*eta), G <= Gcap)?
#   UB(p,s,S) = max_{y in r-ball(S), eta_l in [0, eta_cap]} min_delta [ sum G_l y_l kappa_l - (sum eta_l^2 G_l^2 y_l^2 kappa_l)^{1/2} ] / Gamma^{(S)}
# (n = 2: D = 3^{1-s} B_p Gamma^{(S)}; any n with pi replaced by nu^{1-s} gives an upper bound on gamma/gamma_IF).
# Also the one-line statistic at the programme optimum y*:  ell = max_{delta in B} (sum y*^2 kappa)^{1/2} / Gamma,
#   K_eff = 1/ell^2;  uniform-eta family is safe iff theta <= ell.
# Usage: python keff.py theta Gcap s p1,p2,... [Smode]     Smode: F (default) | AP<K> (S = {1..K}) | rand<K>
import sys, json, time
import numpy as np
from scipy.optimize import minimize
import hlib2 as H2

theta = float(sys.argv[1]); Gcap = float(sys.argv[2]); s = float(sys.argv[3])
plist = [int(v) for v in sys.argv[4].split(',')]; Smode = sys.argv[5] if len(sys.argv) > 5 else 'F'
r = 1 / s; eta_cap = (1 - 1 / Gcap) / theta
rng = np.random.default_rng(11)


def run(p):
    F = (p - 1) // 2
    if Smode == 'F': S = list(range(1, F + 1))
    elif Smode.startswith('AP'): S = list(range(1, min(F, int(Smode[2:])) + 1))
    else:
        K = int(Smode[4:]); S = sorted(rng.choice(np.arange(1, F + 1), size=min(K, F), replace=False).tolist())
    t0 = time.time()
    R = H2.prog_bb(p, s, S=S, maxnodes=1500 if F > 12 else 100000, restarts=40 if F < 60 else 12)
    G, ystar, bind = R['G'], R['y'], R['bind']
    Km = H2.kmat(p)                                   # rows delta, cols l
    Ks = Km[:, [l - 1 for l in S]]
    ys = ystar[[l - 1 for l in S]]
    L2b = np.sqrt((Ks[[d - 1 for d in bind]] * ys[None, :] ** 2).sum(1))
    ell = L2b.max() / G
    # ---------------- UB optimisation
    k = len(S)

    def parts(z):
        y = np.exp(z[:k]); sg = 1 / (1 + np.exp(-z[k:])); eta = eta_cap * sg
        Gl = 1 / (1 - theta * eta); dG = theta * Gl ** 2 * eta_cap * sg * (1 - sg)     # dG/dz_eta
        return y, eta, Gl, dG, sg

    def fvals(z):
        y, eta, Gl, dG, sg = parts(z)
        u = Gl * y; e = eta * Gl * y
        c = Ks @ u; q = np.sqrt(Ks @ (e ** 2) + 1e-300)
        N = np.sum(y ** r) ** s
        return (c - q) / N, (y, eta, Gl, dG, sg, u, e, c, q, N)

    def obj(z, tau):
        f, (y, eta, Gl, dG, sg, u, e, c, q, N) = fvals(z)
        fm = f.min(); w = np.exp(-(f - fm) / tau); Z = w.sum(); lam = w / Z
        val = fm - tau * np.log(Z)
        # df_delta/dz_y = y * [G kappa - eta^2 G^2 y kappa / q] / N - f * (y^r / sum y^r)
        Ky = Ks * (Gl * y)[None, :] - Ks * ((eta * Gl) ** 2 * y ** 2)[None, :] / q[:, None]
        sy = y ** r / np.sum(y ** r)
        gy = (lam @ Ky) / N - (lam @ f) * sy
        # df/dz_eta = [dG y kappa - (e * y * (G' * eta + G) ... ) / q] / N, with d(eta)/dz = eta_cap sg (1-sg)
        deta = eta_cap * sg * (1 - sg)
        dGde = theta * Gl ** 2
        Ke = Ks * (dGde * deta * y)[None, :] - Ks * (e * y * (Gl * deta + eta * dGde * deta))[None, :] / q[:, None]
        ge = (lam @ Ke) / N
        return -val, -np.r_[gy, ge]

    best = (-np.inf, None)
    starts = []
    yb = np.maximum(ys, 1e-6 * ys.max())
    for et in (0.05, 0.3, 0.7, 0.95):
        starts.append(np.r_[np.log(yb), np.full(k, np.log(et / (1 - et)))])
        starts.append(np.r_[np.zeros(k), np.full(k, np.log(et / (1 - et)))])
        starts.append(np.r_[np.log(yb + ys.max() * rng.uniform(0, 0.5, k)), rng.normal(0, 1.5, k)])
    for z0 in starts:
        z = z0.copy()
        for tau in (3e-2, 1e-2, 3e-3, 1e-3, 3e-4):
            rres = minimize(obj, z, args=(tau * G,), jac=True, method='L-BFGS-B', options={'maxiter': 2000})
            z = rres.x
        f, info = fvals(z)
        if f.min() > best[0]: best = (f.min(), z.copy())
    f, (y, eta, Gl, dG, sg, u, e, c, q, N) = fvals(best[1])
    yn = y / N
    sig = yn > 1e-3 * yn.max()
    out = dict(p=p, s=s, Smode=Smode, theta=theta, Gcap=Gcap, Gamma=G, Gamma_gap=R['gap'], Kstar=len(R['supp']),
               ell=ell, K_eff=1 / ell ** 2, uniform_eta_safe=bool(theta <= ell),
               UB=best[0] / G, UB_features=int(sig.sum()), UB_eta_range=[float(eta[sig].min()), float(eta[sig].max())],
               time=round(time.time() - t0, 1))
    print(json.dumps(out), flush=True)


for p in plist: run(p)
