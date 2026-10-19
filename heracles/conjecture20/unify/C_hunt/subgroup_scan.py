# Where is the L2 relaxation weakest?  Flat classes S = multiplicative subgroups H of Z_p^* (mod +-1): every feature of S
# is equivalent under multiplication by H, so the restricted programme may keep many features (large K_eff).
# Stage 1 (cheap, Gauss periods): uniform allocation on S:  Gamma_unif = |S|^{-s} (|S| - eta_max),  eta_max = max_delta sum_{l in S} cos(2 pi l delta/p);
#   K_eff(unif) = |S| - eta_max,  sigma_unif = theta(s) sqrt(K_eff) / rho_p.
# Stage 2 (candidates with sigma_unif >= thr): the exact-ish restricted programme Gamma^{(S)} (reweighted LP + support local
#   search), K*_S, sigma at y*_S, and the full relaxation maximum UB(p, s, S) (keff2 objective, L-BFGS, many starts).
# Usage: python subgroup_scan.py s pmin pmax thr [dmin dmax]
import sys, json, time, warnings
import numpy as np
from scipy.optimize import minimize
import hlib2 as H2
warnings.filterwarnings('ignore')
s = float(sys.argv[1]); pmin = int(sys.argv[2]); pmax = int(sys.argv[3]); thr = float(sys.argv[4])
dmin = int(sys.argv[5]) if len(sys.argv) > 5 else 10; dmax = int(sys.argv[6]) if len(sys.argv) > 6 else 80
theta = (1 - 3 ** (1 - s)) / np.sqrt(3); Gcap = 3 ** (s - 1); r = 1 / s; eta_cap = (1 - 1 / Gcap) / theta
rng = np.random.default_rng(5)


def primes(a, b):
    return [p for p in range(max(a, 5), b + 1) if all(p % q for q in range(2, int(p ** 0.5) + 1))]


def prim_root(p):
    fac = [q for q in range(2, p) if (p - 1) % q == 0 and all(q % d for d in range(2, int(q ** 0.5) + 1))]
    for g in range(2, p):
        if all(pow(g, (p - 1) // q, p) != 1 for q in fac): return g


def red(l, p):
    l %= p
    return min(l, p - l)


def ub_opt(p, S, G, ys):
    Ks = H2.kmat(p)[:, [l - 1 for l in S]]; k = len(S); rho = np.sqrt((p - 1) / (p + 1))

    def fvals(z):
        y = np.exp(z[:k]); sg = 1 / (1 + np.exp(-z[k:])); eta = eta_cap * sg
        Gl = 1 / (1 - theta * eta); e = eta * Gl * y
        c = Ks @ (Gl * y); q = np.sqrt(Ks @ (e ** 2) + 1e-300); N = np.sum(y ** r) ** s
        return (c - rho * q) / N, (y, eta, Gl, sg, e, q, N)

    def obj(z, tau):
        f, (y, eta, Gl, sg, e, q, N) = fvals(z)
        fm = f.min(); w = np.exp(-(f - fm) / tau); lam = w / w.sum(); val = fm - tau * np.log(w.sum())
        Ky = Ks * (Gl * y)[None, :] - rho * Ks * ((eta * Gl) ** 2 * y ** 2)[None, :] / q[:, None]
        gy = (lam @ Ky) / N - (lam @ f) * (y ** r / np.sum(y ** r))
        deta = eta_cap * sg * (1 - sg); dGde = theta * Gl ** 2
        Ke = Ks * (dGde * deta * y)[None, :] - rho * Ks * (e * y * (Gl * deta + eta * dGde * deta))[None, :] / q[:, None]
        return -val, -np.r_[gy, (lam @ Ke) / N]

    best = -np.inf; yb = np.maximum(ys, 1e-7 * ys.max())
    for et in (0.05, 0.3, 0.7, 0.95):
        lg = np.log(et / (1 - et))
        for z0 in (np.r_[np.log(yb), np.full(k, lg)], np.r_[np.zeros(k), np.full(k, lg)], np.r_[np.log(yb) + rng.normal(0, 0.5, k), rng.normal(lg, 1, k)]):
            z = z0.copy()
            for tau in (3e-2, 1e-2, 3e-3, 1e-3, 3e-4):
                z = minimize(obj, z, args=(tau * G,), jac=True, method='L-BFGS-B', options={'maxiter': 2000}).x
            best = max(best, fvals(z)[0].min())
    return best / G


t0 = time.time(); ncand = 0; maxsig = (0, None)
for p in primes(pmin, pmax):
    rho = np.sqrt((p - 1) / (p + 1)); g = None
    for d in range(dmin, dmax + 1, 2):
        if (p - 1) % d: continue
        if g is None: g = prim_root(p)
        h = pow(g, (p - 1) // d, p)
        S = sorted({red(pow(h, k, p), p) for k in range(d)})
        dl = np.arange(1, (p - 1) // 2 + 1)
        csum = np.cos(2 * np.pi * np.outer(dl, S) / p).sum(1)
        Keu = len(S) - csum.max()
        sig_u = theta * np.sqrt(Keu) / rho
        if sig_u > maxsig[0]: maxsig = (sig_u, (p, d))
        if sig_u < thr: continue
        ncand += 1
        R = H2.prog_bb(p, s, S=S, maxnodes=3000 if len(S) <= 12 else 200, restarts=40)
        G = R['G']; y = R['y']; ys = y[[l - 1 for l in S]]
        if len(S) ** (-s) * Keu > G:           # the uniform allocation is better than the incumbent: use it
            G = len(S) ** (-s) * Keu; ys = np.full(len(S), len(S) ** (-s)); R = dict(R, supp=S)
        K = H2.kmat(p)[:, [l - 1 for l in S]]; cov = K @ ys
        B = np.nonzero(cov < G * (1 + 1e-7))[0]
        ell = np.sqrt((K[B] * ys[None, :] ** 2).sum(1)).max() / G
        Gu = len(S) ** (-s) * Keu
        UB = ub_opt(p, S, G, ys)
        print(json.dumps(dict(s=s, p=p, d=d, S_size=len(S), Gamma_S=G, Gamma_unif=Gu, unif_efficiency=Gu / G, KstarS=len(R['supp']),
                              K_eff_unif=Keu, sigma_unif=sig_u, K_eff_opt=1 / ell ** 2, sigma_opt=theta / (rho * ell), UB=UB,
                              t=round(time.time() - t0))), flush=True)
print(f"# done s={s} p in [{pmin},{pmax}] candidates={ncand} max sigma_unif={maxsig[0]:.3f} at (p,d)={maxsig[1]} time={time.time()-t0:.0f}s", flush=True)
