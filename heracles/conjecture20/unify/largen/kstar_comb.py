# Large-n lower bounds for the slot-symmetric k*(n,s) via a structured "Gaussian comb" family, then a full polish.
#   phi_i = (i-c) Delta,  |beta_i| = exp(-(i-c)^2/(2 sig^2)),  arg beta_i = kap (i-c) + q (i-c)^2,  i=0..m-1, c=(m-1)/2.
# Every value printed is the exact ratio (Re A - nu^{1-s}Q^s)/sqrt(E/2) of an explicit atom system (a valid lower bound
# on k*(n,s)); the final atom systems are saved and re-checked in exact rational/mpmath arithmetic by kstar_check.py.
import sys, math, numpy as np, warnings
warnings.filterwarnings("ignore")
from scipy.optimize import minimize
sys.path.insert(0, '.')
from kstar_sym import make

def family(m, D, sig, kap, q):
    i = np.arange(m) - (m - 1) / 2
    rho = -(i ** 2) / (2 * sig ** 2); al = kap * i + q * i ** 2; ph = D * i
    return np.concatenate([rho, al, ph])

def best_family(n, s, m, rng, tries=30):
    f = make(n, s)
    def neg(par):
        D, sig, kap, q = par
        if D <= 0 or sig <= 0.2: return 1e6
        v, _, info = f(family(m, D, sig, kap, q), m, False)
        return -v if (info is not None and np.isfinite(v) and info['num'] > 0) else 1e6
    best = (np.inf, None)
    for t in range(tries):
        x0 = [rng.uniform(2.0, 3.6) / (1.45 * n + 1), rng.uniform(1.2, 3.5), rng.uniform(0.3, 1.2), rng.uniform(-0.02, 0.02)]
        r = minimize(neg, x0, method='Nelder-Mead', options=dict(maxiter=3000, xatol=1e-7, fatol=1e-10))
        if r.fun < best[0]: best = (r.fun, r.x)
    return -best[0], best[1]

def polish(n, s, z, m):
    f = make(n, s)
    def fun(zz):
        v, g, _ = f(zz, m)
        if not np.isfinite(v) or g is None: return 1e6, np.zeros_like(zz)
        return -v, -g
    r = minimize(fun, z, jac=True, method='L-BFGS-B', options=dict(maxiter=20000, gtol=1e-12, maxcor=30))
    v, _, info = f(r.x, m, False)
    return (v if info is not None and info['num'] > 0 else -np.inf), r.x, info

if __name__ == "__main__":
    ns = [int(v) for v in sys.argv[1].split(',')]
    bval = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
    fr = [float(v) for v in sys.argv[3].split(',')] if len(sys.argv) > 3 else [0.68, 0.72, 0.76, 0.80]
    rng = np.random.default_rng(7)
    for n in ns:
        s = (n + 1) / bval
        out = []
        for a in fr:
            m = max(2, int(round(a * n)))
            vfam, par = best_family(n, s, m, rng)
            if par is None or not np.isfinite(vfam): continue
            v, z, info = polish(n, s, family(m, *par), m)
            out.append((v, vfam, m, par, z, info))
        out.sort(key=lambda t: -t[0])
        v, vfam, m, par, z, info = out[0]
        L = np.exp(z[:m]).sum()
        print(f"n={n} b={bval} s={s:.2f}: k*_sym >= {v:.4f} (m={m}; family {vfam:.4f}, D={par[0]:.4f} sig={par[1]:.2f} kap={par[2]:.3f})"
              f"  |A|/L={abs(info['A'])/L:.3f} IF/L={info['IF']/L:.2e} E/|A|^2={info['E']/abs(info['A'])**2:.3f} | others "
              + " ".join(f"{t[2]}:{t[0]:.3f}" for t in out[1:]), flush=True)
        np.save(f"comb_opt_n{n}_b{bval}.npy", np.concatenate([[m], z]))
