# Arc-initialised polish for slot-symmetric k*: exact arc solution beta = G^{-1} 1 (mpmath), then L-BFGS (float64)
# on all parameters (log|beta|, arg beta, phi), then exact re-evaluation in mpmath (60+n digits).
import sys, math, numpy as np, mpmath as mp, warnings
warnings.filterwarnings("ignore")
from scipy.optimize import minimize
sys.path.insert(0, '.')
from kstar_exact import eval_arc, G_matrix, ratio_full
from kstar_sym import make

def to_z(d):
    beta = [complex(b) for b in d['beta']]; phis = [float(x) for x in d['phis']]
    rho = np.log(np.abs(beta)); al = np.angle(beta)
    return np.concatenate([rho, al, phis])

def exact_ratio(z, m, n, s, dps):
    mp.mp.dps = dps
    rho, al, ph = z[:m], z[m:2*m], z[2*m:]
    beta = [mp.mpf(float(np.exp(r))) * mp.expj(mp.mpf(float(a))) for r, a in zip(rho, al)]
    phis = [mp.mpf(float(x)) for x in ph]
    G, _ = G_matrix(phis, n)
    r, A, E, IF = ratio_full(beta, G, n, s)
    return r, A, E, IF, beta, phis

if __name__ == "__main__":
    for n in [int(v) for v in sys.argv[1].split(',')]:
        s = (n + 1) / 2.0
        cand = []
        for f in [0.72, 0.76, 0.80, 0.84]:
            m = int(round(f * n))
            for c in [1.15, 1.2, 1.25, 1.3, 1.35]:
                d = eval_arc(n, m, 2 * math.pi / (c * n + 2), s, dps=40 + n)
                cand.append((float(d['ratio']), m, c, d))
        cand.sort(key=lambda t: -t[0])
        out = []
        for r0, m, c, d in cand[:4]:
            f = make(n, s)
            z0 = to_z(d)
            def fun(zz):
                v, g, _ = f(zz, m)
                if not np.isfinite(v) or g is None: return 1e6, np.zeros_like(zz)
                return -v, -g
            res = minimize(fun, z0, jac=True, method='L-BFGS-B', options=dict(maxiter=20000, gtol=1e-12, maxcor=40))
            r, A, E, IF, beta, phis = exact_ratio(res.x, m, n, s, 60 + n)
            out.append((float(r), r0, m, c, res.x, A, E, IF))
        out.sort(key=lambda t: -t[0])
        r, r0, m, c, z, A, E, IF = out[0]
        L = float(np.exp(z[:m]).sum()) / abs(complex(A))
        print(f"n={n} s={s}: exact ratio after polish {r:.5f} (arc start {r0:.5f}, m={m}, c={c})  IF/A={float(IF/abs(A)):.3f}"
              f"  E/|A|^2={float(E/abs(A)**2):.4f}  L/|A|={L:.2f}", flush=True)
        np.save(f"arcpolish_n{n}.npy", np.concatenate([[m], z]))
