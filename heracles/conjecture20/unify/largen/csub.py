# Subordination payment constant for slot-symmetric class-C single features:
#   c_sub(n,s,R) = sup over atom systems of  gain / max_{1<=k<=n} C(n,k) R^k |m_k|,   gain = Re A - nu^{1-s} Q^s,
#   m_k = sum_i beta_i xi_i^k.  The cone condition gives (Rogosinski on the diagonal slice, disc of radius R=cos(pi/p)):
#   max_k C(n,k) R^k |m_k| <= (4/p)|A-c|, so (Q1) holds in slot-symmetric class C whenever (4/p) c_sub(n,s,cos(pi/p)) < cos(pi/p).
# Smooth surrogate: max replaced by a p-norm (q large) during optimisation; exact max reported.
import sys, math, numpy as np, warnings
warnings.filterwarnings("ignore")
from scipy.optimize import minimize
from scipy.special import gammaln

def make(n, s, R, q=40.0):
    nu = n + 1; r = 1.0 / s; k = np.arange(n + 1)
    lw = gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1) + k * math.log(R)
    w = np.exp(lw); w[0] = 0.0
    def f(z, m, exact=False):
        rho, al, ph = z[:m], z[m:2*m], z[2*m:]
        rho = rho - rho.max()                                   # scale invariance
        b = np.exp(rho + 1j * al); Xk = np.exp(1j * np.outer(k, ph)); mk = Xk @ b
        A = mk[0]; lQ = np.log(np.sum(np.exp(r * rho)))
        IF = math.exp((1 - s) * math.log(nu) + s * lQ); num = A.real - IF
        a = w * np.abs(mk)                                     # detection coefficients
        if exact: return num / a.max(), num, a.max(), A, IF
        amax = a.max(); D = amax * (np.sum((a / amax) ** q)) ** (1 / q)
        return num / D
    return f

def search(n, s, R, m, starts, seed):
    f = make(n, s, R); rng = np.random.default_rng(seed); best = (-np.inf, None)
    for t in range(starts):
        if t % 2 == 0:
            ph = np.linspace(-1, 1, m) * rng.uniform(0.5, 3.0); rho = -0.5 * (np.linspace(-1, 1, m) * rng.uniform(0.5, 2.5)) ** 2
            al = ph * rng.uniform(0.0, 2.0)
        else:
            ph = rng.uniform(-np.pi, np.pi, m); rho = rng.uniform(-1, 0, m); al = rng.uniform(-np.pi, np.pi, m)
        z0 = np.concatenate([rho, al, ph])
        def fun(z):
            v = f(z, m)
            return -v if np.isfinite(v) else 1e6
        res = minimize(fun, z0, method='L-BFGS-B', options=dict(maxiter=3000))
        res = minimize(fun, res.x, method='Nelder-Mead', options=dict(maxiter=4000, xatol=1e-9, fatol=1e-12))
        v, num, D, A, IF = f(res.x, m, exact=True)
        if num > 0 and v > best[0]: best = (v, res.x.copy(), m)
    return best

if __name__ == "__main__":
    ns = [int(v) for v in sys.argv[1].split(',')]
    Rs = [float(v) for v in sys.argv[2].split(',')]
    starts = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    for n in ns:
        s = (n + 1) / 2.0
        for R in Rs:
            best = (-np.inf, None, None)
            for m in range(1, n + 3):
                v, z, mm = search(n, s, R, m, starts, seed=7 * n + m)
                if v > best[0]: best = (v, z, mm)
            expl = (1 - (n / (n + 1)) ** (s - 1)) / R ** n
            print(f"n={n} b=2 R={R:.4f}: c_sub >= {best[0]:.4f} (m={best[2]})   explicit family A(1+z1..zn): {expl:.4f}", flush=True)
