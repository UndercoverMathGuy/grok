# Slot-symmetric k*: atoms beta_i in C with a single phase xi_i = e^{i phi_i} used in every slot.
#   m_k = sum_i beta_i xi_i^k (k=0..n),  A = m_0,  E = sum_{k=1}^n C(n,k) |m_k|^2,
#   gain = Re A - nu^{1-s} (sum |beta_i|^{1/s})^s,  ratio = gain / sqrt(E/2).
# A lower bound for k*(n,s); the extremisers found by the general search (kstar_large.py) are slot-symmetric for n<=8.
# Usage: python kstar_sym.py n1,n2,.. mlo:mhi starts [b]
import sys, math, numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

def make(n, s):
    nu = n + 1; r = 1.0 / s
    k = np.arange(n + 1)
    lw = gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)
    w = np.exp(lw); w[0] = 0.0                       # weights C(n,k), k>=1
    def f(z, m, want_grad=True):
        rho, al, ph = z[:m], z[m:2*m], z[2*m:]
        b = np.exp(rho + 1j * al)
        Xk = np.exp(1j * np.outer(k, ph))           # (n+1) x m
        mk = Xk @ b                                  # moments
        A = mk[0]
        E = float(np.sum(w * np.abs(mk) ** 2))
        lQ = np.log(np.sum(np.exp(r * rho)))
        IF = math.exp((1 - s) * math.log(nu) + s * lQ)
        num = A.real - IF
        if E <= 0: return -np.inf, None, None
        sq = math.sqrt(E / 2); val = num / sq
        if not want_grad: return val, None, dict(A=A, E=E, IF=IF, num=num, L=np.abs(b).sum(), mk=mk)
        S = (w * np.conj(mk)) @ Xk                   # S_j = sum_k w_k conj(m_k) X_j^k
        Sk = ((k * w) * np.conj(mk)) @ Xk
        dE_rho = 2 * np.real(b * S); dE_al = -2 * np.imag(b * S); dE_ph = -2 * np.imag(b * Sk)
        dIF_rho = IF * np.exp(r * rho - lQ)
        dn_rho = np.real(b) - dIF_rho; dn_al = -np.imag(b)
        g = np.concatenate([(dn_rho - num * dE_rho / (2 * E)) / sq, (dn_al - num * dE_al / (2 * E)) / sq,
                            (- num * dE_ph / (2 * E)) / sq])
        return val, g, dict(A=A, E=E, IF=IF, num=num, L=np.abs(b).sum(), mk=mk)
    return f

def run(n, m, s, starts, seed, z0s=()):
    f = make(n, s); rng = np.random.default_rng(seed); best = (-np.inf, None, None)
    inits = list(z0s)
    for t in range(starts):
        # structured init: atoms on an arc, bell weights, phase linear in angle (as found for n<=8)
        if t % 3 == 0:
            ph = np.linspace(-1, 1, m) * rng.uniform(0.6, 1.6) + rng.uniform(-0.2, 0.2, m) * 0.1
            rho = -0.5 * (np.linspace(-1, 1, m) * rng.uniform(1, 2.5)) ** 2
            al = ph * rng.uniform(0.8, 2.0)
        elif t % 3 == 1:
            ph = rng.uniform(-np.pi, np.pi, m); rho = rng.uniform(-1, 0, m); al = rng.uniform(-np.pi, np.pi, m)
        else:
            ph = np.sort(rng.uniform(-1.5, 1.5, m)); rho = rng.uniform(-1.5, 0, m); al = ph * rng.uniform(0.5, 2.5)
        inits.append(np.concatenate([rho, al, ph]))
    for z0 in inits:
        def fun(z):
            v, g, _ = f(z, m)
            if not np.isfinite(v) or g is None: return 1e6, np.zeros_like(z)
            return -v, -g
        try:
            res = minimize(fun, z0, jac=True, method='L-BFGS-B', options=dict(maxiter=4000, gtol=1e-11))
        except Exception:
            continue
        v, _, info = f(res.x, m, False)
        if info is not None and info['num'] > 0 and v > best[0]:
            best = (v, res.x.copy(), info)
    return best

if __name__ == "__main__":
    ns = [int(v) for v in sys.argv[1].split(',')]
    mlo, mhi = [int(v) for v in sys.argv[2].split(':')]
    starts = int(sys.argv[3]); bval = float(sys.argv[4]) if len(sys.argv) > 4 else 2.0
    for n in ns:
        s = (n + 1) / bval
        res = {}; bestall = (-np.inf, None, None, None)
        for m in range(mlo, min(mhi, n + 1) + 1):
            v, z, info = run(n, m, s, starts, seed=97 * n + m)
            res[m] = v
            if v > bestall[0]: bestall = (v, z, info, m)
        v, z, info, m = bestall
        line = f"n={n} b={bval} s={s:.2f} k*_sym={v:.4f} (m={m}) | " + " ".join(f"{mm}:{res[mm]:.3f}" for mm in res)
        if info is not None:
            rho = z[:m]; mod = np.exp(rho); L = mod.sum()
            line += f" | |A|/L={abs(info['A'])/L:.3f} IF/L={info['IF']/L:.4f} E/|A|^2={info['E']/abs(info['A'])**2:.3f}"
        print(line, flush=True)
