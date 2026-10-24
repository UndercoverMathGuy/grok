import numpy as np, itertools, mpmath as mp
from scipy.optimize import minimize
from vlib import *
rng = np.random.default_rng(404)
mp.mp.dps = 40
u2 = (1 + mp.sqrt(1 - 2 / mp.sqrt(5))) / 2; u1 = (1 - mp.sqrt(1 - 2 / mp.sqrt(5))) / 2
print("elem lemma: u1, u2, F(u2) =", mp.nstr(u1, 12), mp.nstr(u2, 12), mp.nstr(-mp.log(1 - u2) - mp.sqrt(5) * u2 ** 2, 12))
print("k_5 - sqrt5 =", mp.nstr(1 / mp.cos(2 * mp.pi / 5) - 1 - mp.sqrt(5), 5))

def build(p, n, z, m, dc_on):
    a = np.arange(p)
    amp = np.exp(z[:m * n].reshape(m, n)); ph = z[m * n:2 * m * n].reshape(m, n)
    wamp = np.exp(z[2 * m * n:2 * m * n + m])
    dc = z[2 * m * n + m:2 * m * n + m + m * n].reshape(m, n) * dc_on
    wdc = z[2 * m * n + m + m * n:] * dc_on
    U = amp[:, :, None] * np.cos(2 * np.pi * a[None, None, :] / p + ph[:, :, None]) + dc[:, :, None]
    W = wamp[:, None] * np.cos(2 * np.pi * a[None, :] / p + ph.sum(1)[:, None]) + wdc[:, None]   # aligned atoms
    return U, W

def analyse(U, W, s):
    m, n, p = U.shape; nu = n + 1
    f = table(U, W); g = gaps(f, p, n); h = g.min()
    Phi = fields(f, p, n)[0]
    A = Phi.mean(); th = np.pi / p
    cm = Phi.real - np.abs(Phi.imag) / np.tan(th); c = cm.min()
    circ = abs(np.mean((Phi - A) ** 2)) / np.mean(np.abs(Phi - A) ** 2 + 1e-300)
    beta, xi = atoms(U, W, 1)
    L = np.abs(beta).sum(); al = 1 - abs(beta.sum()) / L
    kmin = 1 - np.cos(2 * np.pi / p); b = nu / s
    gam = h / np.sum(sqnorm(U, W) ** (b / 2)) ** s; gif = nu ** (1 - s) * Bp(p, n) * kmin
    sumT = sum(abs(coef(Phi, tuple((2 * T[t]) % p for t in range(n)), p, n)) ** 2 for T in itertools.product([0, 1], repeat=n) if any(T))
    kp = 1 / np.cos(2 * th) - 1
    x = 1 - c / abs(A)
    return gam / gif, h, c, abs(A), x, sumT / abs(A) ** 2, kp, circ, al

for (p, n, b, m, dc_on) in [(5, 3, 2.0, 4, 1), (5, 3, 2.0, 5, 1), (5, 4, 2.0, 5, 1), (7, 3, 2.0, 4, 1), (5, 2, 2.0, 3, 1), (7, 2, 2.0, 3, 1)]:
    nu = n + 1; s = nu / b; kmin = 1 - np.cos(2 * np.pi / p); gif = nu ** (1 - s) * Bp(p, n) * kmin
    dim = 2 * m * n + m + m * n + m
    def neg(z, tau):
        U, W = build(p, n, z, m, dc_on)
        g = gaps(table(U, W), p, n).ravel() / np.sum(sqnorm(U, W) ** (b / 2)) ** s / gif
        gm = g.min()
        return -(gm - tau * np.log(np.mean(np.exp(-(g - gm) / tau))))
    best = (-np.inf, None)
    for seed in range(5):
        r0 = np.random.default_rng(seed + 10 * m)
        z = np.concatenate([r0.normal(scale=0.2, size=m * n), r0.uniform(0, 2 * np.pi, m * n), r0.normal(scale=0.2, size=m),
                            0.2 * r0.normal(size=m * n), 0.2 * r0.normal(size=m)])
        for tau in [0.1, 0.03, 0.01, 0.003]:
            z = minimize(neg, z, args=(tau,), method='L-BFGS-B', options={'maxiter': 500}).x
        U, W = build(p, n, z, m, dc_on)
        res = analyse(U, W, s)
        if res[0] > best[0]: best = (res[0], res)
    ratio, h, c, A, x, sT, kp, circ, al = best[1]
    bound = (1 - x) * (1 + 2 * kp * x * x / nu) ** (s - 1)
    print(f"(p,n,b,m)=({p},{n},{b},{m}) aligned+DC: best gamma/gIF={ratio:.6f}; x={x:.4f}; (i) sumT/|A|^2={sT:.4f} <= k x^2={kp*x*x:.4f}; (iii) bound={bound:.6f}; circ={circ:.1e}; 1-|A|/L={al:.1e}")
