# Independent adversarial search for R12's hypothesis at (p,n)=(5,2):
#   W_{1}(s) := sup{ gamma(theta) : beta_2(theta) = 0 }  vs  needed  < Gamma_s/kappa_min * 3^{1-s} B_5 kappa_min = 3^{1-s} B_5 Gamma_s.
# Arbitrary inputs/outputs (all frequencies, DC); beta_2 forced to 0 by an increasing penalty; own gradient code.
# Reports ratio R = gamma / (3^{1-s} B_5 kappa_min); hypothesis needs R < Gamma_s/kappa_min (1.2792 at b=2).
import sys, math, numpy as np
from scipy.optimize import minimize
from vnet import B_p, table, gaps, cost, clean, if_block, freq_basis

p, n = 5, 2; nu = 3
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
b = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
s = nu / b
rng = np.random.default_rng(seed)
idx = np.indices((p, p)); Y = (idx[0] + idx[1]) % p
a = np.arange(p); rho = np.exp(2j * np.pi / p)
km = 1 - math.cos(2 * math.pi / p)
ref = 3 ** (1 - s) * B_p(p, n) * km
need = 2.5 * 2 ** (-s) / km


def obj(z, m, tau, mu):
    U = z[:m * n * p].reshape(m, n, p); W = z[m * n * p:].reshape(m, p)
    f = np.einsum('ia,ib,ic->abc', U[:, 0], U[:, 1], W)
    G = gaps(f)                                  # (p,p,p-1)
    P = cost(U, W, b); c = P ** (-s)
    g = G * c / ref
    gmin = g.min(); e = np.exp(-(g - gmin) / tau); Z = e.sum(); lam = e / Z
    val = gmin - tau * math.log(Z)
    # dval/df: Lambda(x,c)
    Lam = np.zeros((p, p, p))
    for d in range(1, p):
        l = lam[..., d - 1]
        np.add.at(Lam, (idx[0], idx[1], Y), l)
        np.add.at(Lam, (idx[0], idx[1], (Y - d) % p), -l)
    Lam *= c / ref
    gW = np.einsum('abc,ia,ib->ic', Lam, U[:, 0], U[:, 1])
    gU0 = np.einsum('abc,ib,ic->ia', Lam, U[:, 1], W)
    gU1 = np.einsum('abc,ia,ic->ib', Lam, U[:, 0], W)
    Eg = (lam * g).sum()
    S = (U ** 2).sum((1, 2)) + (W ** 2).sum(1); dP = b * S ** (b / 2 - 1)
    gU = np.stack([gU0, gU1], 1) - s * Eg / P * dP[:, None, None] * U
    gW = gW - s * Eg / P * dP[:, None] * W
    # penalty mu |beta_2| c / ref
    Uh = np.fft.fft(U, axis=2); Wh = np.fft.fft(W, axis=1); j = 2
    pr = Uh[:, 0, j] * Uh[:, 1, j]
    beta = 2 * p ** -3 * np.sum(np.conj(Wh[:, j]) * pr); mag = abs(beta) + 1e-300; ph = np.conj(beta) / mag
    pen = mag * c / ref
    dW = np.real(ph * 2 * p ** -3 * pr[:, None] * rho ** (j * a)[None, :])
    dU0 = np.real(ph * 2 * p ** -3 * (np.conj(Wh[:, j]) * Uh[:, 1, j])[:, None] * rho ** (-j * a)[None, :])
    dU1 = np.real(ph * 2 * p ** -3 * (np.conj(Wh[:, j]) * Uh[:, 0, j])[:, None] * rho ** (-j * a)[None, :])
    gpU = np.stack([dU0, dU1], 1) * c / ref - s * pen / P * dP[:, None, None] * U
    gpW = dW * c / ref - s * pen / P * dP[:, None] * W
    tot = val - mu * pen
    grad = np.concatenate([(gU - mu * gpU).ravel(), (gW - mu * gpW).ravel()])
    return -tot, -grad


def run(m, kind):
    if kind == 'gauss':
        U = rng.normal(size=(m, n, p)); W = rng.normal(size=(m, p))
    elif kind == 'if':
        U0, W0 = if_block(p, n, 1); k = min(m, 3)
        U = 0.4 * rng.normal(size=(m, n, p)); W = 0.4 * rng.normal(size=(m, p)); U[:k] += U0[:k]; W[:k] += W0[:k]
    elif kind == 'sign':
        U = np.sign(rng.normal(size=(m, n, p))) * rng.uniform(0.5, 1.5, size=(m, n, p)); W = rng.normal(size=(m, p))
    elif kind == 'dcpartner':
        B = np.vstack([freq_basis(p, 0) * 2, freq_basis(p, 1), freq_basis(p, 2)])
        U = rng.normal(size=(m, n, 5)) @ B; W = rng.normal(size=(m, 5)) @ B
    z = np.concatenate([U.ravel(), W.ravel()])
    for tau, mu in [(3e-2, 0.3), (1e-2, 1), (3e-3, 3), (1e-3, 10), (3e-4, 30), (1e-4, 100), (3e-5, 300), (1e-5, 1000)]:
        res = minimize(obj, z, args=(m, tau, mu), jac=True, method='L-BFGS-B', options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-12})
        z = res.x
    U = z[:m * n * p].reshape(m, n, p); W = z[m * n * p:].reshape(m, p)
    f = table(U, W); h = gaps(f).min(); P = cost(U, W, b)
    cl = np.abs(clean(f))
    return h / P ** s / ref, cl[1] / max(cl.max(), 1e-300), cl


best = []
for st in range(int(sys.argv[3]) if len(sys.argv) > 3 else 16):
    m = int(rng.integers(3, 10)); kind = ['gauss', 'if', 'sign', 'dcpartner'][st % 4]
    R, off, cl = run(m, kind)
    best.append((R, off, m, kind))
    print(f"b={b} start {st} m={m} {kind:9s}: R={R:.6f}  |beta_2|/max|beta|={off:.2e}", flush=True)
ok = [x for x in best if x[1] < 1e-4]
print(f"b={b}: best R with |beta_2|/max < 1e-4: {max(ok)[0] if ok else None:}; hypothesis needs R < {need:.4f}")
