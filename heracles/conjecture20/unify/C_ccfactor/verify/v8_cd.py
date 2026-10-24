# Referee check v8 (CC14): (a) re-evaluate the stored best_cd_*.npz with vlib; (b) a short independent CD search at
# (5,2,2), S = {1,2} (not covered by the agent's runs), seeded from the CC12 counterexample (cheap one-neuron feature +
# KM2 block + extras) and from random class-C starts, unequal widths.
import sys, glob, time
import numpy as np, torch
sys.path.insert(0, '/home/user/grok/heracles/conjecture20/unify/C_ccfactor/verify')
import vlib as V
torch.set_default_dtype(torch.float64)
D = '/home/user/grok/heracles/conjecture20/unify/C_ccfactor/'
print('=== (a) stored best_cd networks')
for f in sorted(glob.glob(D + 'best_cd_*.npz')):
    d = np.load(f); U, W, lab = d['U'], d['W'], d['lab']; m, n, p = U.shape; nu = n + 1
    b = float(f.split('_b')[1].split('_')[0]); s = nu / b; B = V.Bp(p, n); pi = nu ** (1 - s); F = (p - 1) // 2
    h = V.gaps(U, W).min(); sq = V.sq(U, W)
    Pl = np.array([(sq[lab == l] ** (b / 2)).sum() for l in range(1, F + 1)])
    print(f'  {f.split("/")[-1]}: CD ratio {h / (pi * B * V.Mcov(Pl ** s, p)):.6f}')

print('=== (b) CD search at (5,2,2), S={1,2}')
p, n, b = 5, 2, 2.0; nu = 3; s = nu / b; B = V.Bp(p, n); pi = 3 ** (1 - s)
a = np.arange(p); X = np.indices((p,) * n).reshape(n, -1).T; Y = torch.tensor(X.sum(1) % p); N = p ** n; ar = torch.arange(N)
Kt = torch.tensor(V.kap(p))
cex = np.load(D + 'cex_charge_rules_p5_n2_b2.npz')
rng = np.random.default_rng(5); t0 = time.time(); best = -9


def coeffs(v, l):
    vh = np.fft.fft(v)[l] * 2 / p
    return np.array([vh.real, -vh.imag])


def run(cu, cw, lab):
    bas = torch.tensor(np.stack([np.stack([np.cos(2 * np.pi * l * a / p), np.sin(2 * np.pi * l * a / p)]) for l in lab]))
    oh = torch.tensor(np.array([[1.0 if l == k else 0.0 for k in (1, 2)] for l in lab]))
    cu = torch.tensor(cu, requires_grad=True); cw = torch.tensor(cw, requires_grad=True)

    def q(tau):
        U = torch.einsum('itk,ikx->itx', cu, bas); W = torch.einsum('ik,ikx->ix', cw, bas)
        f = torch.einsum('ia,ib,ic->abc', U[:, 0], U[:, 1], W).reshape(N, p)
        G = torch.stack([f[ar, Y] - f[ar, (Y - d) % p] for d in range(1, p)], 1).reshape(-1)
        Pl = oh.T @ ((U ** 2).sum((1, 2)) + (W ** 2).sum(1)) ** (b / 2)
        cov = Kt @ Pl ** s
        if tau is None:
            return G.min() / (pi * B * cov.min())
        sc = G.detach().abs().mean(); h = -tau * sc * torch.logsumexp(-G / (tau * sc), 0)
        cc = cov.detach().mean(); M = -tau * cc * torch.logsumexp(-cov / (tau * cc), 0)
        return h / (pi * B * M)
    for tau in [0.03, 0.01, 3e-3, 1e-3, 3e-4, 1e-4]:
        o = torch.optim.LBFGS([cu, cw], lr=1, max_iter=200, line_search_fn='strong_wolfe')

        def closure():
            o.zero_grad(); loss = -q(tau); loss.backward(); return loss
        try:
            o.step(closure)
        except Exception:
            pass
    with torch.no_grad():
        return float(q(None))


Uc, Wc, labc = cex['U'], cex['W'], cex['lab']
cu0 = np.array([[coeffs(Uc[i, t], labc[i]) for t in range(n)] for i in range(len(labc))])
cw0 = np.array([coeffs(Wc[i], labc[i]) for i in range(len(labc))])
sc = np.sqrt((cu0 ** 2).sum((1, 2)) + (cw0 ** 2).sum(1)).mean()
for rep in range(14):
    if rep == 0:
        val = run(cu0 / sc, cw0 / sc, labc)
    elif rep % 2 == 1:
        val = run(cu0 / sc * (1 + 0.2 * rng.normal(size=cu0.shape)), cw0 / sc * (1 + 0.2 * rng.normal(size=cw0.shape)), labc)
    else:
        w1, w2 = int(rng.integers(1, 4)), int(rng.integers(2, 7)); lab = np.array([1] * w1 + [2] * w2)
        val = run(rng.normal(size=(len(lab), n, 2)), rng.normal(size=(len(lab), 2)), lab)
    best = max(best, val)
    print(f'  rep {rep}: max CD ratio found = {val:.6f}', flush=True)
    if time.time() - t0 > 420: break
print(f'  best CD ratio at (5,2,2) S={{1,2}}: {best:.6f}')
