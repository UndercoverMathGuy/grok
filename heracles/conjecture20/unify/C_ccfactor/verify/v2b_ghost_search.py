# Referee check v2b: can class-C ghost neurons (frequencies k != 1 with EXACTLY cancelling atoms, beta_k = 0) raise
# gamma above pi B_p kappa_min for clean support S = {1}?  (If yes: RD_{1} fails in class C.  If no: evidence only.)
# Exact parameterisation: in each ghost group the last neuron's output is solved so that sum of atoms = 0.
# usage: python v2b_ghost_search.py p w1 wg restarts seed
import sys, time
import numpy as np, torch
torch.set_default_dtype(torch.float64)
p, w1, wg, R, seed = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
n, b = 2, 2.0; nu = 3; s = nu / b; F = (p - 1) // 2
Bp = 2.0 * (2.0 * p * nu) ** (-nu / 2.0); pi = 3 ** (1 - s); kmin = 1 - np.cos(2 * np.pi / p)
a = np.arange(p)
ghost_freqs = list(range(2, F + 1))
X = np.indices((p,) * n).reshape(n, -1).T; Y = X.sum(1) % p; N = p ** n
Yt = torch.tensor(Y); ar = torch.arange(N)


def basis(k):
    return torch.tensor(np.stack([np.cos(2 * np.pi * k * a / p), np.sin(2 * np.pi * k * a / p)]))


def z_of(c):            # c = (cos coef, sin coef) -> complex proxy of DFT at k
    return torch.complex(c[..., 0], -c[..., 1])


def c_of(z):
    return torch.stack([z.real, -z.imag], -1)


def build(params):
    Us, Ws = [], []
    cu1, cw1 = params[0], params[1]
    B1 = basis(1)
    Us.append(torch.einsum('itk,kx->itx', cu1, B1)); Ws.append(torch.einsum('ik,kx->ix', cw1, B1))
    for gi, k in enumerate(ghost_freqs):
        cu, cw = params[2 + 2 * gi], params[3 + 2 * gi]        # cu (wg, n, 2), cw (wg-1, 2)
        zu = z_of(cu); zw = z_of(cw)
        atoms = torch.conj(zw) * zu[:-1, 0] * zu[:-1, 1]
        zw_last = torch.conj(-atoms.sum() / (zu[-1, 0] * zu[-1, 1]))
        cw_full = torch.cat([cw, c_of(zw_last)[None, :]], 0)
        Bk = basis(k)
        Us.append(torch.einsum('itk,kx->itx', cu, Bk)); Ws.append(torch.einsum('ik,kx->ix', cw_full, Bk))
    return torch.cat(Us), torch.cat(Ws)


def gamma(params, tau):
    U, W = build(params)
    f = torch.einsum('ia,ib,ic->abc', U[:, 0], U[:, 1], W).reshape(N, p)
    top = f[ar, Yt]
    G = torch.stack([top - f[ar, (Yt - d) % p] for d in range(1, p)], 1).reshape(-1)
    if tau is None:
        h = G.min()
    else:
        sc = G.detach().abs().mean(); h = -tau * sc * torch.logsumexp(-G / (tau * sc), 0)
    P = ((U ** 2).sum((1, 2)) + (W ** 2).sum(1)).pow(b / 2).sum()
    return h / P ** s


rng = np.random.default_rng(seed); best = -9; t0 = time.time()
for rep in range(R):
    params = []
    if rep % 2 == 0:   # IF block at 1 + noise
        cu1 = np.zeros((w1, n, 2)); cw1 = np.zeros((w1, 2))
        for j in range(w1):
            al = np.pi * (j % nu) / nu
            cu1[j, :, 0] = np.cos(al); cu1[j, :, 1] = -np.sin(al); cw1[j] = [np.cos(n * al), -np.sin(n * al)]
        cu1 += 0.3 * rng.normal(size=cu1.shape); cw1 += 0.3 * rng.normal(size=cw1.shape)
    else:              # random (cheap) start
        cu1 = rng.normal(size=(w1, n, 2)); cw1 = rng.normal(size=(w1, 2))
    params += [cu1, cw1]
    for k in ghost_freqs:
        params += [0.5 * rng.normal(size=(wg, n, 2)), 0.5 * rng.normal(size=(wg - 1, 2))]
    params = [torch.tensor(x, requires_grad=True) for x in params]
    for tau in [0.1, 0.03, 0.01, 3e-3, 1e-3, 3e-4]:
        o = torch.optim.LBFGS(params, lr=1, max_iter=200, line_search_fn='strong_wolfe')

        def closure():
            o.zero_grad(); loss = -gamma(params, tau) / (pi * Bp * kmin); loss.backward(); return loss
        try:
            o.step(closure)
        except Exception:
            pass
    with torch.no_grad():
        val = float(gamma(params, None) / (pi * Bp * kmin))
        U, W = build(params)
        gsz = float(((U[w1:] ** 2).sum() + (W[w1:] ** 2).sum()) / ((U ** 2).sum() + (W ** 2).sum()))
    print(f'rep {rep}: gamma/(pi B kappa_min) = {val:.6f}   ghost share of squared norm = {gsz:.3e}', flush=True)
    best = max(best, val)
    if time.time() - t0 > 400: break
print(f'SUMMARY p={p} w1={w1} wg={wg}: best gamma/(pi B_p kappa_min) over clean-support {{1}} class-C ghost networks = {best:.6f}')
