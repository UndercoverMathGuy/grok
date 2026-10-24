# C_ccfactor c3: adversarial searches in class C (torch, smooth-min annealing).
#   mode 'cd'  : maximise the canonical-decoupling ratio  h / (pi B_p M(P^s))          (CD; exact factorisation)
#   mode 'rd'  : maximise gamma / (pi B_p Gamma^{(S)}_s)                                (RD, sanity)
#   mode 'pin' : maximise the CD ratio with the cost profile zhat ~ P^s pinned near a target (penalty)
# usage: python c3_search.py mode p n b support(comma) width restarts seed [target(comma)]
import sys, time
import numpy as np, torch
sys.path.insert(0, '/home/user/grok/heracles/conjecture20/unify/E_width')
import elib as E
torch.set_default_dtype(torch.float64)
mode, p, n, b = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
S = [int(v) for v in sys.argv[5].split(',')]; width = int(sys.argv[6]); R = int(sys.argv[7]); seed = int(sys.argv[8])
target = np.array([float(v) for v in sys.argv[9].split(',')]) if len(sys.argv) > 9 else None
nu = n + 1; s = nu / b; r = 1 / s; F = (p - 1) // 2
pi = nu ** (1 - s)          # = 3^{1-s} at n=2 (proved); a lower bound for pi at n>=3
Bp = E.Bp(p, n); GS, ys = E.programme_vertices(p, s, S)
kap = torch.tensor(E.kappa(p)[:, [l - 1 for l in S]])          # (p-1, |S|)
a = np.arange(p)
idx = np.indices((p,) * n).reshape(n, -1).T; Y = idx.sum(1) % p; N = p ** n
freq = [l for l in S for _ in range(width)]; m = len(freq)
bas = torch.tensor(np.stack([np.stack([np.cos(2 * np.pi * l * a / p), np.sin(2 * np.pi * l * a / p)]) for l in freq]))  # (m,2,p)
Xt = torch.tensor(idx); Yt = torch.tensor(Y)
LET = 'abcdefgh'
ein = ','.join('i' + LET[t] for t in range(n)) + ',i' + LET[n] + '->' + LET[:n + 1]
onehot = torch.tensor(np.array([[1.0 if l == k else 0.0 for k in S] for l in freq]))      # (m,|S|)


def build(cu, cw):
    U = torch.einsum('itk,ikx->itx', cu, bas); W = torch.einsum('ik,ikx->ix', cw, bas)
    return U, W


def quantities(cu, cw, tau):
    U, W = build(cu, cw)
    f = torch.einsum(ein, *[U[:, t] for t in range(n)], W).reshape(N, p)
    top = f[torch.arange(N), Yt]
    G = torch.stack([top - f[torch.arange(N), (Yt - d) % p] for d in range(1, p)], 1)
    sq = (U ** 2).sum((1, 2)) + (W ** 2).sum(1)
    Pl = onehot.T @ sq ** (b / 2)
    cov = kap @ Pl ** s
    if tau is None:
        h = G.min(); M = cov.min()
    else:
        sc = G.detach().abs().mean(); h = -tau * sc * torch.logsumexp(-G.reshape(-1) / (tau * sc), 0)
        cc = cov.detach().mean(); M = -tau * cc * torch.logsumexp(-cov / (tau * cc), 0)
    P = Pl.sum()
    zh = Pl ** s / (Pl ** s).sum()
    return h, M, P, zh


def ratios(cu, cw):
    with torch.no_grad():
        h, M, P, zh = quantities(cu, cw, None)
    return float(h / (pi * Bp * M)), float(h / P ** s / (pi * Bp * GS)), zh.numpy()


rng = np.random.default_rng(seed)
best = (-9.0, None)
t0 = time.time()
for rep in range(R):
    if rep % 2 == 0:      # perturbed decoupled IF network with a random allocation
        y = rng.uniform(0.05, 1.0, len(S))
        cu0 = np.zeros((m, n, 2)); cw0 = np.zeros((m, 2))
        for k, l in enumerate(S):
            lam = (2 ** n * y[k] / nu) ** (1.0 / nu)
            for j in range(width):
                i = k * width + j; al = np.pi * (j % nu) / nu
                for t in range(n):
                    cu0[i, t] = lam * np.array([np.cos(al), -np.sin(al)])
                cw0[i] = lam * np.array([np.cos(n * al), -np.sin(n * al)])
        cu0 = cu0 * (1 + 0.3 * rng.normal(size=cu0.shape)) + 0.3 * rng.normal(size=cu0.shape) * np.abs(cu0).mean()
        cw0 = cw0 * (1 + 0.3 * rng.normal(size=cw0.shape)) + 0.3 * rng.normal(size=cw0.shape) * np.abs(cw0).mean()
    else:
        cu0 = rng.normal(size=(m, n, 2)); cw0 = rng.normal(size=(m, 2))
    cu = torch.tensor(cu0, requires_grad=True); cw = torch.tensor(cw0, requires_grad=True)
    for stage, tau in enumerate([0.1, 0.03, 0.01, 3e-3, 1e-3, 3e-4, 1e-4]):
        o = torch.optim.LBFGS([cu, cw], lr=1, max_iter=250, line_search_fn='strong_wolfe')

        def closure():
            o.zero_grad()
            h, M, P, zh = quantities(cu, cw, tau)
            g = h / P ** s
            if stage < 2 and mode != 'rd':
                loss = -g / (pi * Bp * GS)                     # first get a positive margin
            elif mode == 'rd':
                loss = -g / (pi * Bp * GS)
            else:
                rr = h / (pi * Bp * M)
                loss = -rr
                if mode == 'pin':
                    loss = loss + 50.0 * ((zh - torch.tensor(target)) ** 2).sum()
            loss.backward(); return loss
        try:
            o.step(closure)
        except Exception as ex:
            pass
    cd, rd, zh = ratios(cu, cw)
    print(f'rep {rep:2d}: CD={cd:.5f} RD={rd:.5f} zhat={np.round(zh, 3)} ({time.time() - t0:.0f}s)', flush=True)
    if rd > 0 and cd > best[0]:
        best = (cd, rd, zh, cu.detach().numpy().copy(), cw.detach().numpy().copy())
    if time.time() - t0 > 700: break
print(f'SUMMARY {mode} p={p} n={n} b={b} S={S} width={width}: max CD ratio over positive-margin runs = {best[0]:.5f}'
      f' (RD ratio {best[1] if best[1] is not None else 0:.5f}, zhat {np.round(best[2], 3) if best[1] is not None else None})')
if best[1] is not None:
    U, W = build(torch.tensor(best[3]), torch.tensor(best[4]))
    np.savez(f'best_{mode}_p{p}_n{n}_b{b}_S{"-".join(map(str, S))}_w{width}_s{seed}.npz', U=U.detach().numpy(),
             W=W.detach().numpy(), lab=np.array(freq))
