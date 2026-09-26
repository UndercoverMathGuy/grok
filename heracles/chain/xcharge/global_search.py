# Adversarial search for violations of T / CA / CAb (see global_eval.py), unrestricted networks (mixing neurons, DC allowed).
# usage: global_search.py p N mode seeds kind     kind in {lmax, blockmix, random}
import sys, math, numpy as np, torch
sys.path.insert(0, '..')
from common import programme, block_net
from global_eval import evaluate
torch.set_default_dtype(torch.float64)
p, N, mode, seeds, kind = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), sys.argv[5]
h = (p - 1) // 2; Gam, ystar = programme(p, 1.5)
x = torch.arange(p); S = (x[:, None] + x[None, :]) % p
def objective(v, beta, delta):
    E1, E2, W = v.reshape(3, N, p)
    Z = torch.einsum('ia,ib,ic->abc', E1, E2, W); corr = Z[x[:, None], x[None, :], S]
    gaps = torch.stack([corr - Z[x[:, None], x[None, :], (S - D) % p] for D in range(1, p)], -1).reshape(-1)
    m = -torch.logsumexp(-beta * gaps, 0) / beta
    if mode == 'T':
        return m / (3 ** -0.5 * 2 * (6 * p) ** -1.5 * Gam * (v @ v) ** 1.5)
    F1, F2, FW = torch.fft.fft(E1), torch.fft.fft(E2), torch.fft.fft(W)
    if mode == 'CA':
        c2 = (F1[:, 1:h + 1] * F2[:, 1:h + 1] * FW[:, 1:h + 1]).abs() ** 2
        B = ((c2 + delta) ** (1 / 3)).sum() / 3 ** (1 / 3)
    else:
        b = 2 / p * (F1.abs() ** 2 + F2.abs() ** 2 + FW.abs() ** 2)[:, 1:h + 1].sum()
        B = (p / 6) * b / 3 ** (1 / 3)
    return m / (2 / p ** 3 * Gam * B ** 1.5)
rng = np.random.default_rng(1)
def seed_vec(s):
    if kind == 'lmax':
        fs = {5: [3], 7: [9]}[p]
        v = np.load(f'../../claude0923/endpoint_p{p}_s{fs[0]}.npy').reshape(3, -1, p)
        v = np.concatenate([v, np.zeros((3, max(0, N - v.shape[1]), p))], 1)[:, :N]
        return (v + (0.0 if s == 0 else 0.02 * s ** 0.5) * rng.normal(size=v.shape)).ravel()
    if kind == 'blockmix':
        net = block_net(p, ystar)
        v = np.zeros((3, N, p))
        for i, (e1, e2, w) in enumerate(net[:N]): v[:, i] = [e1, e2, w]
        eps = [0.05, 0.1, 0.2, 0.3][s % 4]
        return (v + eps * rng.normal(size=v.shape) * np.abs(v).max()).ravel()     # random mixing + DC perturbation
    return rng.normal(size=3 * N * p)
res = []
for s in range(seeds):
    v = torch.tensor(seed_vec(s)); v = (v / v.norm()).requires_grad_()
    opt = torch.optim.Adam([v], lr=0.01)
    for it in range(4000):
        beta = 300 * (1 + it / 400); delta = 1e-6 * (1e-6) ** (it / 4000)
        loss = -objective(v / v.norm(), beta, delta)
        opt.zero_grad(); loss.backward(); opt.step()
    vn = (v / v.norm()).detach().numpy().reshape(3, N, p)
    r = evaluate(p, [(vn[0, i], vn[1, i], vn[2, i]) for i in range(N)])
    res.append((r[mode], s)); np.save(f'gs_{p}_{mode}_{kind}_s{s}.npy', vn)
    print(f"p={p} {mode} {kind} seed {s}: T {r['T']:.5f} CA {r['CA']:.5f} CAb {r['CAb']:.5f} CAoff {r['CAoff']:.5f}", flush=True)
print("MAX", max(res))
