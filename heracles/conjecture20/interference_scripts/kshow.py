import sys, math, numpy as np, torch
from math import comb
torch.set_default_dtype(torch.float64); torch.set_num_threads(1)
def run(n, s, m, seed, k, iters=4000):
    # maximise ReA - k sqrt(E/2) - nu^{1-s} C^s at C=1 normalisation (homogeneous: use ratio)
    nu = n + 1; r = 1 / s
    g = torch.Generator().manual_seed(seed)
    br = torch.randn(m, generator=g).requires_grad_(); bi = (0.3 * torch.randn(m, generator=g)).requires_grad_()
    th = (2 * math.pi * torch.rand(m, generator=g)).requires_grad_()
    binom = torch.tensor([comb(n, j) for j in range(1, n + 1)], dtype=torch.float64)
    ks = torch.arange(1, n + 1, dtype=torch.float64)
    opt = torch.optim.Adam([br, bi, th], lr=0.01)
    for it in range(iters):
        b = torch.complex(br, bi)
        Mk = (b[None, :] * torch.exp(1j * ks[:, None] * th[None, :])).sum(1)
        E = (binom * Mk.abs() ** 2).sum().clamp_min(1e-30)
        A = b.sum(); C = (b.abs() + 1e-30).pow(r).sum()
        val = (A.real - k * torch.sqrt(E / 2)) / C ** s / nu ** (1 - s)
        opt.zero_grad(); (-val).backward(); opt.step()
    return val.item(), b.detach(), th.detach(), (E / A.abs() ** 2).item()
n = int(sys.argv[1]); k = float(sys.argv[2]); s = (n + 1) / 2
best = max((run(n, s, m, sd, k) for m in range(2, n + 3) for sd in range(3)), key=lambda t: t[0])
v, b, th, Er = best
np.save("katoms_n%d_k%.3f.npy" % (n, k), np.stack([b.real.numpy(), b.imag.numpy(), th.numpy()]))
print("n=%d k=%.3f  best (ReA - k sigma)/IF = %.4f  m=%d  E/|A|^2=%.3f" % (n, k, v, len(b), Er))
print("  |beta|/max:", np.round((b.abs() / b.abs().max()).numpy(), 3), " phases(deg):", np.round((th.numpy() * 180 / math.pi) % 360, 1))
