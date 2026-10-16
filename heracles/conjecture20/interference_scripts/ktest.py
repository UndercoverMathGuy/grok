# robust constant k*(n,s) at larger n with slot-symmetric atoms (fast) -- lower bounds on the sup
import sys, math, numpy as np, torch
from math import comb
torch.set_default_dtype(torch.float64); torch.set_num_threads(1)
def run(n, s, m, seed, iters=3000):
    nu = n + 1; r = 1 / s
    g = torch.Generator().manual_seed(seed)
    br = torch.randn(m, generator=g).requires_grad_(); bi = (0.3 * torch.randn(m, generator=g)).requires_grad_()
    th = (2 * math.pi * torch.rand(m, generator=g)).requires_grad_()
    binom = torch.tensor([comb(n, k) for k in range(1, n + 1)], dtype=torch.float64)
    ks = torch.arange(1, n + 1, dtype=torch.float64)
    opt = torch.optim.Adam([br, bi, th], lr=0.01); best = -1e9
    for it in range(iters):
        b = torch.complex(br, bi)
        Mk = (b[None, :] * torch.exp(1j * ks[:, None] * th[None, :])).sum(1)
        E = (binom * Mk.abs() ** 2).sum().clamp_min(1e-30)
        A = b.sum(); C = (b.abs() + 1e-30).pow(r).sum()
        val = (A.real - nu ** (1 - s) * C ** s) / torch.sqrt(E / 2)
        opt.zero_grad(); (-val).backward(); opt.step()
        if A.real.item() > nu ** (1 - s) * C.item() ** s: best = max(best, val.item())
    return best
for n in [int(x) for x in sys.argv[1].split(',')]:
    s = (n + 1) / 2
    v = max(run(n, s, m, sd) for m in range(2, n + 3) for sd in range(3))
    print("sym n=%d s=%.1f k*>=%.4f" % (n, s, v), flush=True)
