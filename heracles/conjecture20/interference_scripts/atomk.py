# k*(n,s) = sup over atoms (beta_i in C, chains xi_i in T^n) of (Re A - nu^{1-s} C^s) / sqrt(E/2)
#   A = sum beta_i, C = sum |beta_i|^{1/s}, E = sum_{T != empty} |sum_i beta_i xi_i^T|^2
import sys, math, numpy as np, torch
torch.set_default_dtype(torch.float64); torch.set_num_threads(1)
def run(n, s, m, seed, iters=4000, sym=False):
    nu = n + 1; r = 1 / s
    g = torch.Generator().manual_seed(seed)
    br = torch.randn(m, generator=g).requires_grad_(); bi = (0.3 * torch.randn(m, generator=g)).requires_grad_()
    th = (2 * math.pi * torch.rand(m, 1 if sym else n, generator=g)).requires_grad_()
    opt = torch.optim.Adam([br, bi, th], lr=0.01)
    best = -1e9
    for it in range(iters):
        b = torch.complex(br, bi)
        T = th.expand(m, n) if sym else th
        d = T[:, None, :] - T[None, :, :]
        G = torch.prod(1 + torch.exp(1j * d), dim=2)  # G_ij = prod_t (1 + xi_it conj xi_jt)
        tot = (b[:, None] * b.conj()[None, :] * G).sum().real
        A = b.sum()
        E = (tot - A.abs() ** 2).clamp_min(1e-30)
        C = (b.abs() + 1e-30).pow(r).sum()
        val = (A.real - nu ** (1 - s) * C ** s) / torch.sqrt(E / 2)
        loss = -val
        opt.zero_grad(); loss.backward(); opt.step()
        if A.real.item() > nu ** (1 - s) * C.item() ** s:
            best = max(best, val.item())
    return best
n = int(sys.argv[1]); s = float(sys.argv[2]); sym = len(sys.argv) > 3 and sys.argv[3] == 'sym'
nu = n + 1
res = []
for m in range(1, nu + 3):
    v = max(run(n, s, m, sd, sym=sym) for sd in range(4))
    res.append(v); print("n=%d s=%.3f m=%d k>=%.4f" % (n, s, m, v), flush=True)
print("SUP n=%d s=%.3f %s k*>=%.4f" % (n, s, 'sym' if sym else 'gen', max(res)))
