# kappa*(nu,s) = sup over complex nu x nu H with unit diagonal of (1 - nu^{-s} ||H||_{S_r}) / ||strictlower(H)||_F
import sys, math, numpy as np, torch
torch.set_default_dtype(torch.float64); torch.set_num_threads(1)
def run(nu, s, seed, iters=3000):
    g = torch.Generator().manual_seed(seed)
    r = 1 / s
    X = (torch.randn(nu, nu, generator=g) + 1j * torch.randn(nu, nu, generator=g)) * float(torch.rand(1, generator=g)) * 2
    X = X.to(torch.complex128).requires_grad_()
    I = torch.eye(nu, dtype=torch.complex128)
    lowmask = torch.tril(torch.ones(nu, nu), -1).to(torch.complex128)
    offmask = (1 - torch.eye(nu)).to(torch.complex128)
    opt = torch.optim.Adam([X], lr=0.02)
    best = -1e9
    for it in range(iters):
        H = I + X * offmask
        sv = torch.linalg.svdvals(H)
        Sr = (sv.clamp_min(1e-12) ** r).sum() ** (1 / r)
        Lf = (H * lowmask).abs().pow(2).sum().sqrt()
        val = (1 - nu ** (-s) * Sr) / (Lf + 1e-9)
        loss = -val
        opt.zero_grad(); loss.backward(); opt.step()
        best = max(best, val.item())
    return best
nu = int(sys.argv[1]); svals = [float(x) for x in sys.argv[2].split(',')]
for s in svals:
    b = max(run(nu, s, sd) for sd in range(6))
    print("nu=%d s=%.3f kappa*>=%.4f" % (nu, s, b), flush=True)
