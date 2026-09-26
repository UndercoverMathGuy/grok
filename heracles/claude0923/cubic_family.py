# astra PROJECTION_OBSTRUCTION cubic-phase family (n=2, q=2) on the true full table, against the ansatz.
# One neuron per k=1..h: E1=E2=cos(2pi k x/p + a_k), W=cos(2pi k c/p + 2a_k), a_k = pi k^3/p.
# Reports true normalized margin m/P^{3/2} relative to: the dense junk-free block (same clean spectrum, cheapest clean rep)
# and the ansatz 3^{-1/2} B Gamma*, B = 2(6p)^{-3/2}, Gamma* = multistart LOWER bound (so ratios to the ansatz are over-estimates).
# ceilA = sqrt3 * block/ansatz = own-share ceiling for ANY net whose clean shares are uniform over all h frequencies.
# usage: cubic_family.py p1 p2 ...
import numpy as np, torch, math, sys
torch.set_default_dtype(torch.float64)

def margin(p, E1, E2, W):
    y = np.arange(p); m = np.inf
    for x1 in range(p):
        Z = (E1[:, x1][:, None] * E2).T @ W
        yy = (x1 + y) % p
        corr = Z[y, yy].copy(); Z[y, yy] = -np.inf
        m = min(m, (corr - Z.max(1)).min())
    P = (E1**2).sum() + (E2**2).sum() + (W**2).sum()
    return m / P**1.5

def family(p, a1, a2):
    h = (p - 1) // 2; x = np.arange(p); k = np.arange(1, h + 1)[:, None]
    return (np.cos(2*np.pi*k*x/p + a1[:, None]), np.cos(2*np.pi*k*x/p + a2[:, None]),
            np.cos(2*np.pi*k*x/p + (a1 + a2)[:, None]))

def gamma_lb(p, trials=60):
    h = (p - 1) // 2; k = torch.arange(1, h + 1)
    A = 1 - torch.cos(2*math.pi*torch.outer(k, k)/p); best = 0; g = torch.Generator().manual_seed(0)
    for t in range(trials):
        z = (torch.randn(h, generator=g)*2).requires_grad_(); opt = torch.optim.Adam([z], lr=0.05)
        for s in range(600):
            tau = 0.05*(0.01**(s/600)); v = A @ torch.softmax(z, 0)**1.5
            loss = tau*torch.logsumexp(-v/tau, 0); opt.zero_grad(); loss.backward(); opt.step()
        best = max(best, (A @ torch.softmax(z, 0).detach()**1.5).min().item())
    return best

rng = np.random.default_rng(1)
for p in [int(a) for a in sys.argv[1:]]:
    h = (p - 1) // 2; k = np.arange(1, h + 1)
    G = gamma_lb(p); ans = 3**-0.5 * 2*(6*p)**-1.5 * G
    block = 3**-0.5 * 2*(6*p)**-1.5 * h**-1.5 * (h + 0.5)          # dense junk-free block, closed form
    cub = margin(p, *family(p, np.pi*k**3/p, np.pi*k**3/p))
    rnd = max(margin(p, *family(p, *rng.uniform(0, 2*np.pi, (2, h)))) for _ in range(3))
    print(f"p={p:4d} Gamma>={G:.5f}  cubic/block={cub/block:+.3f}  cubic/ansatz={cub/ans:+.3f}  "
          f"random/ansatz={rnd/ans:+.3f}  ceilA={3**0.5*block/ans:.3f}", flush=True)
