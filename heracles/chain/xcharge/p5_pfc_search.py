# Adversarial search at p = 5: how far above the per-frequency ceiling PFC-b can a network's margin go, and at what coverage tau_b?
# R_b = margin / [(2/p^3) min_D sum_k cov_k(D) Yb_k],  Yb_k = ((p/6) b_k / 3^{1/3})^{3/2},  b_k = frequency-k weight share.
# tau_b = min_D sum cov Yb / (Gamma (sum Yb^{2/3})^{3/2}).  Note gamma/gamma_sym <= R_b * tau_b * (budget factor <= 1).
import sys, math, numpy as np, torch
torch.set_default_dtype(torch.float64)
p = int(sys.argv[1]) if len(sys.argv) > 1 else 5; N = int(sys.argv[2]) if len(sys.argv) > 2 else 8; seeds = int(sys.argv[3]) if len(sys.argv) > 3 else 40
h = (p - 1) // 2
x = torch.arange(p); Y = (x[:, None] + x[None, :]) % p
Dv = torch.arange(1, h + 1, dtype=torch.float64); kv = torch.arange(1, h + 1, dtype=torch.float64)
COV = 1 - torch.cos(2 * math.pi * Dv[:, None] * kv[None, :] / p)          # [D, k]
Gam = {5: 0.8838834764831844, 7: 0.6735753140545634, 11: 0.5357259257995056}[p]
def parts(v):
    E1, E2, W = v.reshape(3, N, p)
    Z = torch.einsum('ia,ib,ic->abc', E1, E2, W)
    corr = Z[x[:, None], x[None, :], Y]
    gaps = torch.stack([corr - Z[x[:, None], x[None, :], (Y - D) % p] for D in range(1, p)], -1)
    F1, F2, FW = torch.fft.fft(E1), torch.fft.fft(E2), torch.fft.fft(W)
    b = 2 / p * (F1.abs() ** 2 + F2.abs() ** 2 + FW.abs() ** 2)[:, 1:h + 1].sum(0)     # [k]
    Yb = ((p / 6) * b / 3 ** (1 / 3)) ** 1.5
    cov_line = COV @ Yb
    return gaps, cov_line, Yb
def objective(v, beta):
    gaps, cov_line, Yb = parts(v)
    m = -torch.logsumexp(-beta * gaps.reshape(-1), 0) / beta
    c = -torch.logsumexp(-beta * 1e3 * cov_line, 0) / (beta * 1e3)
    return m / (2 / p ** 3 * c)
def exact(v):
    with torch.no_grad():
        gaps, cov_line, Yb = parts(v)
        m = gaps.min().item(); c = cov_line.min().item()
        tau = c / (Gam * (Yb ** (2 / 3)).sum().item() ** 1.5)
        P = (v ** 2).sum().item()
        gratio = m / P ** 1.5 / (3 ** -0.5 * 2 * (6 * p) ** -1.5 * Gam)
        return m / (2 / p ** 3 * c), tau, gratio
best = []
for s in range(seeds):
    g = torch.Generator().manual_seed(s)
    v = torch.randn(3 * N * p, generator=g)
    if len(sys.argv) > 4 and sys.argv[4] == 'lmax':
        v0 = torch.tensor(np.load('../../claude0923/endpoint_p5_s3.npy')).reshape(3, 9, p)
        v = torch.cat([v0, 0.02 * torch.randn(3, N - 9, p, generator=g)], 1).reshape(-1) if N > 9 else v0[:, :N].reshape(-1).clone()
        v = v + (0.0 if s == 0 else 0.03 * s ** 0.5) * torch.randn(v.shape, generator=g) / 3
    v.requires_grad_()
    opt = torch.optim.Adam([v], lr=0.02)
    for it in range(3000):
        beta = 200 + it
        loss = -objective(v / v.norm(), beta)
        opt.zero_grad(); loss.backward(); opt.step()
    Rb, tau, gr = exact(v / v.norm())
    best.append((Rb, tau, gr, s))
    print(f"seed {s}: R_b {Rb:.4f} tau_b {tau:.4f} gamma/gamma_sym {gr:.4f}", flush=True)
best.sort(reverse=True)
print("TOP:", [tuple(round(t, 4) for t in b[:3]) + (b[3],) for b in best[:5]])
