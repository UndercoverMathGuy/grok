# Adversarial test for "junk cannot hide on a competitive spectrum" (n=2, q=2, pure neurons, exact profile identity of ptrend.py).
# Frequencies on an AP {d, 2d, ..., Ld} (mod p, folded to 1..h) get ONE atom each (forced full junk, cost 1 instead of 3^{1/3}),
# initialised with chirp or random junk phases (flat junk hides best on an AP). Every other frequency gets 3 atoms (may be
# junk-free), initialised as clean blocks on the Gamma-optimal support, so it can cover the AP's bad offsets.
# Then everything is optimised. ratio = 1 at the ansatz (Gamma = multistart LOWER bound, so the ratio is an over-estimate).
# Reports final ratio and the share of the budget left on the junky AP.  usage: adv_ap.py p L d seeds [steps] [sigma]
# sigma >= 0 pins the AP's share of the penalty budget at sigma (AP atoms rescaled every step), tracing ratio vs junk share.
import torch, math, sys
torch.set_default_dtype(torch.float64)
p, L, d, seeds = (int(a) for a in sys.argv[1:5]); steps = int(sys.argv[5]) if len(sys.argv) > 5 else 3000
sigma = float(sys.argv[6]) if len(sys.argv) > 6 else -1.0
h = (p - 1) // 2; k = torch.arange(1, h + 1); x = torch.arange(p); D = torch.arange(1, p)
A = 1 - torch.cos(2*math.pi*torch.outer(k, k)/p)
def gamma_prog(trials=100):
    best = 0; bb = None; g = torch.Generator().manual_seed(0)
    for t in range(trials):
        z = (torch.randn(h, generator=g)*2).requires_grad_(); opt = torch.optim.Adam([z], lr=0.05)
        for s in range(600):
            tau = 0.05*(0.01**(s/600)); v = A @ torch.softmax(z, 0)**1.5
            loss = tau*torch.logsumexp(-v/tau, 0); opt.zero_grad(); loss.backward(); opt.step()
        b = torch.softmax(z, 0).detach(); val = (A @ b**1.5).min().item()
        if val > best: best = val; bb = b
    return best, bb
Gam, bopt = gamma_prog()
W = 1 - torch.exp(-2j*math.pi*torch.outer(k, D)/p)
U = torch.exp(-2j*math.pi*2*torch.outer(k, x)/p)
ap = sorted({min((d*j) % p, p - (d*j) % p) for j in range(1, L + 1)})
isap = torch.zeros(h, dtype=torch.bool); isap[[a - 1 for a in ap]] = True
mask = torch.ones(h, 3); mask[isap, 1:] = 0                    # AP frequencies: one atom only
def pinned(cr, ci):
    if sigma < 0: return cr*mask, ci*mask
    m2 = (cr*mask)**2 + (ci*mask)**2 + 1e-24; cost = m2.pow(1/3)
    cap = cost[isap].sum(); ccov = cost[~isap].sum()
    t = (sigma/(1 - sigma)*ccov/cap)**1.5
    sc = torch.where(isap, t, torch.ones(h))[:, None]
    return cr*mask*sc, ci*mask*sc
def profile(cr, ci, a1, a2):
    cr, ci = pinned(cr, ci)
    c = torch.complex(cr, ci); v1 = torch.exp(1j*a1); v2 = torch.exp(1j*a2)
    C0 = c.sum(1); C1 = (c*v1).sum(1); C2 = (c*v2).sum(1); C12 = (c*v1*v2).sum(1)
    H = C0[:, None, None] + C1[:, None, None]*U[:, :, None] + C2[:, None, None]*U[:, None, :] + C12[:, None, None]*U[:, :, None]*U[:, None, :]
    g = torch.einsum('kab,kd->abd', H, W).real
    return g, (cr**2 + ci**2 + 1e-24).pow(1/3).sum()
print(f"sigma={sigma} p={p} Gamma>={Gam:.6f} opt support={[i+1 for i in range(h) if bopt[i]>1e-3]}  AP(L={L},d={d})={ap}", flush=True)
best = -9
for sd in range(seeds):
    gg = torch.Generator().manual_seed(sd)
    amp = bopt**1.5 + 0.002*torch.rand(h, generator=gg)
    apamp = 0.5*amp.max()*(0.2 + torch.rand(1, generator=gg)).item()
    amp = torch.where(isap, torch.full_like(amp, apamp), amp)
    cr = (amp[:, None]/3*torch.ones(h, 3)).clone(); ci = 0.01*amp[:, None]*torch.randn(h, 3, generator=gg)
    ph = 2*math.pi*torch.arange(3)/3
    a1 = ph.repeat(h, 1).clone(); a2 = ph.repeat(h, 1).clone()
    j = torch.zeros(h); j[isap] = torch.arange(1, int(isap.sum()) + 1, dtype=torch.float64)
    if sd % 2 == 0: chirp = math.pi*j**2/max(L, 1)            # chirp junk phases on the AP
    else: chirp = 2*math.pi*torch.rand(h, generator=gg)
    a1[:, 0] = torch.where(isap, chirp, a1[:, 0]); a2[:, 0] = torch.where(isap, -chirp if sd % 4 == 0 else chirp, a2[:, 0])
    cr += 0.01*amp[:, None]*torch.randn(h, 3, generator=gg)
    for t in (cr, ci, a1, a2): t.requires_grad_()
    opt = torch.optim.Adam([cr, ci, a1, a2], lr=5e-3)
    for st in range(steps):
        tau = 0.1*(0.005**(st/steps))
        g, cost = profile(cr, ci, a1, a2); r = math.sqrt(3)*g.reshape(-1)/cost**1.5/Gam
        loss = tau*torch.logsumexp(-r/tau, 0); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        g, cost = profile(cr, ci, a1, a2); val = (math.sqrt(3)*g.min()/cost**1.5/Gam).item()
        pr, pi_ = pinned(cr, ci); sh = (pr**2 + pi_**2).pow(1/3).sum(1); sh = sh/sh.sum()
    best = max(best, val)
    print(f"seed {sd}: ratio {val:.4f} best {best:.4f}  AP share {sh[isap].sum().item():.3f}  "
          f"support {[i+1 for i in range(h) if sh[i] > 0.01]}", flush=True)
