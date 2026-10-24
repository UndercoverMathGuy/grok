# Sanity check of the p-dependent Parseval--efficiency closure (relations case):
#   h_model(a,v,th) <= m(a) * (1 + tau' (c - sqrt(2/p))_+ )         for all 0 <= v <= tau' a, all phases,
# where h_model = min_{w,d} [ sum_l A_l kappa_l(d) + phi(w) - phi(w-d) ],  A = a + c v (largest clean amplitude P12 allows).
# Phases are pushed adversarially (smoothed-min ascent) to stress the inequality.  Also checks, at p <= 59, that
#   h_model <= Gamma_ref ||a||  (the Note's closure, recovered here as e_thr(p) = 1).
import numpy as np, torch, math
from lib import *
torch.set_default_dtype(torch.float64); torch.set_num_threads(1)

def hmodel_torch(p, A, v, th, tau=None):
    F = (p-1)//2; l = torch.arange(1, F+1, dtype=torch.float64); w = torch.arange(p, dtype=torch.float64)
    ang = 2*math.pi*torch.outer(l, w)/p + th[:, None]
    phi = (v[:, None]*torch.cos(ang)).sum(0)
    K = torch.tensor(kmat(p))
    vals = []
    for d in range(1, p):
        dd = min(d, p-d)
        base = (A*K[dd-1]).sum()
        D = phi - torch.roll(phi, d)
        vals.append(base + D)
    V = torch.stack(vals).reshape(-1)
    if tau is None: return V.min()
    return -tau*torch.logsumexp(-V/tau, 0) + tau*math.log(V.numel())*0

rng = np.random.default_rng(5)
worst_ratio = 0; worst_small = -1e9
for trial in range(60):
    p = int(rng.choice([31, 43, 59, 61, 67, 101, 151]))
    s = float(rng.uniform(1.0, 1.5)); r, c, tp = consts(s)
    F = (p-1)//2; K = kmat(p)
    k = int(rng.integers(5, F+1)); S = rng.choice(F, k, replace=False)
    a = np.zeros(F); a[S] = rng.uniform(0.2, 1.0, k)**3
    v = np.zeros(F); v[S] = tp*a[S]*rng.uniform(0.5, 1.0, k)
    A = torch.tensor(a + c*v); vt = torch.tensor(v)
    th = torch.tensor(rng.uniform(0, 2*np.pi, F), requires_grad=True)
    opt = torch.optim.Adam([th], lr=0.05)
    for it in range(150):
        opt.zero_grad(); loss = -hmodel_torch(p, A, vt, th, tau=0.01*float(A.sum())); loss.backward(); opt.step()
    with torch.no_grad(): h = float(hmodel_torch(p, A, vt, th))
    m = (K @ a).min()
    bound = m*(1 + tp*max(0.0, c - np.sqrt(2/p)))
    worst_ratio = max(worst_ratio, h/bound)
    if p <= 59:
        G = load_support(p)[0] if abs(s-1.5) < 1e-12 else prog(p, s, restarts=3)[0]
        worst_small = max(worst_small, h/(G*norm_r(a, s)))
print(f"max h_model / [m(a)(1+tau'(c-sqrt(2/p))_+)] over 60 adversarial configs = {worst_ratio:.4f}  (must be <= 1)")
print(f"max h_model / (Gamma ||a||) at p <= 59 = {worst_small:.4f}  (must be <= 1)")
