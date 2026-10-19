# 1-D reduced "shared sign pattern" family for a single feature at general odd p (b general).
#   u_i(a) = eps_a * exp(lam_i * e'_a) / Z_i,  e' a permutation of {0..p-1}, eps fixed so that
#   prod_a (u_i(a) rho^a)^{N_a} = e^{i pi tau/p} e^{lam_i tau} / Z_i^n,  tau = <N, e'> in [0,(p-1)n].
#   Phi(tau) = e^{i pi tau/p} sum_i V_i e^{lam_i tau};  W_i = V_i Z_i^n (unit-u coefficient).
#   efficiency = min_tau cm(Phi) / (B_p P^{nu/b}) / nu^{1-s}   (cone.py normalisation)
import math, sys, numpy as np, torch
torch.set_default_dtype(torch.float64)

def lref(p, n, b):
    nu = n + 1; s = nu / b
    return (1 - s) * math.log(nu) + math.log(2) - (nu / 2) * math.log(2 * p * nu)

def eff(params, m, p, n, b=2.0, tau=None):
    nu = n + 1
    lam, lv, av = params[:m], params[m:2*m], params[2*m:3*m]
    k = torch.arange(p, dtype=torch.float64)
    logZ = 0.5 * torch.logsumexp(2 * lam[:, None] * k[None, :], 1)           # log ||u_i|| before normalising
    logW = lv + n * logZ                                                     # log|W_i|
    lc = (b / 2) * (math.log(nu) + (2 / nu) * (0.5 * math.log(p / 2) + logW))
    lognorm = (nu / b) * torch.logsumexp(lc, 0) + lref(p, n, b)
    if tau is None:
        tau = torch.arange((p - 1) * n + 1, dtype=torch.float64)
    E = lv[None, :] + lam[None, :] * tau[:, None] - lognorm
    ph = av[None, :] + math.pi * tau[:, None] / p
    re = (torch.exp(E) * torch.cos(ph)).sum(1); im = (torch.exp(E) * torch.sin(ph)).sum(1)
    return re - im.abs() / math.tan(math.pi / p)

def optimise(params, m, p, n, b=2.0, stages=(0.3, 0.1, 0.03, 0.01, 3e-3, 1e-3, 3e-4, 1e-4), iters=2000):
    P = torch.tensor(params).clone().requires_grad_()
    for t in stages:
        o = torch.optim.LBFGS([P], lr=1, max_iter=iters, line_search_fn="strong_wolfe", tolerance_grad=1e-16, tolerance_change=1e-18)
        def closure():
            o.zero_grad(); g = eff(P, m, p, n, b); loss = t * torch.logsumexp(-g / t, 0); loss.backward(); return loss
        try: o.step(closure)
        except Exception: pass
    with torch.no_grad():
        return eff(P, m, p, n, b).min().item(), P.detach().numpy().copy()

def init(m, p, n, curv, rng=None, jit=0.0):
    L = (p - 1) * n
    tc = (np.arange(m) + 0.5) * L / m                     # centre of dominance of term i
    lam = curv * (tc - L / 2)
    lv = -0.5 * curv * (tc - L / 2) ** 2 - lam * 0 - lam * L / 2 * 0
    lv = 0.5 * curv * (tc - L/2)**2 - lam * tc             # tangent lines of F(tau)=curv/2 (tau-L/2)^2
    av = -math.pi * tc / p
    x = np.concatenate([lam, lv, av])
    if rng is not None and jit:
        x = x + jit * rng.standard_normal(x.shape)
    return x

if __name__ == "__main__":
    p, n = int(sys.argv[1]), int(sys.argv[2])
    ms = [int(v) for v in sys.argv[3].split(",")]
    trials = int(sys.argv[4]) if len(sys.argv) > 4 else 6
    b = float(sys.argv[5]) if len(sys.argv) > 5 else 2.0
    rng = np.random.default_rng(0)
    for m in ms:
        best = -1e9; bx = None
        for tr in range(trials):
            curv = [0.5, 1.0, 2.0, 4.0, 0.25, 8.0][tr % 6] * (p / ((p - 1) * n / m)) ** 1 / 3
            x0 = init(m, p, n, curv, rng, 0.05 * (tr // 6))
            r, x = optimise(x0, m, p, n, b)
            if r > best: best, bx = r, x
        print("p=%d n=%d m=%d nu=%d  best 1-D efficiency %.5f" % (p, n, m, n + 1, best), flush=True)
        np.save("oned_p%d_n%d_m%d.npy" % (p, n, m), bx)
