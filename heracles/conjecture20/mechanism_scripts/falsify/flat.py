# Flatness of the oscillation term on a support S (unit amplitudes, all of S cheap):
#   c_flat(S) = min_theta max_delta  O_delta / sum_{l in S} kappa_l(delta),   O_delta = osc_w D_delta / 2
#   lam(S)    = value of max_delta O_delta / sqrt(sum kappa)   at the optimal theta  (Parseval: lam >= 1)
# Also the coverage quality of the uniform allocation on S:  g(S) = |S|^{1-s} min_delta mean_l kappa / Gamma.
# A uniform all-cheap allocation on S (v = cap) has ratio <= g(S) * (1 + c cap - cap c_flat(S))  (up to the delta mismatch).
import numpy as np, torch, sys, time, math, json
from sympy import legendre_symbol
torch.set_default_dtype(torch.float64); torch.set_num_threads(1)
from gget import gamma_table

def cflat(p, S, th0, stages=(0.03, 0.01, 0.003, 0.001), iters=400):
    F = (p - 1) // 2; S = np.asarray(S); w = np.arange(p); d = np.arange(1, F + 1)
    ang = 2 * np.pi * np.outer(w, S + 1) / p
    C = torch.tensor(np.cos(ang)); Sn = torch.tensor(np.sin(ang))
    K = torch.tensor(1 - np.cos(2 * np.pi * np.outer(d, S + 1) / p)); cov = K.sum(1)
    idx = torch.tensor((w[None, :] - d[:, None]) % p)
    th = torch.tensor(th0.astype(float)).requires_grad_()
    def obj(th, tau):
        ph = C @ torch.cos(th) - Sn @ torch.sin(th)
        D = ph[None, :] - ph[idx]
        if tau is None: o = (D.max(1).values - D.min(1).values) / 2
        else:
            sc = len(S); o = tau * sc * (torch.logsumexp(D / (tau * sc), 1) + torch.logsumexp(-D / (tau * sc), 1)) / 2
        return o
    for tau in stages:
        opt = torch.optim.LBFGS([th], lr=1, max_iter=iters, line_search_fn='strong_wolfe', tolerance_grad=1e-12, tolerance_change=1e-14)
        def closure():
            opt.zero_grad(); r = obj(th, tau) / cov; t2 = tau / 2
            loss = t2 * torch.logsumexp(r / t2, 0); loss.backward(); return loss
        try: opt.step(closure)
        except Exception: pass
    with torch.no_grad():
        o = obj(th, None); r = o / cov; lam = (o / torch.sqrt(cov)).max().item()
    return r.max().item(), lam, th.detach().numpy(), cov.numpy()

if __name__ == '__main__':
    s = 1.5; cap = 3 ** 0.5; CS = 1 - (2 / 3) ** 0.5; CA = 1 / (2 * 3 ** 0.5)
    ps = [int(a) for a in sys.argv[1].split(',')]; mults = [float(a) for a in sys.argv[2].split(',')]; trials = int(sys.argv[3])
    GT = gamma_table(); out = []
    for p in ps:
        F = (p - 1) // 2; L = math.log(p); G = GT[p]['Gamma'] if p in GT else None
        rng = np.random.default_rng(p)
        for m in mults:
            K = min(F, int(round(m * L)))
            for t in range(trials):
                # random S, keep the best-coverage of 30 draws
                best = None
                for q in range(30):
                    S = np.sort(rng.choice(F, K, replace=False))
                    cv = (1 - np.cos(2 * np.pi * np.outer(np.arange(1, F + 1), S + 1) / p)).sum(1).min() / K
                    if best is None or cv > best[0]: best = (cv, S)
                cv, S = best
                res = []
                for ph0 in ['rand', 'rand', 'leg']:
                    if ph0 == 'leg':
                        chi = np.array([legendre_symbol(int(l) + 1, p) for l in S])
                        th0 = np.where(chi > 0, 0.0, np.pi) if p % 4 == 1 else -chi * np.pi / 2
                    else: th0 = rng.uniform(0, 2 * np.pi, K)
                    res.append(cflat(p, S, th0))
                cf, lam, th, cov = min(res, key=lambda z: z[0])
                g = K ** (1 - s) * cv / G if G else float('nan')
                rs = g * (1 + CS * cap - cap * cf); ra = g * (1 + CA * cap - cap * cf)
                print(f'p={p:5d} K={K:3d} (K/lnp={K / L:.2f}) mincov={cv:.3f} g={g:.3f}  c_flat={cf:.4f} lam={lam:.3f} '
                      f' ratio-bound sharp={rs:.3f} thmA={ra:.3f}', flush=True)
                out.append(dict(p=p, K=K, cov=cv, g=g, cflat=cf, lam=lam))
    json.dump(out, open(f'flat_{sys.argv[1].replace(",", "_")}_{sys.argv[2].replace(",", "_")}.json', 'w'))
