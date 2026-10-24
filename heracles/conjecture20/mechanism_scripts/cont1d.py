# Continuation in n inside the 1-D shared-sign family; tries same m and m+1 (term insertion) at each step.
import math, sys, time, numpy as np, torch
sys.path.insert(0, '.')
from oned import eff
from oned3 import lamstar
from rw import logZ
from oned5 import angeff, phase_search
torch.set_default_dtype(torch.float64); torch.set_num_threads(1)

def polish(x, m, p, n, b, iters=2500):
    P = torch.tensor(x).clone().requires_grad_()
    with torch.no_grad(): g0 = eff(P, m, p, n, b)
    g0m = g0.min().item()
    if g0m > 0:
        temps = [t * g0m for t in (1e-2, 3e-3, 1e-3, 3e-4, 1e-4)]
    else:
        sc = max(abs(g0m), 1e-3)
        temps = sorted(set([t for t in (0.3 * sc, 0.1 * sc, 0.03 * sc, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4) if t <= 0.3 * sc or t <= 1e-2]), reverse=True)
    for t in temps:
        o = torch.optim.LBFGS([P], lr=1, max_iter=iters, line_search_fn="strong_wolfe", tolerance_grad=1e-16, tolerance_change=1e-18)
        def closure():
            o.zero_grad(); g = eff(P, m, p, n, b); loss = t * torch.logsumexp(-g / t, 0); loss.backward(); return loss
        try: o.step(closure)
        except Exception: pass
    with torch.no_grad():
        return eff(P, m, p, n, b).min().item(), P.detach().numpy().copy()

def centres(x, m, p, n):
    lam, lv = x[:m], x[m:2*m]
    L = (p - 1) * n; tau = np.arange(L + 1)
    E = lv[None, :] + lam[None, :] * tau[:, None]
    dom = E.argmax(1)
    c = np.array([tau[dom == i].mean() if np.any(dom == i) else np.nan for i in range(m)])
    # undominated terms: place by slope order
    o = np.argsort(lam)
    for i in range(m):
        if np.isnan(c[i]):
            c[i] = np.interp(np.searchsorted(lam[o], lam[i]), [0, m - 1], [0, L])
    return c

def stretch(x, m, p, n_old, n_new):
    L, L2 = (p - 1) * n_old, (p - 1) * n_new
    th = math.pi / p; k = L / L2
    lam, lv, av = x[:m] * k, x[m:2*m].copy(), x[2*m:].copy()
    c2 = centres(x, m, p, n_old) / k
    av = av - th * c2 * (1 - k)
    return np.concatenate([lam, lv, av])

def insert(x, m, p, n, rng, ph):
    lam, lv, av = x[:m], x[m:2*m], x[2*m:]
    c = centres(x, m, p, n); o = np.argsort(c)
    cs = np.concatenate([[0.0], c[o], [(p - 1) * n]])
    gaps = np.diff(cs); j = int(np.argmax(gaps))
    cn = 0.5 * (cs[j] + cs[j + 1])
    ln = lamstar(min(max(cn / n, 0.02), p - 1.02), p)
    # magnitude: match the envelope of current terms at cn
    En = lv + lam * cn
    lvn = En.max() - ln * cn
    return np.concatenate([np.append(lam, ln), np.append(lv, lvn), np.append(av, ph)])

if __name__ == "__main__":
    p, n0, n1 = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    b = float(sys.argv[4]) if len(sys.argv) > 4 else 2.0
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    tag = sys.argv[6] if len(sys.argv) > 6 else ""
    rng = np.random.default_rng(seed); t0 = time.time()
    th = math.pi / p
    # seed at n0: Descartes-minimal m and m+1, random phase search with structural slopes
    pool = []
    L = (p - 1) * n0
    mmin = int(math.ceil(1 + (th * L - 2 * th) / math.pi - 1e-9))
    for m in [mmin, mmin + 1, mmin + 2, mmin + 3]:
        for trial in range(4):
            ctr = np.linspace(0, L, m)
            if trial: ctr[1:-1] += rng.uniform(-0.25, 0.25, m - 2) * L / max(m - 1, 1)
            lam = np.array([lamstar(min(max(c / n0, 0.03), p - 1.03), p) for c in ctr])
            lv = -np.array([n0 * logZ(l, p) for l in lam])
            a, av = phase_search(lam, lv, p, n0, rng, R=20, steps=800)
            if a <= 0: continue
            r, x = polish(np.concatenate([lam, lv, av]), m, p, n0, b)
            if r > 0: pool.append((r, m, x))
    pool.sort(key=lambda t: -t[0])
    if not pool: print('no seeds'); sys.exit()
    print("p=%d n=%d seeds %s (%.0fs)" % (p, n0, [(round(r, 4), m) for r, m, _ in pool[:6]], time.time() - t0), flush=True)
    keep = {}
    for r, m, x in pool:
        if m not in keep: keep[m] = (r, m, x)
    pool = sorted(keep.values(), key=lambda t: -t[0])[:3]
    for n in range(n0 + 1, n1 + 1):
        cands = []
        for r, m, x in pool:
            xs = stretch(x, m, p, n - 1, n)
            cands.append((m, xs))
            if True:
                for ph in np.linspace(-math.pi, math.pi, 6, endpoint=False):
                    cands.append((m + 1, insert(xs, m, p, n, rng, ph)))
        res = []
        for m, x in cands:
            r, x2 = polish(x, m, p, n, b)
            res.append((r, m, x2))
        res.sort(key=lambda t: -t[0])
        keep = {}
        for r, m, x in res:
            if m not in keep and r > 0: keep[m] = (r, m, x)
        pool = sorted(keep.values(), key=lambda t: -t[0])[:3]
        if not pool:
            print('no feasible continuation at n=%d' % n, flush=True); break
        nu = n + 1; s = nu / b
        print("p=%d n=%d nu=%d  best by m: %s  (%.0fs)" % (p, n, nu, sorted([(m, round(r, 4)) for r, m, _ in pool]), time.time() - t0), flush=True)
        np.save("c1d%s_p%d_n%d.npy" % (tag, p, n), np.concatenate([[pool[0][1]], pool[0][2]]))
