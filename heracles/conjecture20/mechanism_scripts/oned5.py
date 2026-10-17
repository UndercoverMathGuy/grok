import math, sys, time, numpy as np, torch
sys.path.insert(0, '.')
from oned import eff
from oned3 import lamstar
from rw import logZ
from oned4 import opt
torch.set_default_dtype(torch.float64); torch.set_num_threads(1)

def struct(m, p, n, ctr, tclip=0.02):
    lam = np.array([lamstar(min(max(c / n, tclip), p - 1 - tclip), p) for c in ctr])
    lv = -np.array([n * logZ(l, p) for l in lam]) - 0.5 * math.log(p / 2)
    return lam, lv

def angeff(av, lam, lv, p, n):
    L = (p - 1) * n; th = math.pi / p
    tau = torch.arange(L + 1, dtype=torch.float64)
    E = lv[None, :] + lam[None, :] * tau[:, None]
    Em = E.max(1, keepdim=True).values
    M = torch.exp(E - Em)
    ph = av[None, :] + th * tau[:, None]
    re = (M * torch.cos(ph)).sum(1); im = (M * torch.sin(ph)).sum(1)
    env = M.sum(1)
    return (re - im.abs() / math.tan(th)) / env

def phase_search(lam, lv, p, n, rng, R=50, steps=1500):
    lam_t, lv_t = torch.tensor(lam), torch.tensor(lv)
    best = (-9, None)
    for k in range(R):
        av = torch.tensor(rng.uniform(-math.pi, math.pi, len(lam)), requires_grad=True)
        o = torch.optim.Adam([av], lr=0.03)
        for it in range(steps):
            t = 0.05 if it < steps // 2 else 0.01
            o.zero_grad(); g = angeff(av, lam_t, lv_t, p, n)
            loss = t * torch.logsumexp(-g / t, 0); loss.backward(); o.step()
        with torch.no_grad():
            v = angeff(av, lam_t, lv_t, p, n).min().item()
        if v > best[0]: best = (v, av.detach().numpy().copy())
    return best

if __name__ == "__main__":
    p, n = int(sys.argv[1]), int(sys.argv[2]); ms = [int(v) for v in sys.argv[3].split(",")]
    R = int(sys.argv[4]); b = float(sys.argv[5]) if len(sys.argv) > 5 else 2.0
    rng = np.random.default_rng(int(sys.argv[6]) if len(sys.argv) > 6 else 0); t0 = time.time()
    L = (p - 1) * n
    for m in ms:
        out = []
        for trial in range(3):
            ctr = np.linspace(0, L, m)
            if trial: ctr[1:-1] += rng.uniform(-0.2, 0.2, m - 2) * L / (m - 1)
            lam, lv = struct(m, p, n, ctr)
            a, av = phase_search(lam, lv, p, n, rng, R)
            x = np.concatenate([lam, lv, av])
            r = -9
            if a > 0:
                r, x = opt(x, m, p, n, b, (1e-2, 3e-3, 1e-3, 3e-4, 1e-4), 2000)
            out.append((r, a))
            np.save("o5_p%d_n%d_m%d_t%d.npy" % (p, n, m, trial), x)
        print("p=%d n=%d m=%d nu=%d  (eff, angular) per trial: %s  (%.0fs)" % (p, n, m, n + 1, [(round(r, 4), round(a, 3)) for r, a in out], time.time() - t0), flush=True)
