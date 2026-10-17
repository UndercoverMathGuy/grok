# Local copies of the repo's 1-D family tools (elim.lp, oned3.lamstar, rw.logZ, cont1d.polish), unchanged logic.
import os, sys, math, numpy as np, torch
sys.dont_write_bytecode=True
sys.path.insert(0,'..')
from oned import eff
from scipy.optimize import linprog, brentq
torch.set_default_dtype(torch.float64); torch.set_num_threads(1)

def lamstar(t, p):
    k = np.arange(p)
    f = lambda l: (k * np.exp(2*l*k - np.max(2*l*k))).sum() / np.exp(2*l*k - np.max(2*l*k)).sum() - t
    return brentq(f, -60, 60)

def logZ(lam, p):
    k = np.arange(p); z = 2 * lam * k
    return 0.5 * (z.max() + np.log(np.exp(z - z.max()).sum()))

K = 16
def lp(lam, p, n, w):
    J = len(lam); L = (p - 1) * n; th = math.pi / p; cot = 1 / math.tan(th)
    tau = np.arange(L + 1)
    lz = np.array([logZ(l, p) for l in lam])
    logG = lam[None, :] * tau[:, None] - n * lz[None, :] - 0.5 * math.log(p / 2)
    colmax = logG.max(0); logG = logG - colmax[None, :]
    rowmax = logG.max(1); G = np.exp(logG - rowmax[:, None])
    C, S = np.cos(th * tau)[:, None], np.sin(th * tau)[:, None]
    ReX, ReY, ImX, ImY = G * C, -G * S, G * S, G * C
    crow = np.exp(-rowmax); cs = crow.max(); crow = crow / cs
    A1 = np.hstack([-(ReX - cot * ImX), -(ReY - cot * ImY), np.zeros((L + 1, J)), crow[:, None]])
    A2 = np.hstack([-(ReX + cot * ImX), -(ReY + cot * ImY), np.zeros((L + 1, J)), crow[:, None]])
    ph = 2 * math.pi * np.arange(K) / K
    A3 = np.vstack([np.hstack([np.cos(ph[k]) * np.eye(J), np.sin(ph[k]) * np.eye(J), -np.eye(J), np.zeros((J, 1))]) for k in range(K)])
    A4 = np.concatenate([np.zeros(2 * J), w * np.exp(-colmax), [0.0]])[None, :]
    A = np.vstack([A1, A2, A3, A4]); bvec = np.concatenate([np.zeros(2 * (L + 1) + K * J), [1.0]])
    cobj = np.zeros(3 * J + 1); cobj[-1] = -1
    bounds = [(None, None)] * (2 * J) + [(0, None)] * J + [(None, None)]
    res = linprog(cobj, A_ub=A, b_ub=bvec, bounds=bounds, method="highs")
    if res.status != 0: return None
    z = res.x; X, Y = z[:J], z[J:2 * J]
    Vt = (X + 1j * Y) * np.exp(-colmax)
    return z[-1] / cs, Vt, lz

def to_params(lam, Vt, lz, p, n, keep):
    V = Vt[keep] / (math.sqrt(p / 2) * np.exp(n * lz[keep]))
    return np.concatenate([lam[keep], np.log(np.abs(V)), np.angle(V)])

def true_eff(lam, Vt, lz, p, n, b, keep):
    x = to_params(lam, Vt, lz, p, n, keep)
    return eff(torch.tensor(x), int(keep.sum()), p, n, b).min().item(), x

def polish(x, m, p, n, b, iters=2500):
    P = torch.tensor(x).clone().requires_grad_()
    with torch.no_grad(): g0 = eff(P, m, p, n, b)
    g0m = g0.min().item()
    if g0m > 0: temps = [t * g0m for t in (1e-2, 3e-3, 1e-3, 3e-4, 1e-4)]
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
