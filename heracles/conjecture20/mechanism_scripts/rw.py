# Reweighted-l1 SOCP over a dictionary of slopes for the 1-D shared-sign family (single feature, general p, b).
import math, sys, numpy as np, cvxpy as cp, torch
sys.path.insert(0, '.')
from oned import eff, lref, optimise
from oned3 import lamstar

def logZ(lam, p):
    k = np.arange(p); z = 2 * lam * k
    return 0.5 * (z.max() + np.log(np.exp(z - z.max()).sum()))

def solve(p, n, b=2.0, J=240, iters=25, eps_rel=1e-6, tgrid=None, lam_extra=None, verbose=False):
    nu = n + 1; s = nu / b; r = 1 / s; th = math.pi / p
    L = (p - 1) * n; tau = np.arange(L + 1)
    if tgrid is None:
        tgrid = np.linspace(0.004, p - 1 - 0.004, J)
    lam = np.array([lamstar(t, p) for t in tgrid])
    if lam_extra is not None: lam = np.concatenate([lam, lam_extra])
    J = len(lam)
    lz = np.array([logZ(l, p) for l in lam])
    logg = lam[None, :] * tau[:, None] - 0.5 * math.log(p / 2) - n * lz[None, :]   # per-unit-cost contribution
    G = np.exp(logg)
    C, S = np.cos(th * tau), np.sin(th * tau)
    # Re Phi = sum_j G (x C - y S), Im Phi = sum_j G (x S + y C)
    ReX, ReY = G * C[:, None], -G * S[:, None]
    ImX, ImY = G * S[:, None], G * C[:, None]
    cot = 1 / math.tan(th)
    x = cp.Variable(J); y = cp.Variable(J); t = cp.Variable(J); c = cp.Variable()
    w = cp.Parameter(J, nonneg=True)
    cons = [ReX @ x + ReY @ y - cot * (ImX @ x + ImY @ y) >= c,
            ReX @ x + ReY @ y + cot * (ImX @ x + ImY @ y) >= c,
            cp.SOC(t, cp.vstack([x, y]), axis=0) if False else cp.norm(cp.vstack([x, y]), 2, axis=0) <= t,
            w @ t <= 1]
    prob = cp.Problem(cp.Maximize(c), cons)
    wv = np.ones(J); best = (-1e9, None)
    for it in range(iters):
        w.value = wv
        try:
            prob.solve(solver=cp.CLARABEL)
        except Exception as e:
            if verbose: print('solver fail', e); break
        if x.value is None: break
        a = np.sqrt(x.value ** 2 + y.value ** 2)
        Re = (ReX @ x.value + ReY @ y.value); Im = (ImX @ x.value + ImY @ y.value)
        cval = np.min(Re - cot * np.abs(Im))
        P = nu ** (b / 2) * np.sum(a ** r)
        e = cval / (P ** s * math.exp(lref(p, n, b)))
        supp = np.sum(a > 1e-6 * a.max())
        if verbose: print('  it %d  c=%.4g  eff=%.5f  support=%d' % (it, cval, e, supp), flush=True)
        if e > best[0]: best = (e, (lam.copy(), x.value.copy(), y.value.copy(), lz.copy()))
        wv = r * (a + eps_rel * a.max()) ** (r - 1); wv = wv / wv.min()
    return best

def to_params(best, p, n, thresh=1e-6):
    lam, x, y, lz = best[1]
    a = np.sqrt(x ** 2 + y ** 2); keep = a > thresh * a.max()
    V = (x + 1j * y)[keep] / (math.sqrt(p / 2) * np.exp(n * lz[keep]))
    lv = np.log(np.abs(V)); av = np.angle(V)
    return np.concatenate([lam[keep], lv, av]), int(keep.sum())

if __name__ == "__main__":
    p, n = int(sys.argv[1]), int(sys.argv[2])
    b = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
    best = solve(p, n, b, verbose=True)
    par, m = to_params(best, p, n)
    e0 = eff(torch.tensor(par), m, p, n, b).min().item()
    r1, x1 = optimise(par, m, p, n, b, stages=(1e-3, 3e-4, 1e-4))
    print("p=%d n=%d b=%.2f: dictionary eff %.5f (m=%d, nu=%d) -> polished %.5f" % (p, n, b, best[0], m, n + 1, r1), flush=True)
    np.save("rw_p%d_n%d.npy" % (p, n), x1)
