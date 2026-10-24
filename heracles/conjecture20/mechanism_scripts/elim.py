# Backward elimination over a dictionary of efficient tilts; exact LP (HiGHS) for coefficients at each step.
import math, sys, time, numpy as np, torch
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, vstack, hstack, csr_matrix
sys.path.insert(0, '.')
from oned import eff, lref
from oned3 import lamstar
from rw import logZ
from cont1d import polish

K = 16
def lp(lam, p, n, w):
    """max c s.t. cone constraints, sum w_j |V~_j| <= 1 (polygonal); V~ = cost-amplitude-normalised coefficient"""
    J = len(lam); L = (p - 1) * n; th = math.pi / p; cot = 1 / math.tan(th)
    tau = np.arange(L + 1)
    lz = np.array([logZ(l, p) for l in lam])
    logG = lam[None, :] * tau[:, None] - n * lz[None, :] - 0.5 * math.log(p / 2)
    colmax = logG.max(0); logG = logG - colmax[None, :]          # column scaling: X_j = V~_j e^{colmax_j}
    rowmax = logG.max(1); G = np.exp(logG - rowmax[:, None])    # row scaling
    C, S = np.cos(th * tau)[:, None], np.sin(th * tau)[:, None]
    ReX, ReY, ImX, ImY = G * C, -G * S, G * S, G * C
    crow = np.exp(-rowmax)                                       # coefficient of c after row scaling
    cs = crow.max(); crow = crow / cs                             # work with c' = c*cs = O(1)
    # variables: X (J), Y (J), T (J), c
    A1 = np.hstack([-(ReX - cot * ImX), -(ReY - cot * ImY), np.zeros((L + 1, J)), crow[:, None]])
    A2 = np.hstack([-(ReX + cot * ImX), -(ReY + cot * ImY), np.zeros((L + 1, J)), crow[:, None]])
    ph = 2 * math.pi * np.arange(K) / K
    rows = []
    for k in range(K):
        rows.append(np.hstack([np.cos(ph[k]) * np.eye(J), np.sin(ph[k]) * np.eye(J), -np.eye(J), np.zeros((J, 1))]))
    A3 = np.vstack(rows)
    A4 = np.concatenate([np.zeros(2 * J), w * np.exp(-colmax), [0.0]])[None, :]
    A = np.vstack([A1, A2, A3, A4]); bvec = np.concatenate([np.zeros(2 * (L + 1) + K * J), [1.0]])
    cobj = np.zeros(3 * J + 1); cobj[-1] = -1
    bounds = [(None, None)] * (2 * J) + [(0, None)] * J + [(None, None)]
    res = linprog(cobj, A_ub=A, b_ub=bvec, bounds=bounds, method="highs", options={"primal_feasibility_tolerance": 1e-10, "dual_feasibility_tolerance": 1e-10})
    if res.status != 0: return None
    z = res.x; X, Y = z[:J], z[J:2 * J]
    Vt = (X + 1j * Y) * np.exp(-colmax)                           # cost-amplitude-normalised coefficients
    return z[-1] / cs, Vt, lz

def to_params(lam, Vt, lz, p, n, keep):
    V = Vt[keep] / (math.sqrt(p / 2) * np.exp(n * lz[keep]))
    return np.concatenate([lam[keep], np.log(np.abs(V)), np.angle(V)])

def true_eff(lam, Vt, lz, p, n, b, keep):
    x = to_params(lam, Vt, lz, p, n, keep)
    return eff(torch.tensor(x), int(keep.sum()), p, n, b).min().item(), x

if __name__ == "__main__":
    p, n = int(sys.argv[1]), int(sys.argv[2]); b = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
    J = int(sys.argv[4]) if len(sys.argv) > 4 else 160
    mstop = int(sys.argv[5]) if len(sys.argv) > 5 else 3
    nu = n + 1; s = nu / b; r = 1 / s; t0 = time.time()
    tg = np.linspace(0.003, p - 1 - 0.003, J)
    lam = np.array([lamstar(t, p) for t in tg])
    active = np.ones(J, bool); w = np.ones(J)
    log = []
    while active.sum() >= mstop:
        idx = np.where(active)[0]
        out = lp(lam[idx], p, n, w[idx])
        if out is None or out[0] <= 0: print('infeasible at m=%d' % len(idx)); break
        c, Vt, lz = out
        a = np.abs(Vt)
        keep = a > 1e-9 * a.max()
        e, x = true_eff(lam[idx], Vt, lz, p, n, b, keep)
        msupp = int(keep.sum())
        log.append((len(idx), msupp, e, x))
        print("active %3d support %3d  true eff %.5f  (%.0fs)" % (len(idx), msupp, e, time.time() - t0), flush=True)
        # drop zero-coefficient atoms, then the weakest atom; reweight toward l_r
        newactive = np.zeros(J, bool); newactive[idx[keep]] = True
        if newactive.sum() == active.sum():
            j = idx[keep][np.argmin(a[keep])]; newactive[j] = False
        active = newactive
        w = np.ones(J); w[idx[keep]] = (a[keep] / a[keep].max() + 1e-3) ** (r - 1)
    best = max(log, key=lambda t: t[2])
    print("best before polish: support %d eff %.5f" % (best[1], best[2]))
    # polish the few best supports
    for ln in sorted(log, key=lambda t: -t[2])[:4]:
        rr, xx = polish(ln[3], ln[1], p, n, b)
        print("  polish support %d: %.5f -> %.5f" % (ln[1], ln[2], rr), flush=True)
        np.save("elim_p%d_n%d_m%d.npy" % (p, n, ln[1]), xx)
