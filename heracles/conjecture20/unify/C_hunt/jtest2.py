# theta*(p,n) = inf over class-C fluctuation designs (S = F, characters 2l*1_T) of the PAIRED detection ratio
#   max_{sign} (-min_x D(x, sign*1)) / sum_{l,T} |a_{l,T}| |c_l(1)|            (delta0 = 1 suffices by dilation)
# Inner problem (phases psi fixed): LP in the magnitudes r.  Outer: minimise rho(psi) over psi by projected gradient
# descent using the LP duals (envelope theorem), many random restarts.  theta* = rho*/2.
# Usage: python jtest2.py p n nrestarts seed [cplx]
import sys, itertools, json, time
import numpy as np
from scipy.optimize import linprog

p = int(sys.argv[1]); n = int(sys.argv[2]); nr = int(sys.argv[3]); seed = int(sys.argv[4])
cplx = len(sys.argv) > 5 and sys.argv[5] == 'cplx'
F = (p - 1) // 2; rng = np.random.default_rng(seed); d0 = 1
X = np.indices((p,) * n).reshape(n, -1).T
Ts = [T for T in itertools.product([0, 1], repeat=n) if any(T)]
cols = [(l, T) for l in range(1, F + 1) for T in Ts]; nc = len(cols)
chi = np.array([np.exp(2j * np.pi * 2 * l * (X @ np.array(T)) / p) for (l, T) in cols])        # (nc, N)
signs = (d0, p - d0)
cv = [np.array([1 - np.exp(-2j * np.pi * l * d / p) for (l, T) in cols]) for d in signs]
w = np.abs(cv[0]) / 2
kap0 = np.array([1 - np.cos(2 * np.pi * l * d0 / p) for l in range(1, F + 1)])
cl = [np.array([1 - np.exp(-2j * np.pi * l * d / p) for l in range(1, F + 1)]) for d in signs]
N = X.shape[0]


def lp(psi):
    E = [np.exp(1j * psi[:nc])[:, None] * cv[q][:, None] * chi for q in range(2)]             # (nc, N) complex
    A = np.vstack([-np.real(E[q]).T for q in range(2)])                                       # rows (2N, nc)
    nA = 0
    if cplx:
        al = psi[nc:]; nA = F
        dev = [np.real(np.exp(1j * al) * cl[q]) - kap0 for q in range(2)]
        A = np.hstack([A, np.vstack([np.tile(-dev[q], (N, 1)) for q in range(2)])])
    nv = nc + nA
    res = linprog(np.r_[np.zeros(nv), 1.0], A_ub=np.hstack([A, -np.ones((2 * N, 1))]), b_ub=np.zeros(2 * N),
                  A_eq=np.r_[w, np.zeros(nA), 0.0][None, :], b_eq=[1.0], bounds=[(0, None)] * nv + [(None, None)],
                  method='highs')
    lam = -res.ineqlin.marginals                                                              # >= 0
    r = res.x[:nv]
    # d rho / d psi_k = sum_rows lam_row * r_k * dA_{row,k}/dpsi_k ; dA/dpsi = -Re(i E) = Im(E)
    g = np.zeros(len(psi))
    for q in range(2):
        g[:nc] += r[:nc] * (np.imag(E[q]) @ lam[q * N:(q + 1) * N])
    if cplx:
        for q in range(2):
            ddev = np.real(1j * np.exp(1j * psi[nc:]) * cl[q])                               # d dev / d alpha
            g[nc:] += r[nc:] * (-ddev) * lam[q * N:(q + 1) * N].sum()
    return res.fun, r, g


t0 = time.time(); best = (np.inf, None, None); vals = []
npar = nc + (F if cplx else 0)
for it in range(nr):
    psi = rng.uniform(0, 2 * np.pi, npar) if it else np.zeros(npar)
    v, r, g = lp(psi); step = 0.3
    for k in range(300):
        psn = psi - step * g / (np.linalg.norm(g) + 1e-15)
        vn, rn, gn = lp(psn)
        if vn < v - 1e-13: psi, v, r, g = psn, vn, rn, gn; step *= 1.3
        else:
            step *= 0.5
            if step < 1e-7: break
    vals.append(v)
    if v < best[0]: best = (v, psi.copy(), r.copy())
v, psi, r = best
print(json.dumps(dict(p=p, n=n, cplx=cplx, restarts=nr, rho_star=v, theta_star=v / 2, median_restart=float(np.median(vals) / 2),
                      nused=int((r[:nc] > 1e-9).sum()), used_features=sorted(set(cols[k][0] for k in range(nc) if r[k] > 1e-9)),
                      cplx_A=(np.round(r[nc:], 4).tolist() if cplx else None), time=round(time.time() - t0, 1))), flush=True)
np.savez(f"out/jtest2_p{p}_n{n}{'_cplx' if cplx else ''}.npz", psi=psi, r=r, rho=v, p=p, n=n)
