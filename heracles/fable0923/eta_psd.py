"""eta_psd.py -- the eta-form one-population inequality at aiming depth A (n=2, nu=3).

Population: atoms c_l (complex), slot phases v_l1, v_l2 (unit).  G = sum c_l (clean),
J_1 = sum c_l v_l1, J_2 = sum c_l v_l2, J_12 = sum c_l v_l1 v_l2 (junk), eta = ||J||_2, cost = sum |c_l|^r.

(ETA-A):  cost >= 3^{1-r} (|G| - A*eta)_+^r .

Part 1 (PSD route, following astra/P3_ANALYTIC with delta -> eta): with G = 1, e = sum|c_l| >= 1,
u = sqrt(2e(e-1)), B = 3e^2 + 2eta^2 + 4 eta u + 2u^2, the proved bound is
   cost >= Phi(e,eta) := max( e^r , 3^{1-r} e^{2-r} (3/B)^{1-r} ).
Find A_PSD = inf{A : Phi(e,eta) >= 3^{1-r}(1 - A eta)_+^r for all e>=1, eta>=0}.

Part 2 (truth): maximise R_A(pop) = 3^{1-r}(|G| - A eta)_+^r / cost over populations with N atoms
(and the l1 variant with ||J||_1), bisect A for sup R = 1.
"""
import sys, numpy as np
from scipy.optimize import minimize
r = 2/3
c3 = 3 ** (1 - r)

def phi(e, eta):
    u = np.sqrt(2 * e * (e - 1))
    B = 3 * e**2 + 2 * eta**2 + 4 * eta * u + 2 * u**2
    return np.maximum(e ** r, c3 * e ** (2 - r) * (3 / B) ** (1 - r))

def psd_ok(A, ne=1201, nh=1201):
    e = np.linspace(1, np.sqrt(3) + 1e-9, ne)[:, None]
    eta = np.linspace(0, 1 / A, nh)[None, :]
    lhs = phi(e, eta)
    rhs = c3 * np.clip(1 - A * eta, 0, None) ** r
    gap = lhs - rhs
    i = np.unravel_index(gap.argmin(), gap.shape)
    return gap.min(), float(e[i[0], 0]), float(eta[0, i[1]])

def part1():
    lo, hi = 0.1, 0.5
    for _ in range(40):
        mid = (lo + hi) / 2
        g, e, eta = psd_ok(mid)
        if g >= 0: hi = mid
        else: lo = mid
    print(f"A_PSD in [{lo:.6f}, {hi:.6f}]; worst point at A={hi:.5f}: e={psd_ok(hi)[1]:.4f} eta={psd_ok(hi)[2]:.4f}")
    for A in (0.20, 0.244, 0.25, 0.26, 0.30):
        g, e, eta = psd_ok(A)
        print(f"  A={A:.3f}: min gap {g:+.5f} at e={e:.4f}, eta={eta:.4f}")
    # with the extra constraint eta <= sqrt(3) e (each |J_T| <= e)?  (only matters if it changes A_PSD)
    return hi

def pop(x, N):
    c = x[:N] * np.exp(1j * x[N:2*N]); v1 = np.exp(1j * x[2*N:3*N]); v2 = np.exp(1j * x[3*N:4*N])
    G = c.sum(); J = np.array([(c*v1).sum(), (c*v2).sum(), (c*v1*v2).sum()])
    return G, J, np.abs(c)

def ratio(x, N, A, norm):
    G, J, a = pop(x, N)
    jn = np.linalg.norm(J) if norm == 2 else np.abs(J).sum()
    y = max(abs(G) - A * jn, 0.0)
    cost = (a ** r).sum()
    return c3 * y ** r / max(cost, 1e-300)

def sup_ratio(N, A, norm, starts, rng):
    best = 0.0; bx = None
    for s in range(starts):
        x0 = np.concatenate([rng.random(N) + 0.2, rng.random(3*N) * 2 * np.pi])
        if s % 3 == 0:  # near-block start
            x0[:N] = 1; x0[N:2*N] = 0; x0[2*N:3*N] = 2*np.pi*np.arange(N)/3 + 0.1*rng.standard_normal(N); x0[3*N:] = x0[2*N:3*N]
        res = minimize(lambda x: -ratio(np.concatenate([np.abs(x[:N]), x[N:]]), N, A, norm), x0, method="Nelder-Mead",
                       options=dict(maxiter=6000, xatol=1e-9, fatol=1e-12))
        val = -res.fun
        if val > best: best, bx = val, res.x
    return best, bx

def part2(norm):
    rng = np.random.default_rng(0)
    print(f"\n== truth, ||J||_{norm} form")
    for N in (1, 2, 3, 4):
        lo, hi = 0.05, 0.5
        for _ in range(14):
            mid = (lo + hi) / 2
            v, _ = sup_ratio(N, mid, norm, 24 if N <= 2 else 40, rng)
            if v <= 1 + 1e-9: hi = mid
            else: lo = mid
        v, bx = sup_ratio(N, hi, norm, 40, rng)
        G, J, a = pop(np.concatenate([np.abs(bx[:N]), bx[N:]]), N)
        print(f"  N={N}: threshold A in [{lo:.4f},{hi:.4f}]  (sup R at hi = {v:.6f}); worst pop |c|={np.round(a/abs(G),3)} |J|/|G|={np.round(np.abs(J)/abs(G),3)}")

if __name__ == "__main__":
    part1()
    part2(2)
    part2(1)
