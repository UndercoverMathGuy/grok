"""Referee check of the rigorous payment F (pay_adapt.payment) at s = 3/2.

For own data |A| = 1 (A = 1), |B| = m1, |C| = m2, |D| = m12, cos(phi_pay) = cos(ph) we compute independently:
  (1) Pmin: a numerical minimum of the TRUE payment P_1 = 3^{-1/2} (sum_i |beta_i|^{2/3})^{3/2} over atom systems
      (beta_i in C, xi_i, eta_i in T, K atoms) with sum beta = 1, sum beta xi = m1, sum beta eta = m2,
      sum beta xi eta = m12 e^{i ph}  (SLSQP, many starts).  Soundness requires F_rig <= Pmin.
  (2) Fgrid: my own float evaluation of the three-chain circle bound on a dense (L, psi) grid with the depressed cubic
      solved by numpy.roots, i.e. an UPPER estimate of the bound the code is supposed to lower-bound.
      Soundness requires F_rig <= max(P_lb, Fgrid) (+ tiny).
Also: random atom systems -> own data -> point box -> check F_rig <= P_1(atoms).
"""
import sys, os, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import numpy as np
from scipy.optimize import minimize
from pay_adapt import payment

rng = np.random.default_rng(7)
s = 1.5; r = 1 / s


def P1_of(beta):
    return 3 ** (1 - s) * np.sum(np.abs(beta) ** r) ** s


def own(beta, xi, eta):
    return np.array([beta.sum(), (beta * xi).sum(), (beta * eta).sum(), (beta * xi * eta).sum()])


def min_P1(m1, m2, m12, ph, K=6, starts=25):
    target = np.array([1.0, m1, m2, m12 * np.exp(1j * ph)])
    best = np.inf

    def unpack(z):
        rho = z[:K] ** 2; a = z[K:2 * K]; b = z[2 * K:3 * K]; c = z[3 * K:4 * K]
        return rho * np.exp(1j * a), np.exp(1j * b), np.exp(1j * c)

    def obj(z):
        return np.sum((z[:K] ** 2 + 1e-12) ** r)

    def cons(z):
        be, xi, eta = unpack(z)
        d = own(be, xi, eta) - target
        return np.concatenate([d.real, d.imag])
    for st in range(starts):
        z0 = np.concatenate([rng.uniform(0.2, 1.0, K), rng.uniform(-np.pi, np.pi, 3 * K)])
        if st == 0:   # aligned start
            z0[K:] = 0.0
        res = minimize(obj, z0, constraints=[{'type': 'eq', 'fun': cons}], method='SLSQP',
                       options=dict(maxiter=500, ftol=1e-12))
        if np.max(np.abs(cons(res.x))) < 1e-8:
            be, _, _ = unpack(res.x)
            best = min(best, P1_of(be))
    return best


def plb(m1, m2, m12):
    M = max(m1, m2, m12)
    return max(1 - (1 - 3 ** -0.5) * M, 1 - np.sqrt(m1 ** 2 + m2 ** 2 + m12 ** 2) / (2 * np.sqrt(3)), max(1, M) / np.sqrt(3))


def S_min(P, q):
    """min of sum (1+mu)^{2/3} over K with P' <= P, q' >= q: S(P, max(q, lowK(P))) or inf"""
    P = min(P, 3.0)
    up_ = 2 * (P / 3) ** 1.5
    if q > up_ + 1e-15:
        return np.inf
    low = max(P - 1, -up_)
    qq = min(max(q, low), up_)
    mu = np.roots([1, 0, -P, -qq]).real
    return np.sum(np.maximum(1 + mu, 0) ** r)


def chain_grid(m, ph, nL=300, nps=360, Lspan=1.5):
    Lmin = max(1.0, max(m))
    best_chain = 0
    for (ix, iy, iz) in [(2, 0, 1), (1, 0, 2), (0, 1, 2)]:
        mX, mY, mZ = m[ix], m[iy], m[iz]
        v = np.inf
        for L in Lmin * (1 + np.concatenate([[0], np.geomspace(1e-6, Lspan, nL)])):
            rho = np.sqrt(max(2 * L * (L - 1), 0))
            for psi in np.linspace(-np.pi, np.pi, nps, endpoint=False):
                P = (mY ** 2 + mZ ** 2 + mX ** 2 + rho ** 2 + 2 * mX * rho * np.cos(psi - ph)) / L ** 2
                q = 2 * mY * mZ * (mX * np.cos(ph) + rho * np.cos(psi)) / L ** 3
                v = min(v, L * (S_min(P, q) / 3) ** 1.5)
        v = min(v, Lmin * (1 + Lspan) / np.sqrt(3))
        best_chain = max(best_chain, v)
    return best_chain


if __name__ == '__main__':
    pts = [((1, 1, 1), 0.0), ((0, 0, 0), 0.0), ((0.524, 0.497, 0.635), 0.117), ((0.628, 0.628, 0.628), 0.022),
           ((0.65, 0.55, 0.7), 0.0), ((0.7, 0.5, 0.9), -0.6), ((0.1, 0.1, 1.1), np.pi), ((1.0, 1.0, 0.0), 1.0),
           ((1.2, 0.3, 0.3), 2.0), ((0.5, 0.5, 0.5), np.pi)]
    for _ in range(22):
        pts.append((tuple(np.round(rng.uniform(0, 1.4, 3), 3)), round(rng.uniform(-np.pi, np.pi), 3)))
    worst_gap = np.inf; nviol = 0
    print(" m1    m2    m12   ph     F_rig   P_lb    Fgrid   Pmin(SLSQP)  Pmin-F_rig")
    for (m, ph) in pts:
        m = np.array(m, float)
        if m[0] < m[1]:
            m[[0, 1]] = m[[1, 0]]
        t0 = time.time()
        Fr = payment(m, m, ph, ph)
        Fg = max(plb(*m), chain_grid(m, ph, nL=120, nps=180))
        Pm = min_P1(m[0], m[1], m[2], ph)
        gap = Pm - Fr
        worst_gap = min(worst_gap, gap)
        flag = ''
        if Fr > Fg + 1e-9 or gap < -1e-9:
            nviol += 1; flag = '  <-- VIOLATION'
        print(f"{m[0]:.3f} {m[1]:.3f} {m[2]:.3f} {ph:+.3f}  {Fr:.5f} {plb(*m):.5f} {Fg:.5f} {Pm:.5f}  {gap:+.5f}{flag}")
    print(f"min (Pmin - F_rig) = {worst_gap:.5f};  violations = {nviol}")
    # random atom systems -> own data -> F_rig at the point box <= P_1(atoms)
    nv2 = 0; ratios = []
    for t in range(60):
        K = rng.integers(1, 6)
        be = rng.exponential(1, K) * np.exp(1j * rng.normal(0, rng.choice([0.1, 0.5, 2.0]), K))
        xi = np.exp(1j * rng.normal(0, rng.choice([0.05, 0.5, 3.0]), K)); eta = np.exp(1j * rng.normal(0, rng.choice([0.05, 0.5, 3.0]), K))
        A, B, C, D = own(be, xi, eta)
        if abs(A) < 1e-3:
            continue
        sc = 1 / abs(A)
        m = np.array([abs(B), abs(C), abs(D)]) * sc
        php = np.angle(A * D * np.conj(B * C)) if abs(B * C * D) > 0 else 0.0
        if max(m) > 1.4:
            continue
        mm = m.copy()
        if mm[0] < mm[1]: mm[[0, 1]] = mm[[1, 0]]
        Fr = payment(mm, mm, php, php)
        P1 = P1_of(be * sc)
        ratios.append(P1 / Fr)
        if Fr > P1 * (1 + 1e-12):
            nv2 += 1; print("VIOLATION atoms", K, m, php, Fr, P1)
    print(f"random atom systems: {len(ratios)} tested, violations {nv2}, min P_1/F_rig = {min(ratios):.5f}")
