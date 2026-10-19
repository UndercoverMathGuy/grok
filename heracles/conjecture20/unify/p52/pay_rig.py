"""Rigorous (interval) lower bounds on the IF-price amplitude P_1 = 3^{1-s} Q^s at s = 3/2, for boxes of own data
(|A| = 1, |B| in [m1], |C| in [m2], |D| in [m12], cos(phi_pay) in [cp]), phi_pay = arg(B C conj(D) conj(A)).

  P_lb   : max( 1 - c_1 M,  1 - R_T/(2 sqrt 3),  3^{-1/2} max(1, M) )   (Note Thm c1sharp; Part I Thm A; trivial)
  TW     : decoupled twisted Rotfel'd bound  min_L  L (Smin(Pbar(L), qlo(L))/3)^{3/2}   (see p52 notes, Lemma T)
All arithmetic is outward-rounded interval arithmetic (iv.py).
"""
import numpy as np
from iv import I, dn, up, pow23, SQ3, imax

R23 = 2.0 / 3.0


def _p_of(t, P, q):
    """interval enclosure of t^3 - P t - q at float arrays t, P, q"""
    T = I(t)
    return T * T * T - I(P) * T - I(q)


def S_lb_point(P, q):
    """rigorous lower bound of sum_k (1+mu_k)^{2/3}, mu_k the roots of mu^3 - P mu - q, at float points (P, q)
    assumed to have three real roots (27 q^2 <= 4 P^3)."""
    P = np.asarray(P, float); q = np.asarray(q, float)
    Pp = np.maximum(P, 0.0)
    s3 = (I(Pp) / 3.0).sqrt()                      # sqrt(P/3)
    # float approximations of the roots (trigonometric form)
    m = 2 * np.sqrt(Pp / 3)
    with np.errstate(divide='ignore', invalid='ignore'):
        arg = np.where(Pp > 1e-300, 3 * q / (Pp * m), 0.0)
    th = np.arccos(np.clip(arg, -1, 1)) / 3
    r3 = m * np.cos(th)                            # largest
    r1 = m * np.cos(th - 2 * np.pi / 3)            # smallest  (cos(th - 2pi/3) <= ... ) order check below
    r2 = m * np.cos(th + 2 * np.pi / 3)
    roots = np.sort(np.stack([r1, r2, r3]), axis=0)
    r1, r2, r3 = roots[0], roots[1], roots[2]
    eps = 1e-9 * (1 + m)
    # largest root: t3 = r3 - eps, need t3 >= sqrt(P/3) and p(t3) < 0
    t3 = r3 - eps
    ok3 = (t3 >= s3.hi) & (_p_of(t3, P, q).hi < 0)
    lb3 = np.where(ok3, t3, s3.lo)
    # middle root: t2 = r2 - eps in [-sqrt(P/3), sqrt(P/3)], p(t2) > 0
    t2 = r2 - eps
    ok2 = (t2 >= -s3.lo) & (t2 <= s3.lo) & (_p_of(t2, P, q).lo > 0)
    lb2 = np.where(ok2, t2, -s3.hi)
    # smallest root: t1 = r1 - eps <= -sqrt(P/3), p(t1) < 0
    t1 = r1 - eps
    ok1 = (t1 <= -s3.hi) & (_p_of(t1, P, q).hi < 0)
    lb1 = np.where(ok1, t1, dn(-2.0 * s3.hi))
    tot = pow23(I(1.0) + I(lb1)).lo + pow23(I(1.0) + I(lb2)).lo + pow23(I(1.0) + I(lb3)).lo
    return dn(dn(tot))


def LB_Smin(Pbar, qlo):
    """rigorous lower bound of min{ S(P',q') : (P',q') feasible, P' <= Pbar, q' >= qlo } (vectorised).
    Feasible set K: 0<=P<=3, max(P-1, -2(P/3)^{3/2}) <= q <= 2(P/3)^{3/2}.  Returns +inf where provably empty."""
    Pc = np.minimum(np.asarray(Pbar, float), 3.0)
    Pc = np.maximum(Pc, 0.0)
    qlo = np.asarray(qlo, float)
    PcI = I(Pc)
    dh = (PcI / 3.0).sqrt() * (PcI / 3.0) * 2.0    # disc_hi = 2 (P/3)^{3/2}
    infeasible = qlo > dh.hi
    res = np.full(Pc.shape, -np.inf)
    # PSD-boundary candidate (P >= 3/4): eigenvalues 0, (3 +- sqrt(4P-3))/2
    big = Pc >= 0.75
    rt = (PcI * 4.0 - 3.0).clip0().sqrt()
    e1 = (I(3.0) + rt) / 2.0; e2 = (I(3.0) - rt) / 2.0
    sb = dn(pow23(e1).lo + pow23(e2).lo)
    res = np.where(big, np.maximum(res, sb), res)
    # discriminant-boundary candidate (P <= 3/4): roots sqrt(P/3) (double), -2 sqrt(P/3)
    small = Pc <= 0.75
    s3 = (PcI / 3.0).sqrt()
    sd = dn(2.0 * pow23(I(1.0) + s3).lo + pow23(I(1.0) - s3 * 2.0).lo)
    res = np.where(small, np.maximum(res, sd), res)
    # interior candidate at (Pc, qlo) when (Pc, qlo) is provably feasible
    lowb = imax(PcI - 1.0, -dh)
    feas = (qlo >= lowb.hi) & (qlo <= dh.lo)
    if np.any(feas):
        si = np.full(Pc.shape, -np.inf)
        idx = np.where(feas)[0] if Pc.ndim == 1 else np.where(feas)
        si_vals = S_lb_point(Pc[idx], qlo[idx])
        si[idx] = si_vals
        res = np.maximum(res, si)
    res = np.where(infeasible, np.inf, res)
    return res


def tw_bound(m_lo, m_hi, cp_lo, cp_hi, nL=56):
    """rigorous lower bound of the decoupled twisted payment over boxes.
    m_lo, m_hi: arrays (nbox, 3) for |B|, |C|, |D|; cp_lo, cp_hi: bounds on cos(phi_pay).  |A| = 1."""
    m_lo = np.atleast_2d(m_lo); m_hi = np.atleast_2d(m_hi)
    nb = m_lo.shape[0]
    Lmin = np.maximum(1.0, m_lo.max(axis=1))
    tgrid = np.concatenate([[0.0], np.geomspace(1e-7, 1.0, nL - 1)])   # L = Lmin (1 + t)
    best = np.full(nb, np.inf)
    B = I(m_lo[:, 0], m_hi[:, 0]); C = I(m_lo[:, 1], m_hi[:, 1]); D = I(m_lo[:, 2], m_hi[:, 2])
    CP = I(cp_lo, cp_hi)
    for j in range(len(tgrid) - 1):
        La = I(Lmin) * (1.0 + tgrid[j]); Lb = I(Lmin) * (1.0 + tgrid[j + 1])
        L = I(La.lo, Lb.hi)
        rho = (L * (L - 1.0) * 2.0).clip0().sqrt()                # sqrt(2 L (L - |A|)), |A| = 1
        Dm = I(np.minimum((D + rho).hi, L.hi))
        num = B.sq() + C.sq() + Dm.sq()
        Pbar = np.minimum(up(num.hi / L.lo ** 2 * (1 + 4e-16)), 3.0)
        qI = (B * C * (D * CP - rho) * 2.0) / (L * L * L)
        qlo = qI.lo
        S = LB_Smin(Pbar, qlo)
        # value lower bound: L.lo * (S/3)^{3/2}
        Sq = I(np.where(np.isfinite(S), np.maximum(S, 0.0), 0.0)) / 3.0
        val = (I(L.lo) * (Sq * Sq.sqrt())).lo
        val = np.where(np.isinf(S), np.inf, val)
        best = np.minimum(best, val)
    # beyond the last L: P_1 >= 3^{-1/2} L >= 3^{-1/2} * 2 Lmin
    tail = (I(2.0) * I(Lmin) / SQ3).lo
    best = np.minimum(best, tail)
    return best


def plb_bound(m_lo, m_hi):
    m_lo = np.atleast_2d(m_lo); m_hi = np.atleast_2d(m_hi)
    c1 = I(1.0) - I(1.0) / SQ3
    Mhi = m_hi.max(axis=1); Mlo = m_lo.max(axis=1)
    b1 = (I(1.0) - c1 * I(Mhi)).lo
    RT = (I(m_hi[:, 0]).sq() + I(m_hi[:, 1]).sq() + I(m_hi[:, 2]).sq()).sqrt()
    b2 = (I(1.0) - RT / (SQ3 * 2.0)).lo
    b3 = (I(np.maximum(1.0, Mlo)) / SQ3).lo
    return np.maximum(np.maximum(b1, b2), b3)


if __name__ == '__main__':
    from tw2 import pay_tw2
    rng = np.random.default_rng(0)
    # point boxes: compare with the float decoupled payment
    for k in range(10):
        m = rng.uniform(0.3, 1.3, 3); cp_ = rng.uniform(-1, 1)
        rig = tw_bound(m[None, :], m[None, :], np.array([cp_]), np.array([cp_]), nL=200)[0]
        fl = pay_tw2(1.0, m[0], m[1], m[2], cp_, 1.5, nL=400)
        print(f"m={np.round(m,3)} cos={cp_:.3f}: rigorous {rig:.4f}  float {fl:.4f}  P_lb {plb_bound(m[None,:], m[None,:])[0]:.4f}")
    print("single atom box:", tw_bound(np.array([[1, 1, 1]]), np.array([[1, 1, 1]]), np.array([1.0]), np.array([1.0]), nL=200))
    print("IF box:", tw_bound(np.array([[0, 0, 0]]), np.array([[0, 0, 0]]), np.array([1.0]), np.array([1.0])))
