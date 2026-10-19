"""Rigorous circle-coupled three-chain twisted payment over boxes (interval arithmetic), s = 3/2, |A| = 1.

Box: |B| in [m1], |C| in [m2], |D| in [m12], phi_pay in [p_lo, p_hi] (phi_pay = arg(alpha D) - arg(B C), alpha = A/|A|).
For each chain (uncertain entry X in {B, C, D}; Y, Z the exact ones) and each cell (L-interval x psi-interval):
   P <= Phi = (mY^2 + mZ^2 + mX^2 + rho^2 + 2 mX rho cos(psi - phi))/L^2   (capped at 3)
   q >= qlo = 2 mY mZ (mX cos(phi) + rho cos(psi)) / L^3
   value >= L_lo (LB_Smin(Phi, qlo)/3)^{3/2}.
payment >= max over chains of min over cells (and of the tail L >= Lcap: 3^{-1/2} L).
Fully vectorised over (box, L-cell, psi-cell).
"""
import numpy as np
from iv import I, dn, up, cos_iv, SQ3
from pay_rig import LB_Smin, plb_bound


def tw3_bound(m_lo, m_hi, p_lo, p_hi, nL=48, npsi=48, Lspan=1.5, tgrid=None):
    m_lo = np.atleast_2d(np.asarray(m_lo, float)); m_hi = np.atleast_2d(np.asarray(m_hi, float))
    p_lo = np.atleast_1d(np.asarray(p_lo, float)); p_hi = np.atleast_1d(np.asarray(p_hi, float))
    nb = m_lo.shape[0]
    Lmin = np.maximum(1.0, m_lo.max(axis=1))
    if tgrid is None:
        tgrid = Lspan * (np.arange(nL + 1) / nL) ** 2                      # nL cells, uniform in sqrt(L-1)
    nL = len(tgrid) - 1
    Lspan = tgrid[-1]
    psig = np.linspace(-np.pi, np.pi, npsi + 1)
    sh = (nb, nL, npsi)
    B3 = lambda a: np.broadcast_to(np.asarray(a, float).reshape(-1, 1, 1), sh)
    La = (I(B3(Lmin)) * I(np.broadcast_to(1.0 + tgrid[:-1].reshape(1, -1, 1), sh))).lo
    Lb = (I(B3(Lmin)) * I(np.broadcast_to(1.0 + tgrid[1:].reshape(1, -1, 1), sh))).hi
    L = I(La, Lb)
    rho = (L * (L - 1.0) * 2.0).clip0().sqrt()
    L2 = L * L; L3 = L2 * L
    ps_lo = np.broadcast_to(psig[:-1].reshape(1, 1, -1), sh); ps_hi = np.broadcast_to(psig[1:].reshape(1, 1, -1), sh)
    cpsi = cos_iv(ps_lo, ps_hi)
    d_lo = np.maximum(ps_lo - B3(p_hi), -6.3); d_hi = np.minimum(ps_hi - B3(p_lo), 6.3)
    cdiff = cos_iv(d_lo, d_hi)
    cphi = cos_iv(B3(p_lo), B3(p_hi))
    out = np.zeros(nb)
    for (ix, iy, iz) in [(2, 0, 1), (1, 0, 2), (0, 1, 2)]:
        mX = I(B3(m_lo[:, ix]), B3(m_hi[:, ix])); mY = I(B3(m_lo[:, iy]), B3(m_hi[:, iy])); mZ = I(B3(m_lo[:, iz]), B3(m_hi[:, iz]))
        num = mY.sq() + mZ.sq() + mX.sq() + rho.sq() + mX * rho * cdiff * 2.0
        Phi = np.minimum(up(num.hi / L2.lo * (1 + 4e-16)), 3.0)
        Phi = np.maximum(Phi, 0.0)
        qI = (mY * mZ * (mX * cphi + rho * cpsi) * 2.0) / L3
        S = LB_Smin(Phi.ravel(), qI.lo.ravel()).reshape(sh)
        Sq = I(np.where(np.isfinite(S), np.maximum(S, 0.0), 0.0)) / 3.0
        val = (I(La) * (Sq * Sq.sqrt())).lo
        val = np.where(np.isinf(S), np.inf, val)
        best = val.reshape(nb, -1).min(axis=1)
        tail = (I(Lmin) * (1.0 + Lspan) / SQ3).lo
        best = np.minimum(best, tail)
        out = np.maximum(out, best)
    return out


def payment_box(m_lo, m_hi, p_lo, p_hi, **kw):
    """rigorous lower bound on P_1/|A| over the box: max(P_lb, three-chain twisted)."""
    return np.maximum(plb_bound(m_lo, m_hi), tw3_bound(m_lo, m_hi, p_lo, p_hi, **kw))


if __name__ == '__main__':
    import time
    from tw3 import pay_tw3
    pts = [(0.524, 0.497, 0.635, 0.117), (0.555, 0.582, 0.856, 0.269), (1, 1, 1, 0.0), (0.3, 0.2, 0.9, 1.0)]
    m = np.array([p[:3] for p in pts]); ph = np.array([p[3] for p in pts])
    t0 = time.time()
    rig = tw3_bound(m, m, ph, ph, nL=64, npsi=72)
    print("rigorous point boxes:", np.round(rig, 4), " time", round(time.time() - t0, 2))
    for (a, b, c, p_) in pts[:2]:
        print("float", (a, b, c, p_), round(pay_tw3(a, b, c, p_, nL=60, npsi=48), 4))
    m_lo = np.array([[0.50, 0.48, 0.62]]); m_hi = np.array([[0.53, 0.51, 0.65]])
    t0 = time.time()
    print("box:", tw3_bound(m_lo, m_hi, np.array([0.08]), np.array([0.15]), nL=48, npsi=48), round(time.time() - t0, 2))
