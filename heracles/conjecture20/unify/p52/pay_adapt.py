"""Adaptive rigorous payment (three-chain circle-coupled twisted bound + P_lb) for one box of own data, s = 3/2, |A| = 1.

Cells (t, psi) with L = Lmin (1 + t), t in [0, TSPAN]; tail L >= Lmin(1+TSPAN) bounded by 3^{-1/2} L.
Every reported value is min over ALL current cells of a rigorous cell lower bound, hence a rigorous lower bound;
refinement only chooses which cells to split (guided by float centre values).
"""
import numpy as np
from iv import I, dn, up, cos_iv, SQ3
from pay_rig import LB_Smin, plb_bound

TSPAN = 1.5
CHAINS = [(2, 0, 1), (1, 0, 2), (0, 1, 2)]


def _cell_lb(mX, mY, mZ, cphi, p_lo, p_hi, Lmin, tlo, thi, slo, shi):
    """rigorous lower bounds for cells (arrays tlo..shi), single box (scalar intervals mX, mY, mZ, cphi)."""
    La = (I(Lmin) * I(1.0 + tlo)).lo
    Lb = (I(Lmin) * I(1.0 + thi)).hi
    L = I(La, Lb)
    rho = (L * (L - 1.0) * 2.0).clip0().sqrt()
    L2 = L * L; L3 = L2 * L
    cpsi = cos_iv(slo, shi)
    cdiff = cos_iv(np.maximum(slo - p_hi, -6.3), np.minimum(shi - p_lo, 6.3))
    num = mY.sq() + mZ.sq() + mX.sq() + rho.sq() + mX * rho * cdiff * 2.0
    Phi = np.minimum(up(num.hi / L2.lo * (1 + 4e-16)), 3.0)
    Phi = np.maximum(Phi, 0.0)
    qI = (mY * mZ * (mX * cphi + rho * cpsi) * 2.0) / L3
    S = LB_Smin(Phi, qI.lo)
    Sq = I(np.where(np.isfinite(S), np.maximum(S, 0.0), 0.0)) / 3.0
    val = (I(La) * (Sq * Sq.sqrt())).lo
    return np.where(np.isinf(S), np.inf, val)


def _centre_val(mx, my, mz, ph, Lmin, t, s_):
    """float value of the chain objective at a point (guidance only)"""
    from tw3 import Smin
    L = Lmin * (1 + t)
    rho = np.sqrt(max(2 * L * (L - 1.0), 0))
    P = (my ** 2 + mz ** 2 + mx ** 2 + rho ** 2 + 2 * mx * rho * np.cos(s_ - ph)) / L ** 2
    q = 2 * my * mz * (mx * np.cos(ph) + rho * np.cos(s_)) / L ** 3
    return L * (Smin(P, q, 2 / 3) / 3) ** 1.5


def payment(m_lo, m_hi, p_lo, p_hi, need=np.inf, n0=(24, 24), rounds=6, maxcells=6000):
    m_lo = np.asarray(m_lo, float); m_hi = np.asarray(m_hi, float)
    plb = plb_bound(m_lo[None, :], m_hi[None, :])[0]
    if plb >= need:
        return plb
    Lmin = max(1.0, m_lo.max())
    tail = (I(Lmin) * (1.0 + TSPAN) / SQ3).lo
    cphi = cos_iv(np.array([p_lo]), np.array([p_hi]))
    cphi = I(cphi.lo[0], cphi.hi[0])
    mc = 0.5 * (m_lo + m_hi); pc = 0.5 * (p_lo + p_hi)
    best_chain = -np.inf
    for (ix, iy, iz) in CHAINS:
        mX = I(m_lo[ix], m_hi[ix]); mY = I(m_lo[iy], m_hi[iy]); mZ = I(m_lo[iz], m_hi[iz])
        ug = np.sqrt(np.linspace(0, 1, n0[0] + 1)) * 0 + (np.arange(n0[0] + 1) / n0[0]) ** 2 * TSPAN
        sg = np.linspace(-np.pi, np.pi, n0[1] + 1)
        T0, S0 = np.meshgrid(np.arange(n0[0]), np.arange(n0[1]), indexing='ij')
        tlo = ug[T0.ravel()]; thi = ug[T0.ravel() + 1]; slo = sg[S0.ravel()]; shi = sg[S0.ravel() + 1]
        lb = _cell_lb(mX, mY, mZ, cphi, p_lo, p_hi, Lmin, tlo, thi, slo, shi)
        for rd in range(rounds):
            cur = min(lb.min(), tail)
            if cur >= need or cur <= best_chain - 0.05:
                break
            # guide: float centre values
            tmid = 0.5 * (tlo + thi); smid = 0.5 * (slo + shi)
            cen = _cell_lb(I(mc[ix]), I(mc[iy]), I(mc[iz]), I(np.cos(pc)), pc, pc, Lmin, tmid, tmid, smid, smid)
            U = min(cen.min(), need)
            ref = lb < U
            if not np.any(ref) or len(lb) + 3 * ref.sum() > maxcells:
                # refine only the lowest cells within budget
                k = max(1, (maxcells - len(lb)) // 3)
                order = np.argsort(lb)[:k]
                ref = np.zeros(len(lb), bool); ref[order] = True
                if len(lb) >= maxcells:
                    break
            keep = ~ref
            tm = 0.5 * (tlo + thi); sm = 0.5 * (slo + shi)
            ntlo = []; nthi = []; nslo = []; nshi = []
            for (a, b) in [(tlo, tm), (tm, thi)]:
                for (c, d) in [(slo, sm), (sm, shi)]:
                    ntlo.append(a[ref]); nthi.append(b[ref]); nslo.append(c[ref]); nshi.append(d[ref])
            ntlo = np.concatenate(ntlo); nthi = np.concatenate(nthi); nslo = np.concatenate(nslo); nshi = np.concatenate(nshi)
            nlb = _cell_lb(mX, mY, mZ, cphi, p_lo, p_hi, Lmin, ntlo, nthi, nslo, nshi)
            tlo = np.concatenate([tlo[keep], ntlo]); thi = np.concatenate([thi[keep], nthi])
            slo = np.concatenate([slo[keep], nslo]); shi = np.concatenate([shi[keep], nshi])
            lb = np.concatenate([lb[keep], nlb])
        chain_val = min(lb.min(), tail)
        best_chain = max(best_chain, chain_val)
        if best_chain >= need:
            break
    return max(plb, best_chain)


if __name__ == '__main__':
    import time
    for (m, ph) in [((0.524, 0.497, 0.635), 0.117), ((0.628, 0.628, 0.628), 0.022), ((1, 1, 1), 0.0)]:
        t0 = time.time()
        v = payment(np.array(m), np.array(m), ph, ph)
        print(m, ph, "adaptive rigorous payment", round(v, 4), "time", round(time.time() - t0, 2))
