"""Rigorous degree-2 Hermitian-square certificates at (5,2), s = 3/2.

A certificate is a family (G_delta)_{delta=1..4} of PSD 9x9 Gram matrices, stored as G_delta = sum_k lam_k u_k u_k^*
with float lam_k >= 0 and float vectors u_k (so w_delta(theta) = sum_k lam_k |u_k^* v(theta)|^2 >= 0 EXACTLY for the
real numbers these floats denote, for every theta on the continuous torus).  All derived quantities are enclosed with
interval arithmetic (iv.py):
   mass M = sum_delta tr G_delta,  a_0, a_T (own pairings), leakage norm n.
Rigorous ratio bound over a cell of data (alpha, m1, m2, m12, phi) with payment lower bound F:
   ratio <= sqrt( max(Bnd0,0)^2 / F^2 + 12 n^2 ) / Gamma_{3/2},
   Bnd0 = cos(a) Re a0 + sin(a) Im a0 + m1 Re a1 + m2 Re a2 + m12 (cos(phi) Re a12 + sin(phi) Im a12).
"""
import numpy as np
from iv import I, dn, up, cos_iv, sin_iv, COS72, COS144, SIN72, SIN144, SQ2

E2 = [(a, b) for a in range(3) for b in range(3)]
FR2 = sorted(set((e2[0] - e[0], e2[1] - e[1]) for e in E2 for e2 in E2))
FT = [(1, 0), (0, 1), (1, 1)]
OWN_PAIR = {(0, 0)} | {(-f[0], -f[1]) for f in FT}

# c_1(delta) = 1 - rho^{-delta} = (1 - cos(2 pi delta/5)) + i sin(2 pi delta/5), delta = 1..4 (interval re, im)
_C1 = {1: (I(1.0) - COS72, SIN72), 2: (I(1.0) - COS144, SIN144), 3: (I(1.0) - COS144, -SIN144), 4: (I(1.0) - COS72, -SIN72)}
C1 = [_C1[d] for d in (1, 2, 3, 4)]
C2 = [_C1[(2 * d) % 5] for d in (1, 2, 3, 4)]
GAMMA = I(2.5) / (SQ2 * 2.0)       # Gamma_{3/2}(5) = 2.5 * 2^{-3/2}


def cmul(a, b):
    """complex interval multiply, a=(re,im), b=(re,im)"""
    return (a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0])


def cadd(a, b):
    return (a[0] + b[0], a[1] + b[1])


def czero():
    return (I(0.0), I(0.0))


def _isum(x, axis):
    """rigorous enclosure of sums of interval arrays along an axis: float sum of endpoints plus the recursive
    summation error bound n u sum|.| (u = 2^-53, doubled for safety)."""
    n = x.lo.shape[axis]
    slo = np.sum(x.lo, axis=axis); shi = np.sum(x.hi, axis=axis)
    elo = 2.0 * n * 2.0 ** -53 * np.sum(np.abs(x.lo), axis=axis) + 1e-300
    ehi = 2.0 * n * 2.0 ** -53 * np.sum(np.abs(x.hi), axis=axis) + 1e-300
    return I(dn(slo - elo), up(shi + ehi))


def rigorise(Gs):
    """Gs: list of 4 Hermitian (9x9) float arrays (approx PSD).  Returns interval data of the exact PSD certificate
    G~ = sum lam_k u_k u_k^* (negative eigenvalues dropped)."""
    what = []          # per delta: dict f -> complex interval (unnormalised)
    mass = I(0.0)
    for G in Gs:
        G = (G + G.conj().T) / 2
        lam, U = np.linalg.eigh(G)
        lam = np.maximum(lam, 0.0)
        n = G.shape[0]
        # G~[e,e2] = sum_k lam_k U[e,k] conj(U[e2,k]);  arrays (e, e2, k)
        ar = I(np.broadcast_to(U.real[:, None, :], (n, n, n))); ai = I(np.broadcast_to(U.imag[:, None, :], (n, n, n)))
        br = I(np.broadcast_to(U.real[None, :, :], (n, n, n))); bi = I(np.broadcast_to(U.imag[None, :, :], (n, n, n)))
        Lm = I(np.broadcast_to(lam[None, None, :], (n, n, n)))
        Gre = _isum(Lm * (ar * br + ai * bi), axis=2)
        Gim = _isum(Lm * (ai * br - ar * bi), axis=2)
        tr = _isum(I(np.diag(Gre.lo), np.diag(Gre.hi)), axis=0)
        mass = mass + tr
        wd = {}
        for f in FR2:
            idx = [(i, j) for i, e in enumerate(E2) for j, e2 in enumerate(E2) if (e2[0] - e[0], e2[1] - e[1]) == f]
            ii = np.array([t[0] for t in idx]); jj = np.array([t[1] for t in idx])
            re = _isum(I(Gre.lo[ii, jj], Gre.hi[ii, jj]), axis=0)
            im = _isum(I(Gim.lo[ii, jj], Gim.hi[ii, jj]), axis=0)
            wd[f] = (re, im)
        what.append(wd)
    return what, mass


def _sum_lo(x):
    """rigorous lower bound of the sum of the lower ends (sequential, rounding down)"""
    s = 0.0
    for v in np.atleast_1d(x.lo):
        s = dn(s + v)
    return s


def _sum_hi(x):
    s = 0.0
    for v in np.atleast_1d(x.hi):
        s = up(s + v)
    return s


def cert_data(Gs):
    what, mass = rigorise(Gs)
    if mass.lo <= 0:
        raise ValueError("certificate mass not positive")

    def dsum(cv, f):
        acc = czero()
        for k in range(4):
            acc = cadd(acc, cmul(cv[k], what[k][f]))
        return (acc[0] / mass, acc[1] / mass)
    a0 = dsum(C1, (0, 0))
    aT = [dsum(C1, (-f[0], -f[1])) for f in FT]
    n2 = I(0.0)
    for f in FR2:
        if f not in OWN_PAIR:
            x = dsum(C1, f); n2 = n2 + x[0].sq() + x[1].sq()
        if f != (0, 0):
            x = dsum(C2, f); n2 = n2 + x[0].sq() + x[1].sq()
    n = n2.sqrt()
    return dict(a0=a0, aT=aT, n=n)


def eval_cells(cd, al_lo, al_hi, m_lo, m_hi, ph_lo, ph_hi, F_lo):
    """vectorised rigorous ratio upper bound over cells.  m_lo, m_hi: (N,3); angles arrays (N,); F_lo (N,)."""
    ca = cos_iv(al_lo, al_hi); sa = sin_iv(al_lo, al_hi)
    cp = cos_iv(ph_lo, ph_hi); sp = sin_iv(ph_lo, ph_hi)
    a0, (a1, a2, a12) = cd['a0'], cd['aT']
    m1 = I(m_lo[:, 0], m_hi[:, 0]); m2 = I(m_lo[:, 1], m_hi[:, 1]); m12 = I(m_lo[:, 2], m_hi[:, 2])
    B0 = ca * a0[0] + sa * a0[1] + m1 * a1[0] + m2 * a2[0] + m12 * (cp * a12[0] + sp * a12[1])
    b = np.maximum(B0.hi, 0.0)
    t = (I(b) / I(F_lo)).sq() + cd['n'].sq() * 12.0
    return (t.sqrt() / GAMMA).hi


if __name__ == '__main__':
    import time
    from hs2 import elim_ratio
    # single atom test: build certificate at the point and evaluate on the degenerate cell
    F = 3 ** -0.5
    t0 = time.time()
    val, Gs = elim_ratio(0.0, 1.0, 1.0, 1.0, 0.0, F, deg=2, ret=True)
    cd = cert_data(Gs)
    r = eval_cells(cd, np.array([0.0]), np.array([0.0]), np.array([[1.0, 1, 1]]), np.array([[1.0, 1, 1]]),
                   np.array([0.0]), np.array([0.0]), np.array([dn(F)]))
    print("single atom: SDP ratio", round(val, 5), " rigorous", r, " time", round(time.time() - t0, 2))
    print("a0", cd['a0'], "n", cd['n'])
