"""Certified phase-blind semi-rigid channel-resolved bound at (p, n=2) for networks with features in L (general p).

Claim checked: for every s in [1, s1], every network whose features lie in L satisfies
      gamma <= TARGET * 3^{1-s} B_p Gamma_lb(s),   Gamma_lb(s) = (p/2) |F|^{-s}  (uniform allocation, a lower bound
      for Gamma_s(p), equal to it in the dense regime).
Ingredients as in rig72.py (see n2mix_channel_resolved_proof.tex):  single-character tests per feature (pairing
vectors, free indices zero), l2 paid ledger K = 2 ||(w_m)_paid||_2 (T-independent; ledger 'nopaid' gives K = 0 for
L-pure networks), Pythagorean elimination, payment max(1 - c1 x, theta max(1,x), theta g(x)), s-monotonicity
(d/ds log(1/(F_s Gamma_lb)) >= log|F| - log|L| > 0).
Parametrisation (scale invariance): the largest amplitude is 1 (feature m), the others A_l in [0,1];
x_l in [0, XMAX] or the slab x_l > XMAX (payment theta*XMAX, no detection).
Usage: python rigp.py p miss s1 TARGET ledger [maxdepth]     (L = F minus {miss})
"""
import sys, time, itertools
import numpy as np
import cvxpy as cp
import mpmath
from iv import I, dn, up
from chanlib import Chan

XMAX = 1.75


class Setup:
    def __init__(self, p, L, ledger):
        self.p = p; self.L = tuple(L); self.ledger = ledger
        self.F = list(range(1, (p - 1) // 2 + 1))
        self.CH = Chan(p, L)
        mpmath.mp.prec = 100
        ang = [mpmath.mpf(2) * mpmath.pi * k / p for k in range(p)]
        cs = np.array([float(mpmath.cos(a)) for a in ang]); sn = np.array([float(mpmath.sin(a)) for a in ang])
        self.COS = (cs - 1e-16, cs + 1e-16); self.SIN = (sn - 1e-16, sn + 1e-16)
        # kappa_l(d) enclosures
        self.KAP = {l: (np.array([dn(1.0 - self.COS[1][(l * d) % p]) for d in range(1, p)]),
                        np.array([up(1.0 - self.COS[0][(l * d) % p]) for d in range(1, p)])) for l in L}
        self.info = {}
        for l in L:
            inf = self.CH.component(l, '1')
            own = [m for (m, role, f, d) in inf if role == 'own'][0]
            free = [m for (m, role, f, d) in inf if role == 'free']
            paid = [m for (m, role, f, d) in inf if role == 'paid']
            self.info[l] = (own, free, paid)


class Model:
    def __init__(self, S):
        self.S = S
        p = S.p; L = S.L; half = (p - 1) // 2
        self.muh = cp.Variable(half, nonneg=True)
        mu = cp.hstack([self.muh[min(d, p - d) - 1] for d in range(1, p)])
        self.wr = {l: cp.Variable(p - 1) for l in L}; self.wi = {l: cp.Variable(p - 1) for l in L}
        self.cA = cp.Parameter(len(L), nonneg=True); self.cM = cp.Parameter(len(L), nonneg=True)
        self.wK = cp.Parameter(nonneg=True)
        cons = [cp.sum(mu) == 1]
        IR, II = S.CH.IDFT.real, S.CH.IDFT.imag
        za = {}
        for l in L:
            zr = IR @ self.wr[l] - II @ self.wi[l]; zi = IR @ self.wi[l] + II @ self.wr[l]
            za[l] = [cp.norm(cp.hstack([zr[d], zi[d]]), 2) for d in range(p - 1)]
        for d in range(p - 1):
            cons.append(mu[d] >= 2 * sum(za[l][d] for l in L))
        Svars = []; leaks = []
        for l in L:
            own, free, paid = S.info[l]
            Svars.append(self.wr[l][own - 1]); cons.append(self.wi[l][own - 1] == 0)
            for m in free:
                cons += [self.wr[l][m - 1] == 0, self.wi[l][m - 1] == 0]
            for m in paid:
                leaks += [self.wr[l][m - 1], self.wi[l][m - 1]]
        K = 2 * cp.norm(cp.hstack(leaks), 2) if (S.ledger == 'l2' and leaks) else cp.Constant(0.0)
        KM = np.array([[1 - np.cos(2 * np.pi * l * d / p) for d in range(1, p)] for l in L])
        a = self.cA @ (KM @ mu) - self.cM @ cp.hstack(Svars)
        t = cp.Variable(nonneg=True)
        cons.append(t >= a)
        self.prob = cp.Problem(cp.Minimize(cp.norm(cp.hstack([t, self.wK * K]), 2)), cons)

    def solve(self, s, A, x, inf):
        r = 1 / s
        F = np.sum(np.array([(3.0 ** (1 - s) * XMAX if inf[i] else pay(x[i], s)) * A[i] for i in range(len(A))]) ** r) ** s
        self.cA.value = np.asarray(A, float) / F
        self.cM.value = np.array([0.0 if inf[i] else x[i] * A[i] for i in range(len(A))]) / F
        self.wK.value = 3.0 ** (s - 1)
        try:
            self.prob.solve(solver=cp.CLARABEL)
        except Exception:
            self.prob.solve(solver=cp.SCS, eps=1e-9)
        return (np.maximum(self.muh.value, 0.0), {l: self.wr[l].value + 1j * self.wi[l].value for l in self.S.L})


def pay(x, s):
    c1 = 1 - 3.0 ** (1 - s); r = 1 / s
    v = max(1 - c1 * x, 3.0 ** (1 - s) * max(1.0, x))
    if x <= 1:
        v = max(v, 3.0 ** (1 - s) * (((1 + x) / 2) ** r + ((1 - x) / 2) ** r) ** s)
    return v


def zabs_up(S, w):
    p = S.p
    out = np.zeros(p - 1)
    for di, d in enumerate(range(1, p)):
        re = I(0.0); im = I(0.0)
        for m in range(1, p):
            k = (m * d) % p
            c = I(S.COS[0][k], S.COS[1][k]); sn = I(S.SIN[0][k], S.SIN[1][k])
            wr = float(w[m - 1].real); wi = float(w[m - 1].imag)
            re = re + (c * wr + sn * wi); im = im + (c * wi - sn * wr)
        re = I(dn(re.lo / p), up(re.hi / p)); im = I(dn(im.lo / p), up(im.hi / p))
        ar = max(abs(float(re.lo)), abs(float(re.hi))); ai = max(abs(float(im.lo)), abs(float(im.hi)))
        out[di] = up(np.sqrt(up(up(ar * ar) + up(ai * ai))))
    return out


def certify(S, muh, w):
    p = S.p; L = S.L
    w = {l: w[l].copy() for l in L}
    for l in L:
        own, free, paid = S.info[l]
        for m in free:
            w[l][m - 1] = 0.0
        w[l][own - 1] = max(w[l][own - 1].real, 0.0) + 0j
    mu = np.array([muh[min(d, p - d) - 1] for d in range(1, p)], dtype=float)
    za = {l: zabs_up(S, w[l]) for l in L}
    need = np.zeros(p - 1)
    for d in range(p - 1):
        acc = 0.0
        for l in L:
            acc = up(acc + up(2 * za[l][d]))
        need[d] = acc
    for d in range(p - 1):
        mu[d] = max(mu[d], need[d], need[p - 2 - d])
    for d in range(p - 1):
        if not (mu[d] >= need[d]):
            return None
    Z = I(0.0)
    for v in mu:
        Z = Z + float(v)
    Sdet = {l: max(dn(float(w[l][S.info[l][0] - 1].real)), 0.0) for l in L}
    if S.ledger == 'l2':
        acc = 0.0
        for l in L:
            for m in S.info[l][2]:
                acc = up(acc + up(up(w[l][m - 1].real ** 2) + up(w[l][m - 1].imag ** 2)))
        Kup = up(2.0 * up(np.sqrt(acc)))
    else:
        Kup = 0.0
    Kl = {}
    for l in L:
        acc = 0.0
        for d in range(p - 1):
            acc = up(acc + up(S.KAP[l][1][d] * mu[d]))
        Kl[l] = acc
    return dict(Z=float(Z.lo), S=Sdet, Kup=Kup, Kl=Kl)


def box_bound(S, cert, box, inf, s1):
    """box: dict l -> [A_lo, A_hi, x_lo, x_hi]; inf: dict l -> bool. Bound valid for all s in [1, s1]."""
    iv = mpmath.iv; iv.prec = 80
    L = S.L
    # a multilinear in (A_l, x_l): maximise over vertices
    a = -np.inf
    for corner in itertools.product(*[[(0, 0), (0, 1), (1, 0), (1, 1)] if not inf[l] else [(0, None), (1, None)] for l in L]):
        v = 0.0
        for l, (ia, ix) in zip(L, corner):
            A = (box[l][0], box[l][1])[ia]
            v = up(v + up(cert['Kl'][l] * A))
            if ix is not None:
                xx = (box[l][2], box[l][3])[ix]
                v = up(v - dn(dn(cert['S'][l] * xx) * A))
        a = max(a, v)
    aZ = up(a / cert['Z']) if a > 0 else 0.0
    KZ = up(cert['Kup'] / cert['Z'])
    S1 = iv.mpf(s1)
    th = iv.mpf(3) ** (1 - S1); c1 = 1 - th
    tot = iv.mpf(0)
    for l in L:
        if inf[l]:
            Fl = (th * XMAX).a
        else:
            xlo, xhi = box[l][2], box[l][3]
            v1 = (1 - c1 * xhi).a; v2 = (th * max(1.0, xlo)).a
            Fl = max(v1, v2)
            if xhi < 1.0:
                X = iv.mpf(xhi)
                v3 = (th * (((1 + X) / 2) ** (1 / S1) + ((1 - X) / 2) ** (1 / S1)) ** S1).a
                Fl = max(Fl, v3)
        y = mpmath.mpf(Fl) * mpmath.mpf(box[l][0])
        if y > 0:
            tot = tot + iv.mpf([y, y]) ** (1 / S1)
    Flow = (tot ** S1).a if tot.a > 0 else mpmath.mpf(0)
    if Flow <= 0:
        return np.inf
    nF = len(S.F)
    Gam_lo = (iv.mpf(S.p) / 2 * iv.mpf(nF) ** (-S1)).a
    w2 = (iv.mpf(3) ** (2 * S1 - 2)).b
    val = iv.sqrt((iv.mpf(aZ) / iv.mpf(Flow)) ** 2 + w2 * iv.mpf(KZ) ** 2) / iv.mpf(Gam_lo)
    return float(mpmath.mpf(val.b))


def precheck(S, box, inf, s1):
    """cheap valid bound: mu = point mass on {+d,-d}, no detection: h <= sum_l kappa_l(d) A_l; returns min over d of
    the resulting ratio bound (rigorous, outward rounding)."""
    iv = mpmath.iv; iv.prec = 80
    L = S.L; p = S.p
    S1 = iv.mpf(s1)
    th = iv.mpf(3) ** (1 - S1); c1 = 1 - th
    tot = iv.mpf(0)
    for l in L:
        if inf[l]:
            Fl = (th * XMAX).a
        else:
            xlo, xhi = box[l][2], box[l][3]
            Fl = max((1 - c1 * xhi).a, (th * max(1.0, xlo)).a)
            if xhi < 1.0:
                X = iv.mpf(xhi)
                Fl = max(Fl, (th * (((1 + X) / 2) ** (1 / S1) + ((1 - X) / 2) ** (1 / S1)) ** S1).a)
        y = mpmath.mpf(Fl) * mpmath.mpf(box[l][0])
        if y > 0:
            tot = tot + iv.mpf([y, y]) ** (1 / S1)
    if tot.a <= 0:
        return np.inf
    Flow = (tot ** S1).a
    Gam_lo = (iv.mpf(p) / 2 * iv.mpf(len(S.F)) ** (-S1)).a
    best = np.inf
    for d in range(1, (p - 1) // 2 + 1):
        a = 0.0
        for l in L:
            a = up(a + up(S.KAP[l][1][d - 1] * box[l][1]))
        val = (iv.mpf(a) / iv.mpf(Flow) / iv.mpf(Gam_lo)).b
        best = min(best, float(mpmath.mpf(val)))
    return best


def run(p, miss, s1, TARGET, ledger, maxdepth=30, verbose=True):
    F = list(range(1, (p - 1) // 2 + 1)); L = tuple(l for l in F if l != miss)
    S = Setup(p, L, ledger); M = Model(S)
    # scale invariance + s-monotonicity: H(w) <= log|L| < log|F|
    t0 = time.time(); nbox = 0; fails = 0; worst = 0.0
    stack = []
    for mx in L:
        for slabs in itertools.product([False, True], repeat=len(L)):
            box = {l: [ (1.0 if l == mx else 0.0), 1.0, 0.0, XMAX] for l in L}
            stack.append((box, dict(zip(L, slabs)), 0))
    while stack:
        box, inf, dep = stack.pop()
        nbox += 1
        v = precheck(S, box, inf, s1)
        if v > TARGET:
            A = [0.5 * (box[l][0] + box[l][1]) for l in L]
            x = [0.5 * (box[l][2] + box[l][3]) for l in L]
            muh, w = M.solve(s1, A, x, [inf[l] for l in L])
            cert = certify(S, muh, w)
            v = np.inf if cert is None else box_bound(S, cert, box, inf, s1)
        if v <= TARGET:
            worst = max(worst, v); continue
        if dep >= maxdepth:
            fails += 1
            print(f"FAIL box={box} inf={inf} v={v}", flush=True); continue
        # split the widest coordinate (A widths weight 1, x widths weight 1/XMAX)
        cand = []
        for l in L:
            cand.append((box[l][1] - box[l][0], l, 0))
            if not inf[l]:
                cand.append(((box[l][3] - box[l][2]) / XMAX, l, 2))
        wdt, l, k = max(cand)
        mid = 0.5 * (box[l][k] + box[l][k + 1])
        b1 = {q: list(v_) for q, v_ in box.items()}; b2 = {q: list(v_) for q, v_ in box.items()}
        b1[l][k + 1] = mid; b2[l][k] = mid
        stack += [(b1, inf, dep + 1), (b2, inf, dep + 1)]
        if verbose and nbox % 2000 == 0:
            print(f"  ... boxes {nbox} stack {len(stack)} fails {fails} worst {worst:.5f} [{time.time()-t0:.0f}s]", flush=True)
    print(f"DONE p={p} L={L} s in [1,{s1}] TARGET={TARGET} ledger={ledger}: boxes={nbox} fails={fails} worst={worst:.5f} [{time.time()-t0:.0f}s]", flush=True)


if __name__ == '__main__':
    p = int(sys.argv[1]); miss = int(sys.argv[2]); s1 = float(sys.argv[3]); TARGET = float(sys.argv[4]); ledger = sys.argv[5]
    maxdepth = int(sys.argv[6]) if len(sys.argv) > 6 else 30
    assert 3.0 ** (1 - s1) * XMAX >= 1.0
    run(p, miss, s1, TARGET, ledger, maxdepth)
