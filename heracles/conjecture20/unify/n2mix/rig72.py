"""Certified phase-blind semi-rigid channel-resolved bound at (p,n) = (7,2), features in L = {1,2}.

THEOREM checked here (for the s-range given): every network at (7,2) with beta_3 = 0 has gamma <= TARGET * gamma_IF.
Ingredients (all proved in the accompanying text):
  * delta-Fourier single-character tests (pairing vectors w^(j), z_j = IDFT w^(j)), blind to free (F2) channels;
  * channel-resolved Pythagorean ledger: 3^{1-s} B_7 P^s >= sqrt(F^2 + 3^{2-2s} (sum rho_i)^2);
  * neuron-level paid constant K <= K-plan (kplan.py);
  * payment F_j >= A_j max(1 - c_1 x_j, 3^{1-s} max(1, x_j)), c_1 = 1 - 3^{1-s} (Note Thm c1sharp + trivial);
  * gamma/gamma_IF <= sqrt((a_+/F)^2 + 3^{2s-2} K^2) / Gamma_s,  Gamma_s(7) = 3.5 * 3^{-s}  (p52 R11).
Parameters: tau = A_2/(A_1+A_2) in [0,1]; x_j = max_T |beta_{j,T}| / A_j in [0, XMAX] or the slab x_j > XMAX.
For each box and each of the 5 slot-set cases (T1,T2) (mirror cases are equal by slot symmetry) a test is computed
by an SOCP at the box centre, then made exact (free entries zero; positivity enforced by an interval-verified scale
factor), and the bound is evaluated with outward rounding over the whole box (and the whole s-cell).
ledger: 'pat' (K-plan) or 'nopaid' (K = 0; valid for networks with no DC and no frequency-3 content, 'L-pure').
Usage: python rig72.py s1 TARGET ledger [maxdepth]   (certifies every s in [1, s1])
"""
import sys, time, itertools
import numpy as np
import cvxpy as cp
import mpmath
from iv import I, dn, up, _cos_pt, _sin_pt
from chanlib import Chan
from kplan import make_plan, eval_cvx, eval_up

p = 7
CH = Chan(7, (1, 2))
CASES = [('1', '1'), ('1', '2'), ('1', '12'), ('12', '1'), ('12', '12')]
XMAX = 1.75
import os
KSCALE = float(os.environ.get('KSCALE', '1.0'))   # what-if factor on the paid ledger (1.0 = proved ledger)
TWO_PI_7 = 2 * np.pi / 7

# rigorous cos/sin(2 pi k / 7), k = 0..6 : enclose 2 pi k/7 by an interval and use Taylor enclosures at both ends
mpmath.mp.prec = 100
_ang = [mpmath.mpf(2) * mpmath.pi * k / 7 for k in range(7)]
ANG_LO = np.array([float(mpmath.mpf(a) - mpmath.mpf(2) ** -60) for a in _ang])
ANG_HI = np.array([float(mpmath.mpf(a) + mpmath.mpf(2) ** -60) for a in _ang])
# exact cos values bracketed via mpmath (directly; tight and rigorous)
COS7 = I(np.array([float(mpmath.cos(a)) for a in _ang]) - 1e-16, np.array([float(mpmath.cos(a)) for a in _ang]) + 1e-16)
SIN7 = I(np.array([float(mpmath.sin(a)) for a in _ang]) - 1e-16, np.array([float(mpmath.sin(a)) for a in _ang]) + 1e-16)
# kappa_l(d) = 1 - cos(2 pi l d / 7), l = 1,2 ; d = 1..6
KAPI = {l: I(np.array([1.0 - COS7.hi[(l * d) % 7] for d in range(1, 7)]), np.array([1.0 - COS7.lo[(l * d) % 7] for d in range(1, 7)]))
        for l in (1, 2)}
KAPI = {l: I(dn(KAPI[l].lo), up(KAPI[l].hi)) for l in KAPI}


def comp_info(j, T):
    info = CH.component(j, T)
    own = [m for (m, role, f, d) in info if role == 'own'][0]
    free = [m for (m, role, f, d) in info if role == 'free']
    paid = [(m, tuple(int(t) for t in f), d) for (m, role, f, d) in info if role == 'paid']
    return own, free, paid


class Model:
    """SOCP for one case."""
    def __init__(self, case, ledger):
        self.case = case; self.ledger = ledger
        self.mu3 = cp.Variable(3, nonneg=True)
        mu = cp.hstack([self.mu3[0], self.mu3[1], self.mu3[2], self.mu3[2], self.mu3[1], self.mu3[0]])
        self.wr = {j: cp.Variable(6) for j in (1, 2)}; self.wi = {j: cp.Variable(6) for j in (1, 2)}
        self.cA = cp.Parameter(2, nonneg=True); self.cM = cp.Parameter(2, nonneg=True)
        self.wK = cp.Parameter(nonneg=True)
        cons = [cp.sum(mu) == 1]
        IR, II = CH.IDFT.real, CH.IDFT.imag
        zabs = {}
        for j in (1, 2):
            zr = IR @ self.wr[j] - II @ self.wi[j]; zi = IR @ self.wi[j] + II @ self.wr[j]
            zabs[j] = [cp.norm(cp.hstack([zr[d], zi[d]]), 2) for d in range(6)]
        for d in range(6):
            cons.append(mu[d] >= 2 * zabs[1][d] + 2 * zabs[2][d])
        self.info = {}
        entry = {}
        S = {}
        for j, T in ((1, case[0]), (2, case[1])):
            own, free, paid = comp_info(j, T)
            self.info[j] = (own, free, paid)
            S[j] = self.wr[j][own - 1]
            cons.append(self.wi[j][own - 1] == 0)
            for m in free:
                cons += [self.wr[j][m - 1] == 0, self.wi[j][m - 1] == 0]
            for (m, f, d) in paid:
                ab = cp.norm(cp.hstack([self.wr[j][m - 1], self.wi[j][m - 1]]), 2)
                entry[f] = entry.get(f, 0) + 2 ** (d / 2) * ab
        self.plan = make_plan(entry.keys())
        K = eval_cvx(self.plan, entry) if ledger == 'pat' else cp.Constant(0.0)
        Kmu = CH.KM[:2] @ mu
        a = self.cA[0] * Kmu[0] + self.cA[1] * Kmu[1] - self.cM[0] * S[1] - self.cM[1] * S[2]
        t = cp.Variable(nonneg=True)
        cons.append(t >= a)
        self.prob = cp.Problem(cp.Minimize(cp.norm(cp.hstack([t, self.wK * K]), 2)), cons)

    def solve(self, s, tau, x1, x2, inf1, inf2):
        r = 1 / s
        A1, A2 = 1 - tau, tau
        F1 = 3.0 ** (1 - s) * XMAX if inf1 else pay(x1, s); F2 = 3.0 ** (1 - s) * XMAX if inf2 else pay(x2, s)
        F = ((F1 * A1) ** r + (F2 * A2) ** r) ** s
        self.cA.value = np.array([A1, A2]) / F
        self.cM.value = np.array([0.0 if inf1 else x1 * A1, 0.0 if inf2 else x2 * A2]) / F
        self.wK.value = 3.0 ** (s - 1) * KSCALE
        try:
            self.prob.solve(solver=cp.CLARABEL)
        except Exception:
            self.prob.solve(solver=cp.SCS, eps=1e-9)
        mu3 = np.maximum(self.mu3.value, 0.0)
        w = {j: self.wr[j].value + 1j * self.wi[j].value for j in (1, 2)}
        return mu3, w, self.prob.value / (3.5 * 3.0 ** (-s))


def pay(x, s):
    """payment lower bound P_j / A_j >= max(1 - c_1 x, 3^{1-s} max(1,x), [x<=1] 3^{1-s} (((1+x)/2)^r + ((1-x)/2)^r)^s)
    (c1sharp, trivial, and the two-atom cap of arith2 R1 with min_T |beta_T| <= x A)"""
    c1 = 1 - 3.0 ** (1 - s); r = 1 / s
    v = max(1 - c1 * x, 3.0 ** (1 - s) * max(1.0, x))
    if x <= 1:
        v = max(v, 3.0 ** (1 - s) * (((1 + x) / 2) ** r + ((1 - x) / 2) ** r) ** s)
    return v


# ---------------------------------------------------------------- rigorous evaluation
def zabs_up(w):
    """upper bounds of |z(d)|, d = 1..6, z(d) = -(1/7) sum_{m=1}^6 w_m rho^{-m d}"""
    out = np.zeros(6)
    for di, d in enumerate(range(1, 7)):
        re = I(0.0); im = I(0.0)
        for m in range(1, 7):
            k = (m * d) % 7
            c = I(COS7.lo[k], COS7.hi[k]); s_ = I(SIN7.lo[k], SIN7.hi[k])
            # rho^{-k} = cos - i sin
            wr = float(w[m - 1].real); wi = float(w[m - 1].imag)
            re = re + (c * wr + s_ * wi)
            im = im + (c * wi - s_ * wr)
        re = re * (1.0 / 7.0) if False else I(dn(re.lo / 7.0), up(re.hi / 7.0))
        im = I(dn(im.lo / 7.0), up(im.hi / 7.0))
        ar = max(abs(float(re.lo)), abs(float(re.hi))); ai = max(abs(float(im.lo)), abs(float(im.hi)))
        out[di] = up(np.sqrt(up(up(ar * ar) + up(ai * ai))))
    return out


def certify_test(case, mu3, w, ledger):
    """make the test exact; return dict with Z (interval), S1, S2 (floats, lower bounds), Kup, K1, K2 (intervals)"""
    own = {}; paid = {}
    w = {j: w[j].copy() for j in (1, 2)}
    for j, T in ((1, case[0]), (2, case[1])):
        o, free, pd = comp_info(j, T)
        own[j] = o; paid[j] = pd
        for m in free:
            w[j][m - 1] = 0.0
        w[j][o - 1] = max(w[j][o - 1].real, 0.0) + 0j
    mu3 = np.maximum(np.asarray(mu3, float), 0.0)
    mu = np.array([mu3[0], mu3[1], mu3[2], mu3[2], mu3[1], mu3[0]], dtype=float)
    za1 = zabs_up(w[1]); za2 = zabs_up(w[2])
    need = np.array([up(up(2 * za1[d]) + up(2 * za2[d])) for d in range(6)])
    # raise mu where the solver left a tiny violation (keeps mu symmetric); the test is renormalised by Z below
    for d in range(6):
        mu[d] = max(mu[d], need[d], need[5 - d])
    # exact positivity check: mu_d >= 2|z_1(d)| + 2|z_2(d)| (upper bounds)
    for d in range(6):
        if not (mu[d] >= need[d]):
            return None
    Z = I(0.0)
    for v in mu:
        Z = Z + float(v)
    S = {j: max(dn(float(w[j][own[j] - 1].real)), 0.0) for j in (1, 2)}
    if ledger == 'pat':
        entry = {}
        for j in (1, 2):
            for (m, f, d) in paid[j]:
                ab = up(np.sqrt(up(up(w[j][m - 1].real ** 2) + up(w[j][m - 1].imag ** 2))))
                coef = 1.0 if d == 0 else (up(np.sqrt(2.0)) if d == 1 else 2.0)
                entry[f] = up(entry.get(f, 0.0) + up(coef * ab))
        Kup = eval_up(make_plan(entry.keys()), entry)
    else:
        Kup = 0.0
    Kl = {}
    for l in (1, 2):
        acc = I(0.0)
        for d in range(6):
            acc = acc + I(KAPI[l].lo[d], KAPI[l].hi[d]) * float(mu[d])
        Kl[l] = acc
    return dict(Z=Z, S=S, Kup=Kup, Kl=Kl)


def box_bound(cert, t0, t1, x1lo, x1hi, x2lo, x2hi, inf1, inf2, s1):
    """rigorous upper bound of the ratio over the box, valid for EVERY s in [1, s1]:
    for a fixed test and configuration the bound B(s) = sqrt((a_+/(F_s Gamma_s))^2 + 3^{2s-2} K^2/Gamma_s^2) is
    nondecreasing in s (d/ds log(1/(F_s Gamma_s)) = log 3 - H(w) - sum_j w_j (log F_j)' >= log 3 - log 2 > 0), so the
    value at s1 dominates."""
    iv = mpmath.iv
    iv.prec = 80
    Z = cert['Z']; S = cert['S']; Kl = cert['Kl']
    # a = K1 A1 + K2 A2 - S1 x1 A1 - S2 x2 A2 is multilinear in (tau, x1, x2): its max over the box is at a vertex
    a = -np.inf
    for tau in (t0, t1):
        A1 = 1.0 - tau; A2 = tau          # A1 computed exactly? 1 - tau is exact for tau in [0,1] dyadic; round up/down
        A1u = up(A1); A1d = dn(A1)
        for x1 in ((x1lo, x1hi) if not inf1 else (None,)):
            for x2 in ((x2lo, x2hi) if not inf2 else (None,)):
                v = up(up(float(Kl[1].hi) * A1u) + up(float(Kl[2].hi) * A2))
                if x1 is not None:
                    v = up(v - dn(dn(S[1] * x1) * A1d))
                if x2 is not None:
                    v = up(v - dn(dn(S[2] * x2) * A2))
                a = max(a, v)
    aZ = up(a / float(Z.lo)) if a > 0 else 0.0
    KZ = up(cert['Kup'] / float(Z.lo))
    S1 = iv.mpf(s1)
    th = iv.mpf(3) ** (1 - S1)
    c1 = 1 - th
    def Flo(xlo, xhi, inf):
        if inf:
            return iv.mpf((th * XMAX).a)          # slab x > XMAX: F_j >= theta * x >= theta * XMAX (>= 1)
        v1 = 1 - c1 * xhi
        v2 = th * max(1.0, xlo)
        lo = max(v1.a, v2.a)
        if xhi <= 1.0:
            # two-atom cap (decreasing in x on [0,1]): evaluate at xhi
            X = iv.mpf(xhi)
            v3 = th * (((1 + X) / 2) ** (1 / S1) + ((1 - X) / 2) ** (1 / S1)) ** S1 if xhi < 1.0 else th
            lo = max(lo, v3.a)
        return iv.mpf([lo, lo])
    F1 = Flo(x1lo, x1hi, inf1); F2 = Flo(x2lo, x2hi, inf2)
    R = 1 / S1
    def pw(y):
        if y.a <= 0:
            return iv.mpf([0, 0])
        return iv.mpf([y.a, y.a]) ** R
    # F(tau) = ((F1 (1-tau))^r + (F2 tau)^r)^s is quasi-concave in tau: its min over [t0,t1] is at an endpoint
    Flow = np.inf
    for tau in (t0, t1):
        y1 = F1 * iv.mpf(dn(1.0 - tau)); y2 = F2 * iv.mpf(tau)
        nrm = (pw(y1) + pw(y2)) ** S1
        Flow = min(Flow, float(mpmath.mpf(nrm.a)))
    if Flow <= 0:
        return np.inf
    Gam_lo = (iv.mpf(3.5) * iv.mpf(3) ** (-S1)).a
    w2 = (iv.mpf(3) ** (2 * S1 - 2)).b
    val = iv.sqrt((iv.mpf(aZ) / iv.mpf(Flow)) ** 2 + w2 * iv.mpf(KZ) ** 2) / iv.mpf(Gam_lo)
    return float(mpmath.mpf(val.b))


def run(s1, TARGET, ledger, maxdepth=40, ssolve=None, verbose=False):
    """certify the bound <= TARGET for every s in [1, s1] (monotonicity in s), all boxes"""
    ssolve = s1 if ssolve is None else ssolve
    models = {case: Model(case, ledger) for case in CASES}
    fails = 0; worst = 0.0; t0c = time.time()
    stack = [(0.0, 1.0, 0.0, XMAX, 0.0, XMAX, i1, i2, 0) for i1 in (False, True) for i2 in (False, True)]
    nbox = 0; nsolve = 0
    while stack:
        t0, t1, a0, a1, b0, b1, i1, i2, dep = stack.pop()
        nbox += 1
        ok = True; wb = 0.0
        for case in CASES:
            mu3, w, _ = models[case].solve(ssolve, 0.5 * (t0 + t1), 0.5 * (a0 + a1), 0.5 * (b0 + b1), i1, i2)
            nsolve += 1
            cert = certify_test(case, mu3, w, ledger)
            if cert is None:
                ok = False; break
            v = box_bound(cert, t0, t1, a0, a1, b0, b1, i1, i2, s1)
            wb = max(wb, v)
            if v > TARGET:
                ok = False; break
        if ok:
            worst = max(worst, wb)
            continue
        if dep >= maxdepth:
            fails += 1
            print(f"FAIL tau=[{t0},{t1}] x1=[{a0},{a1}]{'inf' if i1 else ''} x2=[{b0},{b1}]{'inf' if i2 else ''}", flush=True)
            continue
        wdt = [t1 - t0, 0 if i1 else (a1 - a0) / 1.75, 0 if i2 else (b1 - b0) / 1.75]
        i = int(np.argmax(wdt))
        if i == 0:
            m = 0.5 * (t0 + t1); stack += [(t0, m, a0, a1, b0, b1, i1, i2, dep + 1), (m, t1, a0, a1, b0, b1, i1, i2, dep + 1)]
        elif i == 1:
            m = 0.5 * (a0 + a1); stack += [(t0, t1, a0, m, b0, b1, i1, i2, dep + 1), (t0, t1, m, a1, b0, b1, i1, i2, dep + 1)]
        else:
            m = 0.5 * (b0 + b1); stack += [(t0, t1, a0, a1, b0, m, i1, i2, dep + 1), (t0, t1, a0, a1, m, b1, i1, i2, dep + 1)]
        if verbose and nbox % 500 == 0:
            print(f"  ... boxes {nbox} stack {len(stack)} fails {fails} [{time.time()-t0c:.0f}s]", flush=True)
    print(f"DONE s in [1,{s1}] TARGET={TARGET} ledger={ledger}: boxes={nbox} solves={nsolve} fails={fails} worst={worst:.5f} [{time.time()-t0c:.0f}s]", flush=True)


if __name__ == '__main__':
    s1 = float(sys.argv[1]); TARGET = float(sys.argv[2]); ledger = sys.argv[3]
    maxdepth = int(sys.argv[4]) if len(sys.argv) > 4 else 40
    assert 3.0 ** (1 - s1) * XMAX >= 1.0
    run(s1, TARGET, ledger, maxdepth, verbose=True)
