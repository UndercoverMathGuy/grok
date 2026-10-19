"""Referee's independent exact certificate for SPLIT R10 (the (p,n,b) = (5,2,2) decoupling violator).

Route (different from certify_viol.py): every quantity that is algebraic is computed EXACTLY in Q(sqrt5)
(p = 5: cos(2 pi/5) = (sqrt5-1)/4, cos(4 pi/5) = -(sqrt5+1)/4), with exact signs; only the final r-th powers
(cube roots, 3/2 powers) are enclosed with mpmath interval arithmetic at 80 digits.
Weights: (A) the exact binary rationals of the stored float64 weights; (B) the 12-significant-digit rounding used by
the split agent.  The split network theta_split is also handled exactly (its table lies in Q(sqrt5)).
Run: uv run --with numpy --with mpmath python v_r10_certify.py
"""
import itertools, sys
from fractions import Fraction as Fr
import numpy as np
from mpmath import iv, mp
iv.dps = 80; mp.dps = 80

p, n, b = 5, 2, 2
nu = 3


class Q5:
    """a + b sqrt5 with a, b rational."""
    __slots__ = ('a', 'b')
    def __init__(self, a, b=0):
        self.a = Fr(a); self.b = Fr(b)
    def __add__(s, o):
        o = o if isinstance(o, Q5) else Q5(o); return Q5(s.a + o.a, s.b + o.b)
    __radd__ = __add__
    def __neg__(s):
        return Q5(-s.a, -s.b)
    def __sub__(s, o):
        return s + (-(o if isinstance(o, Q5) else Q5(o)))
    def __rsub__(s, o):
        return Q5(o) - s
    def __mul__(s, o):
        o = o if isinstance(o, Q5) else Q5(o); return Q5(s.a * o.a + 5 * s.b * o.b, s.a * o.b + s.b * o.a)
    __rmul__ = __mul__
    def sign(s):
        a, b_ = s.a, s.b
        if a == 0 and b_ == 0: return 0
        if a >= 0 and b_ >= 0: return 1
        if a <= 0 and b_ <= 0: return -1
        if a * a > 5 * b_ * b_: return 1 if a > 0 else -1
        return 1 if b_ > 0 else -1
    def __lt__(s, o): return (s - o).sign() < 0
    def iv(s):
        return iv.mpf(s.a.numerator) / s.a.denominator + (iv.mpf(s.b.numerator) / s.b.denominator) * iv.sqrt(5)
    def __float__(s): return float(s.a) + float(s.b) * 5 ** 0.5


COS = {0: Q5(1), 1: Q5(Fr(-1, 4), Fr(1, 4)), 4: Q5(Fr(-1, 4), Fr(1, 4)), 2: Q5(Fr(-1, 4), Fr(-1, 4)), 3: Q5(Fr(-1, 4), Fr(-1, 4))}
# cos(2 pi m / 5) for m mod 5
assert abs(float(COS[1]) - np.cos(2 * np.pi / 5)) < 1e-15 and abs(float(COS[2]) - np.cos(4 * np.pi / 5)) < 1e-15


def absdft2(v, k):
    """|vhat(k)|^2 = sum_{a,c} v(a) v(c) cos(2 pi k (a-c)/p), exact in Q(sqrt5)."""
    out = Q5(0)
    for a in range(p):
        for c in range(p):
            out = out + COS[(k * (a - c)) % p] * (v[a] * v[c])
    return out


def proj(v, k):
    """(P_k v)(a) = (2/p) sum_c v(c) cos(2 pi k (a-c)/p), exact in Q(sqrt5)."""
    return [sum((COS[(k * (a - c)) % p] * (Fr(2, p) * v[c]) for c in range(p)), Q5(0)) for a in range(p)]


def margin_exact(Us, Ws, num=Q5):
    best = None
    for x in itertools.product(range(p), repeat=n):
        y = sum(x) % p
        f = [sum((Ws[i][c] * Us[i][0][x[0]] * Us[i][1][x[1]] for i in range(len(Ws))), num(0)) for c in range(p)]
        for c in range(p):
            if c == y: continue
            g = f[y] - f[c]
            if best is None or g < best:
                best = g
    return best


def ivpow(x, e):
    return iv.exp(e * iv.log(x))


def analyse(Us, Ws, tag):
    m = len(Ws)
    h = margin_exact(Us, Ws)
    h = h if isinstance(h, Q5) else Q5(h)
    P = sum((sum((Q5(0) + u * u for t in range(n) for u in Us[i][t]), Q5(0)) + sum((Q5(0) + w * w for w in Ws[i]), Q5(0)) for i in range(m)), Q5(0))
    Qs = {}; Pl = {}; live = True; minabs2 = None
    for l in (1, 2):
        Ql = iv.mpf(0); Pl_ = Q5(0)
        for i in range(m):
            w2 = absdft2(Ws[i], l); u2 = [absdft2(Us[i][t], l) for t in range(n)]
            for z in [w2] + u2:
                if z.sign() <= 0: live = False
            mod2 = w2 * u2[0] * u2[1] * Q5(Fr(4, p ** (2 * nu)))     # |beta_il|^2 exactly
            Ql += ivpow(mod2.iv(), iv.mpf(1) / 3)                      # |beta_il|^{2/3} = (|beta_il|^2)^{1/3}
            Pl_ = Pl_ + Q5(Fr(2, p)) * (w2 + u2[0] + u2[1])            # piece energy (b = 2)
            mn = float(mod2)
            minabs2 = mn if minabs2 is None else min(minabs2, mn)
        Qs[l] = Ql; Pl[l] = Pl_
    k_lo = Q5(Fr(5, 4), Fr(-1, 4)); k_hi = Q5(Fr(5, 4), Fr(1, 4))      # (5-sqrt5)/4, (5+sqrt5)/4
    def cover(y1, y2):   # min over delta of y1 kappa_1 + y2 kappa_2 (two values: delta=+-1, +-2)
        c1 = y1 * k_lo.iv() + y2 * k_hi.iv(); c2 = y1 * k_hi.iv() + y2 * k_lo.iv()
        return iv.mpf([min(c1.a, c2.a), min(c1.b, c2.b)])
    Bp = 2 * ivpow(iv.mpf(30), iv.mpf(-3) / 2)
    nu1s = 1 / iv.sqrt(3)
    s = iv.mpf(3) / 2
    Piv = P.iv(); hiv = h.iv()
    Qtot = Qs[1] + Qs[2]
    rhsQ = nu1s * Bp * ivpow(Piv / Qtot, s) * cover(ivpow(Qs[1], s), ivpow(Qs[2], s))
    Pfeat = (Pl[1] + Pl[2]).iv()
    rhsP = nu1s * Bp * ivpow(Piv / Pfeat, s) * cover(ivpow(Pl[1].iv(), s), ivpow(Pl[2].iv(), s))
    Gam = iv.mpf(5) / (4 * iv.sqrt(2))
    gIF = nu1s * Bp * Gam
    rQ = hiv / rhsQ; rP = hiv / rhsP; g = hiv / ivpow(Piv, s) / gIF
    # existential (D) with the programme maximiser split P_j = P/2 (support {1,2} = features): RHS = gamma_IF P^s
    rExist = hiv / (gIF * ivpow(Piv, s))
    print(f'{tag}:')
    print(f'   h (exact, Q(sqrt5)) = {float(h):.10e}  sign {h.sign()}   P = {float(P):.10f}   all pieces live: {live}')
    print(f'   h/RHS(D_Q)     in [{mp.nstr(rQ.a, 12)}, {mp.nstr(rQ.b, 12)}]')
    print(f'   h/RHS(D_split) in [{mp.nstr(rP.a, 12)}, {mp.nstr(rP.b, 12)}]')
    print(f'   gamma/gamma_IF in [{mp.nstr(g.a, 12)}, {mp.nstr(g.b, 12)}]')
    print(f'   existential (D) with P_j = P/2 (the programme maximiser): h/RHS in [{mp.nstr(rExist.a, 10)}, {mp.nstr(rExist.b, 10)}]')
    print(f'   P_1^live/P = {float(Pl[1]) / float(P):.6f}, P_2^live/P = {float(Pl[2]) / float(P):.6f}, '
          f'DC+off energy share = {1 - (float(Pl[1]) + float(Pl[2])) / float(P):.6f}; min |beta_il|^2 = {minabs2:.3e}')
    return rQ, rP, g


d = np.load('../viol_p5_n2_b2.npz')
U = d['U0']; W = d['W0']
m = len(W)
results = {}
for mode in ['exact-float64', '12-digit']:
    if mode == 'exact-float64':
        cv = lambda x: Fr(float(x))
    else:
        cv = lambda x: Fr(float('%.12g' % x))
    Uq = [[[cv(U[i, t, a]) for a in range(p)] for t in range(n)] for i in range(m)]
    Wq = [[cv(W[i, a]) for a in range(p)] for i in range(m)]
    # theta with rational weights: margin is rational; wrap in Q5 for uniform code
    UQ = [[[Q5(x) for x in Uq[i][t]] for t in range(n)] for i in range(m)]
    WQ = [[Q5(x) for x in Wq[i]] for i in range(m)]
    rQ, rP, g = analyse(UQ, WQ, f'theta ({mode} weights, width {m})')
    # exact split (pieces in Q(sqrt5))
    Us = []; Ws = []
    for i in range(m):
        for k in (1, 2):
            Us.append([proj(Uq[i][t], k) for t in range(n)]); Ws.append(proj(Wq[i], k))
    rQs, rPs, gs = analyse(Us, Ws, f'theta_split ({mode}, exact Q(sqrt5) pieces, {len(Ws)} pieces)')
    ok = rQ.a > 1 and rP.a > 1 and g.b < 1 and rQs.b < 1 and rPs.b < 1
    results[mode] = ok
    print(f'   CERTIFIED ({mode}): theta violates (D_Q) and (D_split), gamma < gamma_IF, split satisfies both: {ok}\n')

# independent float re-evaluation of Gamma_{3/2}(5) (non-convex constraint): maximise over u in [0,1]
u = np.linspace(0, 1, 200001)
y1, y2 = u ** 1.5, (1 - u) ** 1.5
klo, khi = (5 - 5 ** .5) / 4, (5 + 5 ** .5) / 4
val = np.minimum(y1 * klo + y2 * khi, y1 * khi + y2 * klo)
print('Gamma_{3/2}(5) grid max = %.12f at u = %.6f ; claimed 5/(4 sqrt2) = %.12f' % (val.max(), u[val.argmax()], 5 / (4 * 2 ** .5)))
print('ALL CERTIFIED:', all(results.values()))
