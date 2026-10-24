"""Referee audit of iv.py (outward-rounded interval primitives) against mpmath at 120 bits.

Tests (random + adversarial):
  * +,-,*,/,sq,sqrt enclosures contain the exact real result (exact rational check via mpmath at high precision);
  * cos_iv / sin_iv over intervals contain cos/sin at endpoints, at many interior points, near the critical points
    k*pi, pi/2 + k*pi, and at the float-pi edges used by the grids (+-3.141592653589793);
  * cbrt_lo / cbrt_hi / pow23: exact check y^3 <= x (lo) and y^3 >= x (hi) in rational arithmetic, including perfect
    cubes, tiny, huge and subnormal inputs; also reports whether cbrt_hi's verification loop can exit unverified;
  * the constants COS72, COS144, SIN72, SIN144, PI contain the true values.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import numpy as np
import mpmath as mp
from fractions import Fraction as Fr
import iv
from iv import I

mp.mp.prec = 200
rng = np.random.default_rng(12345)
bad = []


def contains(Iv, val, k=None):
    lo = Iv.lo if k is None else Iv.lo[k]
    hi = Iv.hi if k is None else Iv.hi[k]
    return mp.mpf(float(lo)) <= val <= mp.mpf(float(hi))


# ---------------------------------------------------------------- arithmetic
N = 4000
a = rng.standard_normal(N) * 10 ** rng.uniform(-8, 8, N)
b = rng.standard_normal(N) * 10 ** rng.uniform(-8, 8, N)
A = I(a); B = I(b)
ops = {'add': (A + B, lambda x, y: x + y), 'sub': (A - B, lambda x, y: x - y), 'mul': (A * B, lambda x, y: x * y),
       'div': (A / B, lambda x, y: x / y)}
for name, (res, f) in ops.items():
    nb = 0
    for k in range(N):
        v = f(mp.mpf(float(a[k])), mp.mpf(float(b[k])))
        if not contains(res, v, k):
            nb += 1
    print(f"arith {name}: {nb} containment failures / {N}")
    if nb: bad.append(name)
sq = A.sq(); nb = sum(not contains(sq, mp.mpf(float(a[k])) ** 2, k) for k in range(N))
print("sq failures", nb); bad += ['sq'] if nb else []
pa = np.abs(a); sr = I(pa).sqrt(); nb = sum(not contains(sr, mp.sqrt(mp.mpf(float(pa[k]))), k) for k in range(N))
print("sqrt failures", nb); bad += ['sqrt'] if nb else []

# ---------------------------------------------------------------- constants
for name, c, v in [('COS72', iv.COS72, mp.cos(2 * mp.pi / 5)), ('COS144', iv.COS144, mp.cos(4 * mp.pi / 5)),
                   ('SIN72', iv.SIN72, mp.sin(2 * mp.pi / 5)), ('SIN144', iv.SIN144, mp.sin(4 * mp.pi / 5)),
                   ('PI', iv.PI, mp.pi), ('SQ2', iv.SQ2, mp.sqrt(2)), ('SQ3', iv.SQ3, mp.sqrt(3))]:
    ok = contains(c, v)
    print(f"const {name}: contains true value {ok}; width {float(c.hi - c.lo):.2e}")
    if not ok: bad.append(name)

# ---------------------------------------------------------------- cos/sin on intervals
crit = []
for k in range(-2, 3):
    crit += [float(k * mp.pi), float(mp.pi / 2 + k * mp.pi)]
crit = [c for c in crit if abs(c) <= 6.3]
los = []; his = []
for _ in range(20000):
    lo = rng.uniform(-6.3, 6.3); w = 10 ** rng.uniform(-17, 0.5)
    hi = min(lo + w, 6.3); los.append(lo); his.append(hi)
for c in crit:          # adversarial: intervals ending/starting exactly at or next to critical floats
    for d in (0.0, 1e-16, -1e-16, 1e-12, -1e-12, 1e-3):
        x = np.nextafter(c, np.inf) if d > 0 else c
        for (l, h) in [(c + d, c + d + 1e-9), (c - 1e-9, c + d), (c + d, c + d), (c - 0.3, c + d)]:
            l, h = min(l, h), max(l, h)
            if -6.3 <= l and h <= 6.3:
                los.append(l); his.append(h)
fpi = 3.141592653589793
for (l, h) in [(fpi - 0.2, fpi), (-fpi, -fpi + 0.2), (fpi, fpi), (-fpi, -fpi), (-2 * fpi, -2 * fpi + 1e-3),
               (2 * fpi - 1e-3, 2 * fpi), (-4.71238898038469, 3.141592653589793)]:
    los.append(l); his.append(h)
los = np.array(los); his = np.array(his)
ci = iv.cos_iv(los, his); si = iv.sin_iv(los, his)
nbc = nbs = 0
for k in range(len(los)):
    pts = [mp.mpf(float(los[k])), mp.mpf(float(his[k]))]
    for t in np.linspace(0, 1, 7)[1:-1]:
        pts.append(mp.mpf(float(los[k])) + mp.mpf(t) * (mp.mpf(float(his[k])) - mp.mpf(float(los[k]))))
    for c in crit:
        cc = mp.mpf(c)
        if pts[0] <= cc <= pts[1]:
            pts.append(cc)
    # true critical points inside
    for kk in range(-2, 3):
        for cc in (kk * mp.pi, mp.pi / 2 + kk * mp.pi):
            if pts[0] <= cc <= pts[1]:
                pts.append(cc)
    for x in pts:
        if not contains(ci, mp.cos(x), k): nbc += 1
        if not contains(si, mp.sin(x), k): nbs += 1
print(f"cos_iv containment failures {nbc}, sin_iv failures {nbs} over {len(los)} intervals")
if nbc or nbs: bad.append('trig')
w = ci.hi - ci.lo
print("max width of cos_iv on point intervals:", np.max(np.where(los == his, w, 0)))

# ---------------------------------------------------------------- cube roots
xs = list(10 ** rng.uniform(-300, 300, 3000)) + list(rng.uniform(0, 10, 3000))
xs += [float(k) ** 3 for k in range(1, 200)] + [(k / 64.0) ** 3 for k in range(1, 200)]
xs += [5e-324, 1e-310, 2.2250738585072014e-308, 0.0, 1.0, 8.0, 27.0, 1e308, 1.7976931348623157e308]
xs = np.array(xs)
lo = iv.cbrt_lo(xs); hi = iv.cbrt_hi(xs)
nlo = nhi = 0
for k in range(len(xs)):
    X = Fr(float(xs[k])); L = Fr(float(lo[k])); H = Fr(float(hi[k]))
    if L ** 3 > X: nlo += 1
    if H ** 3 < X: nhi += 1
print(f"cbrt_lo failures {nlo}, cbrt_hi failures {nhi} over {len(xs)} inputs (exact rational check)")
if nlo or nhi: bad.append('cbrt')
# does the cbrt_hi loop ever exit with an unverified value?  replicate the logic and count
x = np.maximum(xs, 0.0); y = iv.up(iv.up(np.cbrt(x))); it_needed = 0
for it in range(6):
    b_ = iv._cube_dn(y) < x
    if not np.any(b_): break
    it_needed = it + 1
    y = np.where(b_, iv.up(y * (1 + 1e-15)), y)
print("cbrt_hi refinement iterations needed:", it_needed, " unverified at exit:", int(np.sum(iv._cube_dn(y) < x)))
p23 = iv.pow23(I(xs[:200], xs[:200] * 1.5))
n23 = 0
for k in range(200):
    for X in (mp.mpf(float(xs[k])), mp.mpf(float(xs[k] * 1.5))):
        if X > 0 and not contains(p23, X ** (mp.mpf(2) / 3), k): n23 += 1
print("pow23 containment failures", n23)
if n23: bad.append('pow23')
print("ALL OK" if not bad else f"FAILURES: {bad}")
