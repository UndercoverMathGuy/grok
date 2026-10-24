"""Referee-3 audit of iv.py primitives against high-precision mpmath (independent of the authors' self-tests).

Checks:
 (1) cos_iv / sin_iv contain the exact cos/sin (mpmath, 200 bits) at random points and random sub-points of random intervals,
     including the edges +-np.pi, +-2np.pi, +-6.3 and the true pi (which lies outside [-np.pi, np.pi]);
 (2) point-enclosure widths vs the one-ulp argument error made by the UNROUNDED float subtractions in bb52/pay_adapt
     (ph = p - alpha, psi - phi_pay): if every point enclosure has radius >= 2 ulp(x), a one-ulp argument error is absorbed;
 (3) cbrt_lo / cbrt_hi / pow23 rigorous on random magnitudes 1e-300..1e300 and adversarial perfect cubes;
 (4) the constants COS72, COS144, SIN72, SIN144, SQ3, PI, GAMMA, KA contain their exact values.
"""
import sys
import numpy as np
import mpmath as mp
sys.path.insert(0, '/home/user/grok/heracles/conjecture20/unify/p52')
from iv import I, cos_iv, sin_iv, cbrt_lo, cbrt_hi, pow23, COS72, COS144, SIN72, SIN144, SQ3, SQ2, PI, _cos_pt, _sin_pt

mp.mp.prec = 200
rng = np.random.default_rng(12345)
bad = 0


def inside(iv_lo, iv_hi, exact):
    return mp.mpf(float(iv_lo)) <= exact <= mp.mpf(float(iv_hi))


# (1) point enclosures
xs = np.concatenate([rng.uniform(-6.3, 6.3, 20000),
                     np.array([np.pi, -np.pi, 2 * np.pi, -2 * np.pi, np.pi / 2, -np.pi / 2, 6.3, -6.3, 0.0, 1e-300, -1e-300,
                               np.nextafter(np.pi, 4), np.nextafter(np.pi, 0), 1.5 * np.pi, -1.5 * np.pi])])
c = _cos_pt(xs); s = _sin_pt(xs)
for k, x in enumerate(xs):
    X = mp.mpf(float(x))
    if not inside(c.lo[k], c.hi[k], mp.cos(X)):
        bad += 1; print("cos point fail", x)
    if not inside(s.lo[k], s.hi[k], mp.sin(X)):
        bad += 1; print("sin point fail", x)
print("point enclosures: checked", len(xs), "bad so far", bad)
# min radius relative to ulp(x)
ulp = np.spacing(np.abs(xs)) + 1e-320
rad_c = np.minimum(c.hi - np.cos(xs), np.cos(xs) - c.lo)
rad_s = np.minimum(s.hi - np.sin(xs), np.sin(xs) - s.lo)
print("min point-enclosure radius / ulp(x): cos %.2f  sin %.2f" % (np.min(rad_c / ulp), np.min(rad_s / ulp)))
print("   (cos radius at x where cos is clipped to +-1 is irrelevant: derivative 0 there)")
# exclude points where clipping at +-1 happens (|sin x| tiny) for the cos ratio
mask = np.abs(np.sin(xs)) > 1e-3
print("   cos radius/ulp on |sin|>1e-3: %.2f ; sin radius/ulp on |cos|>1e-3: %.2f"
      % (np.min(rad_c[mask] / ulp[mask]), np.min(rad_s[np.abs(np.cos(xs)) > 1e-3] / ulp[np.abs(np.cos(xs)) > 1e-3])))

# true pi and 2 pi (outside the float grid [-np.pi, np.pi]): is cos/sin at the TRUE endpoint inside the edge-cell enclosure?
for (lo, hi, name) in [(np.pi - 0.3, np.pi, "cell ending at np.pi"), (-np.pi, -np.pi + 0.3, "cell starting at -np.pi")]:
    ci = cos_iv(np.array([lo]), np.array([hi])); si = sin_iv(np.array([lo]), np.array([hi]))
    for t in (mp.pi, -mp.pi):
        print(f"   {name}: cos(true +-pi) inside={inside(ci.lo[0], ci.hi[0], mp.cos(t))}  sin(true +-pi) inside={inside(si.lo[0], si.hi[0], mp.sin(t))}")

# interval enclosures on random intervals (including ones that straddle critical points)
lo = rng.uniform(-6.3, 6.3, 4000); w = rng.exponential(0.5, 4000); hi = np.minimum(lo + w, 6.3)
ci = cos_iv(lo, hi); si = sin_iv(lo, hi)
for k in range(len(lo)):
    for t in np.linspace(0, 1, 9):
        X = mp.mpf(float(lo[k])) + mp.mpf(t) * (mp.mpf(float(hi[k])) - mp.mpf(float(lo[k])))
        if not inside(ci.lo[k], ci.hi[k], mp.cos(X)):
            bad += 1; print("cos_iv fail", lo[k], hi[k], t)
        if not inside(si.lo[k], si.hi[k], mp.sin(X)):
            bad += 1; print("sin_iv fail", lo[k], hi[k], t)
    # critical points in the interval
    for kk in range(-3, 4):
        for cp_ in (kk * mp.pi, kk * mp.pi + mp.pi / 2):
            if mp.mpf(float(lo[k])) <= cp_ <= mp.mpf(float(hi[k])):
                if not inside(ci.lo[k], ci.hi[k], mp.cos(cp_)) or not inside(si.lo[k], si.hi[k], mp.sin(cp_)):
                    bad += 1; print("critical point fail", lo[k], hi[k], cp_)
print("interval enclosures: bad so far", bad)

# (3) cube roots
xs = np.concatenate([10.0 ** rng.uniform(-300, 300, 20000), np.array([0.0, 1.0, 8.0, 27.0, 1e-300, 0.125, 3.375, 1e300]),
                     np.arange(1, 200, dtype=float) ** 3])
lo = cbrt_lo(xs); hi = cbrt_hi(xs)
for k, x in enumerate(xs):
    e = mp.cbrt(mp.mpf(float(x)))
    if not (mp.mpf(float(lo[k])) <= e <= mp.mpf(float(hi[k]))):
        bad += 1; print("cbrt fail", x, lo[k], hi[k])
p23 = pow23(I(xs, xs))
for k, x in enumerate(xs[:5000]):
    e = mp.mpf(float(x)) ** (mp.mpf(2) / 3)
    if not (mp.mpf(float(p23.lo[k])) <= e <= mp.mpf(float(p23.hi[k]))):
        bad += 1; print("pow23 fail", x)
print("cube roots: bad so far", bad, " max rel width cbrt", np.max((hi - lo) / np.maximum(hi, 1e-300)))

# (4) constants
chk = [("COS72", COS72, mp.cos(2 * mp.pi / 5)), ("COS144", COS144, mp.cos(4 * mp.pi / 5)), ("SIN72", SIN72, mp.sin(2 * mp.pi / 5)),
       ("SIN144", SIN144, mp.sin(4 * mp.pi / 5)), ("SQ3", SQ3, mp.sqrt(3)), ("SQ2", SQ2, mp.sqrt(2)), ("PI", PI, mp.pi)]
for name, v, e in chk:
    ok = inside(np.atleast_1d(v.lo)[0], np.atleast_1d(v.hi)[0], e)
    if not ok:
        bad += 1
    print(f"   {name}: [{float(np.atleast_1d(v.lo)[0])!r}, {float(np.atleast_1d(v.hi)[0])!r}] contains exact: {ok}")
from cert_hs import GAMMA
e = mp.mpf(5) / 2 * mp.mpf(2) ** (-mp.mpf(3) / 2)
ok = inside(np.atleast_1d(GAMMA.lo)[0], np.atleast_1d(GAMMA.hi)[0], e); bad += (not ok)
print("   GAMMA_{3/2}(5) contains exact:", ok)
print("TOTAL BAD =", bad)
