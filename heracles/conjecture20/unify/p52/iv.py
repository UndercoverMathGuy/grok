"""Minimal vectorised interval arithmetic with outward rounding (numpy float64).

Rigour: IEEE-754 round-to-nearest makes +, -, *, /, sqrt correctly rounded, so the exact result of an operation on
floats lies within one ulp of the computed float; every operation below widens its result by one ulp in each direction
(np.nextafter).  Cube roots are verified by cubing.  cos/sin are enclosed by Taylor polynomials with a rigorous
remainder bound, evaluated in interval arithmetic.  No other transcendental function is used.
"""
import numpy as np

INF = np.inf


def dn(x):
    return np.nextafter(x, -INF)


def up(x):
    return np.nextafter(x, INF)


class I:
    __slots__ = ('lo', 'hi')

    def __init__(self, lo, hi=None):
        self.lo = np.asarray(lo, dtype=float)
        self.hi = self.lo.copy() if hi is None else np.asarray(hi, dtype=float)

    @staticmethod
    def c(x):
        """enclosure of a float constant (exact)"""
        return I(x, x)

    def __repr__(self):
        return f"I({self.lo}, {self.hi})"

    def __add__(a, b):
        b = tI(b)
        return I(dn(a.lo + b.lo), up(a.hi + b.hi))

    __radd__ = __add__

    def __sub__(a, b):
        b = tI(b)
        return I(dn(a.lo - b.hi), up(a.hi - b.lo))

    def __rsub__(a, b):
        return tI(b) - a

    def __neg__(a):
        return I(-a.hi, -a.lo)

    def __mul__(a, b):
        b = tI(b)
        p1 = a.lo * b.lo; p2 = a.lo * b.hi; p3 = a.hi * b.lo; p4 = a.hi * b.hi
        lo = np.minimum(np.minimum(p1, p2), np.minimum(p3, p4))
        hi = np.maximum(np.maximum(p1, p2), np.maximum(p3, p4))
        return I(dn(lo), up(hi))

    __rmul__ = __mul__

    def __truediv__(a, b):
        b = tI(b)
        if np.any((b.lo <= 0) & (b.hi >= 0)):
            raise ZeroDivisionError("interval division by interval containing 0")
        q1 = a.lo / b.lo; q2 = a.lo / b.hi; q3 = a.hi / b.lo; q4 = a.hi / b.hi
        lo = np.minimum(np.minimum(q1, q2), np.minimum(q3, q4))
        hi = np.maximum(np.maximum(q1, q2), np.maximum(q3, q4))
        return I(dn(lo), up(hi))

    def __rtruediv__(a, b):
        return tI(b) / a

    def sq(a):
        lo2 = a.lo * a.lo; hi2 = a.hi * a.hi
        lo = np.where((a.lo <= 0) & (a.hi >= 0), 0.0, np.minimum(lo2, hi2))
        hi = np.maximum(lo2, hi2)
        return I(np.maximum(dn(lo), 0.0), up(hi))

    def sqrt(a):
        lo = np.maximum(a.lo, 0.0)
        return I(np.maximum(dn(np.sqrt(lo)), 0.0), up(np.sqrt(np.maximum(a.hi, 0.0))))

    def maxc(a, x):
        """max(a, x) for a float/array x"""
        return I(np.maximum(a.lo, x), np.maximum(a.hi, x))

    def clip0(a):
        return I(np.maximum(a.lo, 0.0), np.maximum(a.hi, 0.0))


def tI(x):
    return x if isinstance(x, I) else I(x, x)


def imax(a, b):
    return I(np.maximum(a.lo, b.lo), np.maximum(a.hi, b.hi))


def imin(a, b):
    return I(np.minimum(a.lo, b.lo), np.minimum(a.hi, b.hi))


# ---------------------------------------------------------------- cube root (verified) and x^{2/3}
def _cube_up(y):
    return up(up(y * y) * y)


def _cube_dn(y):
    return dn(dn(y * y) * y)


def cbrt_lo(x):
    """rigorous lower bound of x^{1/3} for x >= 0 (array)"""
    x = np.maximum(np.asarray(x, float), 0.0)
    y = dn(dn(np.cbrt(x)))
    for _ in range(6):
        bad = _cube_up(y) > x
        if not np.any(bad):
            break
        y = np.where(bad, dn(y * (1 - 1e-15)), y)
    y = np.where(_cube_up(y) > x, 0.0, y)
    return np.maximum(y, 0.0)


def cbrt_hi(x):
    x = np.maximum(np.asarray(x, float), 0.0)
    y = up(up(np.cbrt(x)))
    for _ in range(6):
        bad = _cube_dn(y) < x
        if not np.any(bad):
            break
        y = np.where(bad, up(y * (1 + 1e-15)), y)
    return y


def pow23(a):
    """enclosure of a^{2/3} for a >= 0 (interval clipped at 0)"""
    a = a.clip0()
    lo = cbrt_lo(a.lo); hi = cbrt_hi(a.hi)
    return I(dn(lo * lo), up(hi * hi))


# ---------------------------------------------------------------- constants
SQ5 = I(5.0).sqrt()
PI = I(3.141592653589793, 3.1415926535897936)       # floats bracketing pi
COS72 = (SQ5 - 1.0) / 4.0
COS144 = -(SQ5 + 1.0) / 4.0
SIN72 = ((SQ5 + 5.0) / 8.0).sqrt()
SIN144 = ((5.0 - SQ5) / 8.0).sqrt()
SQ2 = I(2.0).sqrt()
SQ3 = I(3.0).sqrt()


# ---------------------------------------------------------------- cos / sin on point arrays via Taylor + remainder
def _cos_pt(x):
    """enclosure of cos(x) for float array x, |x| <= 7"""
    xi = I(x, x)
    x2 = xi * xi
    term = I(1.0); tot = I(1.0)
    for k in range(1, 22):
        term = term * x2 / float((2 * k - 1) * (2 * k))
        tot = tot + term * (-1.0 if k % 2 else 1.0)
    # remainder |x|^{44}/44!  (k = 22 term bound)
    rem = (np.abs(x) ** 44) / 2.6582715747884487e54 * (1 + 1e-6)
    return I(np.maximum(dn(tot.lo - rem), -1.0), np.minimum(up(tot.hi + rem), 1.0))


def _sin_pt(x):
    xi = I(x, x)
    x2 = xi * xi
    term = xi; tot = xi
    for k in range(1, 22):
        term = term * x2 / float((2 * k) * (2 * k + 1))
        tot = tot + term * (-1.0 if k % 2 else 1.0)
    rem = (np.abs(x) ** 45) / 1.1962222086548019e56 * (1 + 1e-6)
    return I(np.maximum(dn(tot.lo - rem), -1.0), np.minimum(up(tot.hi + rem), 1.0))


def cos_iv(lo, hi):
    """enclosure of {cos t : t in [lo, hi]} for float arrays with -6.3 <= lo <= hi <= 6.3"""
    lo = np.asarray(lo, float); hi = np.asarray(hi, float)
    a = _cos_pt(lo); b = _cos_pt(hi)
    rlo = np.minimum(a.lo, b.lo); rhi = np.maximum(a.hi, b.hi)
    # interior maxima at 2k pi (k=-1,0,1), minima at (2k+1) pi
    for k in (-2, 0, 2):
        c_lo = k * PI.lo if k >= 0 else k * PI.hi
        c_hi = k * PI.hi if k >= 0 else k * PI.lo
        inside = (lo <= c_hi) & (hi >= c_lo)
        rhi = np.where(inside, 1.0, rhi)
    for k in (-1, 1):
        c_lo = k * PI.lo if k >= 0 else k * PI.hi
        c_hi = k * PI.hi if k >= 0 else k * PI.lo
        inside = (lo <= c_hi) & (hi >= c_lo)
        rlo = np.where(inside, -1.0, rlo)
    return I(rlo, rhi)


def sin_iv(lo, hi):
    """sin t = cos(t - pi/2); enclose directly: extrema at pi/2 + k pi"""
    lo = np.asarray(lo, float); hi = np.asarray(hi, float)
    a = _sin_pt(lo); b = _sin_pt(hi)
    rlo = np.minimum(a.lo, b.lo); rhi = np.maximum(a.hi, b.hi)
    half = PI / 2.0
    for k, ismax in ((-2, False), (-1, True), (0, True), (1, False), (2, True)):
        # critical points pi/2 + j pi : j = -2 -> -3pi/2 (max), j=-1 -> -pi/2 (min), j=0 -> pi/2 (max), j=1 -> 3pi/2 (min)
        pass
    for j in (-2, -1, 0, 1):
        cpt = half + PI * float(j)
        inside = (lo <= cpt.hi) & (hi >= cpt.lo)
        if j % 2 == 0:      # pi/2 + 2m pi -> max
            rhi = np.where(inside, 1.0, rhi)
        else:
            rlo = np.where(inside, -1.0, rlo)
    return I(rlo, rhi)


if __name__ == '__main__':
    # self-checks against numpy
    xs = np.linspace(-6.3, 6.3, 2001)
    c = _cos_pt(xs); s_ = _sin_pt(xs)
    assert np.all(c.lo <= np.cos(xs)) and np.all(np.cos(xs) <= c.hi)
    assert np.all(s_.lo <= np.sin(xs)) and np.all(np.sin(xs) <= s_.hi)
    print("cos/sin point enclosure widths:", np.max(c.hi - c.lo), np.max(s_.hi - s_.lo))
    lo = np.random.default_rng(0).uniform(-6.2, 6.0, 5000); hi = lo + np.random.default_rng(1).uniform(0, 0.3, 5000)
    ci = cos_iv(lo, hi); si = sin_iv(lo, hi)
    for t in np.linspace(0, 1, 11):
        x = lo + t * (hi - lo)
        assert np.all(ci.lo <= np.cos(x) + 1e-15) and np.all(np.cos(x) <= ci.hi + 1e-15)
        assert np.all(si.lo <= np.sin(x) + 1e-15) and np.all(np.sin(x) <= si.hi + 1e-15)
    x = np.random.default_rng(2).uniform(0, 10, 10000)
    assert np.all(cbrt_lo(x) ** 3 <= x * (1 + 1e-15)) and np.all(cbrt_hi(x) ** 3 >= x * (1 - 1e-15))
    print("COS72", COS72, np.cos(2 * np.pi / 5), "SIN72", SIN72, np.sin(2 * np.pi / 5))
    print("COS144", COS144, np.cos(4 * np.pi / 5), "SIN144", SIN144, np.sin(4 * np.pi / 5))
    print("iv self-checks passed")
