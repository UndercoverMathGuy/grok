import sys
sys.argv = ['cert_pt.py', 'pt', '1.8553', '0.845', '0.005', '0.02']
from fractions import Fraction as Fr
import cert_pt as cp
from ptinner import F
a_lo, a_hi, m_lo, m_hi = Fr(30,32), Fr(31,32), Fr(15,32), Fr(16,32)
G = cp.G_up(a_lo, m_hi)
Pst = cp.mpf2fr((cp.kP * cp.ivF(m_hi) ** 2 + 2 * G * G).b)
Tst = cp.mpf2fr((-cp.kT * cp.ivF(m_hi) ** 2 * G).a)
print("G", G, "P*", float(Pst), "T*", float(Tst))
print("inner_low", cp.inner_low(Pst, Tst), " float F:", F(float(Pst), float(Tst), 1/1.8553))
print("lhs", float(a_hi - Fr(0.845)*m_lo), "theta", float(cp.mpf2fr(cp.THETA_LOW)))
v = cp.inner_low(Pst, Tst)
print("type", type(v), "mpf2fr ->", float(cp.mpf2fr(v)), cp.mpf2fr(v))
import mpmath
print("mpmath.mpf(v) =", mpmath.mpf(v), type(mpmath.mpf(v)))
print("THETA_LOW type", type(cp.THETA_LOW), cp.THETA_LOW)
