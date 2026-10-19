import sys, time, cProfile, pstats
sys.argv = ['cert_pt.py', 'pt', '1.8553', '0.845']
from fractions import Fraction as Fr
import cert_pt as cp
from mpmath import iv
def onebox(a_lo, a_hi, m_lo, m_hi):
    G = cp.G_up(a_lo, m_hi)
    Pst = cp.mpf2fr((cp.kP * cp.ivF(m_hi) ** 2 + 2 * G * G).b)
    Tst = cp.mpf2fr((-cp.kT * cp.ivF(m_hi) ** 2 * G).a)
    return cp.inner_low(Pst, Tst)
t = time.time()
pr = cProfile.Profile(); pr.enable()
for i in range(5):
    onebox(Fr(29, 32), Fr(30, 32), Fr(8, 32), Fr(9, 32))
pr.disable()
print("per box", (time.time()-t)/5)
pstats.Stats(pr).sort_stats('cumulative').print_stats(12)
