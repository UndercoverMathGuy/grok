import sys, time
sys.argv = ['cert_pt.py', 'pt', '1.5', '0.6']
from fractions import Fraction as Fr
import cert_pt as cp
def onebox(a_lo, m_hi):
    G = cp.G_up(a_lo, m_hi)
    Pst = cp.mpf2fr((cp.kP * cp.ivF(m_hi) ** 2 + 2 * G * G).b)
    Tst = cp.mpf2fr((-cp.kT * cp.ivF(m_hi) ** 2 * G).a)
    return cp.inner_low(Pst, Tst)
slow = []
for i in range(0, 32, 3):
    for j in range(0, 32, 3):
        t = time.time(); onebox(Fr(i, 32), Fr(j+1, 32)); dt = time.time() - t
        slow.append((dt, i, j))
slow.sort(reverse=True)
print(slow[:8])
