# Per-feature moments of the normalised label-row cost rho_t(U) = (1 + t cos U)_+^r / pi(t)^r, U uniform,
# where pi(t) = a/Re(beta) is the smallest admissible IF price per unit clean amplitude:
#   pi(t) = max(1 - c t, 1 - Kq t^2)  (t<=1),   t (1-c)  (t>=1)      [P12, quadratic payment (Note, Prop quadpay(ii)), two-atom cap]
# e1 = E rho - 1 (Jensen excess),  V = Var rho,  k3 = third central moment.
# Degree-2 test Q = (B - C)(C - x0)^2 with C = sum_l a_l^r rho_l: torus mean is positive iff  sigma^2 > d^2 + d k3tot/sigma^2,
# d = sum a^r e1, sigma^2 = sum a^{2r} V, k3tot = sum a^{3r} k3.  Cauchy-Schwarz: d^2 <= sigma^2 * sum_l e1_l^2/V_l.
import numpy as np
from scipy.integrate import quad

def stats(t, s=1.5, P12only=False):
    r = 1/s; c = 1-(2/3)**(s-1); Kq = 5/(5+4*r)
    if t <= 1: pi_t = (1-c*t) if P12only else max(1-c*t, 1-Kq*t*t)
    else: pi_t = t*(1-c)
    f = lambda u: max(1+t*np.cos(u), 0.0)**r/pi_t**r
    m1 = quad(f, 0, np.pi, limit=400)[0]/np.pi
    m2 = quad(lambda u: f(u)**2, 0, np.pi, limit=400)[0]/np.pi
    m3 = quad(lambda u: (f(u)-m1)**3, 0, np.pi, limit=400)[0]/np.pi
    return m1-1, m2-m1**2, m3

if __name__ == '__main__':
    for s in [1.5, 1.25]:
        ts = np.concatenate([np.linspace(0.01, 1, 100), np.linspace(1.01, 2, 25)])
        rows = [(t,)+stats(t, s) for t in ts]
        M = max(e1**2/V for (t, e1, V, k3) in rows if e1 > 0)
        tm = [t for (t, e1, V, k3) in rows if e1 > 0 and abs(e1**2/V - M) < 1e-12][0]
        print(f"s={s}: max_t e1^2/V = {M:.5f} at t={tm:.3f}  ->  Cauchy-Schwarz capacity 1/M = {1/M:.1f} interfering features", flush=True)
        for (t, e1, V, k3) in rows[::10] + [r_ for r_ in rows if abs(r_[0]-tm) < 1e-9]:
            print(f"   t={t:.3f}: e1={e1:+.5f}  V={V:.5f}  k3={k3:+.5f}  e1^2/V={max(e1,0)**2/V:.5f}")
        # equal weights, uniform t: exact degree-2 condition sigma^2 > d^2 + d k3tot/sigma^2  with a^r = 1
        for (t, e1, V, k3) in rows:
            pass
        Kmax = []
        for (t, e1, V, k3) in rows:
            if e1 <= 0: continue
            # K V > K^2 e1^2 + K e1 * (K k3)/(K V)  <=>  V > K e1^2 + e1 k3/V
            Kmax.append(((V - e1*k3/V)/e1**2, t))
        print(f"   equal weights, uniform t: degree-2 test closes for K < {min(Kmax)[0]:.1f} (worst t={min(Kmax)[1]:.3f})", flush=True)
