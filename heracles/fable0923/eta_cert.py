"""eta_cert.py -- interval-arithmetic certificate for the scalar inequality behind (ETA-A), n=2, r=2/3.

Claim(A):  for all e in [1, sqrt3] and eta in [0, 1/A],
     max( e^r ,  3^{1-r} e^{2-r} (3 / B(e,eta))^{1-r} )  >=  3^{1-r} (1 - A eta)^r ,
     B(e,eta) = 3e^2 + 2 eta^2 + 4 eta u + 2 u^2,  u = sqrt(2 e (e-1)).
(Outside: e >= sqrt3 => e^r >= 3^{1-r} >= RHS; eta >= 1/A => RHS = 0.)
Monotonicity used on a box [e0,e1] x [h0,h1]: e^r increasing in e; e^{2-r} increasing in e; B increasing in e and eta
(u increasing in e); RHS decreasing in eta. So the box is certified if
     max( e0^r , 3^{1-r} e0^{2-r} (3/B(e1,h1))^{1-r} ) >= 3^{1-r} (1 - A h0)^r
evaluated with outward rounding (mpmath.iv). Adaptive bisection on the longer side.
"""
import sys, mpmath
from mpmath import iv, mp
mp.dps = 30; iv.dps = 30

def certify(A, max_boxes=2_000_000, min_side=1e-7):
    A = iv.mpf(A); r = iv.mpf(2)/3; c3 = iv.mpf(3)**(1 - r)
    e_hi = iv.sqrt(iv.mpf(3)); h_hi = 1/A
    stack = [(iv.mpf(1), e_hi, iv.mpf(0), h_hi)]
    done = 0; fails = []
    while stack:
        e0, e1, h0, h1 = stack.pop()
        # bounds as mpf (use interval endpoints .a / .b for conservative directions)
        E0 = iv.mpf(e0.a); E1 = iv.mpf(e1.b); H0 = iv.mpf(h0.a); H1 = iv.mpf(h1.b)
        u1 = iv.sqrt(2*E1*(E1 - 1))
        B = 3*E1**2 + 2*H1**2 + 4*H1*u1 + 2*u1**2
        lhs1 = E0**r
        lhs2 = c3 * E0**(2 - r) * (3/B)**(1 - r)
        lhs = lhs1 if lhs1.a >= lhs2.a else lhs2
        rhs_base = 1 - A*H0
        if rhs_base.b <= 0: done += 1; continue
        rhs = c3 * iv.mpf(max(rhs_base.b, 0)) ** r
        if lhs.a >= rhs.b: done += 1; continue
        de = float(e1.b - e0.a); dh = float(h1.b - h0.a)
        if de < min_side and dh < min_side:
            fails.append((float(e0.a), float(h0.a))); continue
        if de * 3 >= dh:          # split e (scale: e in [1,1.73], eta in [0, ~3.6])
            m = (e0 + e1)/2; stack += [(e0, m, h0, h1), (m, e1, h0, h1)]
        else:
            m = (h0 + h1)/2; stack += [(e0, e1, h0, m), (e0, e1, m, h1)]
        if done + len(fails) > max_boxes: break
    return done, fails

if __name__ == "__main__":
    for A in (sys.argv[1:] or ["0.28"]):
        done, fails = certify(A)
        print(f"A = {A}: certified boxes {done}, failed boxes {len(fails)}" + (f" e.g. {fails[:3]}" if fails else "  -> Claim(A) PROVED"), flush=True)
