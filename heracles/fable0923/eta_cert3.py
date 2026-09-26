"""eta_cert3.py -- rigorous certificate for Claim(A): for e in [1, sqrt3], eta in [0, 1/A],
     max( e^r , 3^{1-r} e^{2-r} (3/B)^{1-r} ) >= 3^{1-r} (1 - A eta)_+^r,   r = 2/3,
     B = 3e^2 + 2eta^2 + 4 eta u + 2u^2,  u = sqrt(2e(e-1)).

Branch 2 alone suffices where Delta(x,eta) := 3e^4 - (1 - A eta)^2 B >= 0 (cube both sides; c3^3 = 3), e = 1+x.

CORNER LEMMA (analytic): on R0 = {0 <= x <= x0, 0 <= eta <= eta0}, with 4 eta u <= 4 eta^2 + u^2 (2ab <= a^2+b^2) and
u^2 = 2x + 2x^2, expanding Delta exactly gives
  Delta >= 9x^2 + 6A eta - eta^2 [ 6 + 3A^2 + 10A^2 x + 7A^2 x^2 + 2A^2 eta^2 + 4A^2 eta u ]   (all dropped terms >= 0)
so Delta >= 0 on R0 as soon as  6A >= eta0 * [6 + 3A^2 + 10A^2 x0 + 7A^2 x0^2 + 2A^2 eta0^2 + 4A^2 eta0 sqrt(2x0(1+x0))].
The exact expansion of Delta is re-derived symbolically below (sympy) so no hand algebra is trusted.

REST: two-stage box cover (float64 search, then mpmath interval re-check of every box) on
     [1, 1+x0] x [eta0, 1/A]   and   [1+x0, sqrt3] x [0, 1/A].
"""
import sys, numpy as np, sympy as sp
from mpmath import iv, mp
r = 2/3; c3 = 3 ** (1 - r)

def corner_symbolic():
    x, h, A, u = sp.symbols('x eta A u', nonnegative=True)
    e = 1 + x
    B = 3*e**2 + 2*h**2 + 4*h*u + 2*u**2
    Delta = sp.expand(3*e**4 - (1 - A*h)**2 * B)
    # substitute u^2 = 2x + 2x^2 where it appears as u**2, keep odd powers of u
    Delta = sp.expand(Delta.subs(u**2, 2*x + 2*x**2))
    # after the AM-GM step 4*h*u <= 4*h^2 + u^2 (used only on the term -4*h*u, coefficient -1 in the (1-Ah)^2 expansion):
    # the terms containing u are: -4 h u * (1 - 2A h + A^2 h^2) = -4hu + 8A h^2 u - 4A^2 h^3 u.
    # Replace -4hu by -(4h^2 + 2x + 2x^2)  (a lower bound), keep +8A h^2 u >= 0 (drop), keep -4A^2 h^3 u.
    coeffs = sp.Poly(Delta, u).all_coeffs()      # Delta = c1*u + c0
    c1, c0 = coeffs if len(coeffs) == 2 else (0, coeffs[0])
    print("Delta = c0 + c1*u with c1 =", sp.factor(c1))
    lower = sp.expand(c0 - (4*h**2 + 2*x + 2*x**2) - 4*A**2*h**3*u)   # -4hu -> -(4h^2+u^2); +8Ah^2u dropped
    print("lower bound on Delta (after AM-GM, dropping +8A eta^2 u):", sp.collect(lower, [x, h]))
    return lower

def corner_check(A, x0, eta0):
    mp.dps = 30; iv.dps = 30
    A = iv.mpf(A); x0 = iv.mpf(x0); eta0 = iv.mpf(eta0)
    br = 6 + 3*A**2 + 10*A**2*x0 + 7*A**2*x0**2 + 2*A**2*eta0**2 + 4*A**2*eta0*iv.sqrt(2*x0*(1 + x0))
    lhs = 6*A; rhs = eta0*br
    print(f"corner: 6A = {lhs}  vs  eta0*bracket = {rhs}  ->", "OK" if lhs.a > rhs.b else "FAIL")
    return lhs.a > rhs.b

def test_float(e0, e1, h0, h1, A, margin=1e-7):
    u1 = np.sqrt(2*e1*(e1 - 1)); B = 3*e1**2 + 2*h1**2 + 4*h1*u1 + 2*u1**2
    lhs = max(e0**r, c3 * e0**(2 - r) * (3/B)**(1 - r)); rhs = c3 * max(1 - A*h0, 0) ** r
    return lhs - rhs >= margin

def cover(A, dom, min_side=1e-7):
    stack = [dom]; final = []; bad = []
    while stack:
        e0, e1, h0, h1 = stack.pop()
        if 1 - A*h0 <= 0 or test_float(e0, e1, h0, h1, A): final.append((e0, e1, h0, h1)); continue
        de, dh = e1 - e0, h1 - h0
        if de < min_side and dh < min_side: bad.append((e0, e1, h0, h1)); continue
        if de * 3 >= dh: m = (e0 + e1)/2; stack += [(e0, m, h0, h1), (m, e1, h0, h1)]
        else: m = (h0 + h1)/2; stack += [(e0, e1, h0, m), (e0, e1, m, h1)]
    return final, bad

def test_iv(box, A):
    e0, e1, h0, h1 = [iv.mpf(x) for x in box]; A = iv.mpf(A); R = iv.mpf(2)/3; C3 = iv.mpf(3)**(1 - R)
    base = 1 - A*h0
    if base.b <= 0: return True
    u1 = iv.sqrt(2*e1*(e1 - 1)); B = 3*e1**2 + 2*h1**2 + 4*h1*u1 + 2*u1**2
    lhs_lo = max((e0**R).a, (C3 * e0**(2 - R) * (3/B)**(1 - R)).a)
    return lhs_lo > (C3 * iv.mpf(base.b) ** R).b

if __name__ == "__main__":
    corner_symbolic()
    x0, eta0 = 0.2, 0.25
    for A in [float(a) for a in sys.argv[1:]] or [0.275]:
        print(f"== A = {A}")
        ok = corner_check(A, x0, eta0)
        mp.dps = 25; iv.dps = 25
        doms = [(1.0, 1 + x0, eta0, 1/A), (1 + x0, np.sqrt(3), 0.0, 1/A)]
        tot = 0; allok = ok
        for dom in doms:
            final, bad = cover(A, dom)
            fails = [b for b in final if not test_iv(b, A)] if not bad else []
            print(f"  domain {dom}: boxes {len(final)}, uncovered {len(bad)}, interval failures {len(fails)}")
            allok &= (not bad) and (not fails); tot += len(final)
        print(f"  => Claim({A}) {'PROVED' if allok else 'NOT proved'} ({tot} boxes + corner lemma on [1,{1+x0}]x[0,{eta0}])")
