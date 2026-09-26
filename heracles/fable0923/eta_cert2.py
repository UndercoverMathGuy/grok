"""eta_cert2.py -- two-stage rigorous certificate for Claim(A) of eta_cert.py (same inequality, same monotone box test).

Stage 1 (float64, NOT rigorous): adaptive bisection to find a box cover on which the monotone test passes with a
safety margin 1e-6.  Stage 2 (rigorous): re-evaluate the monotone test on every final box with mpmath interval
arithmetic (outward rounding), no safety margin.  A box passes only if the interval lower bound of the LHS
exceeds the interval upper bound of the RHS.  Output: number of boxes, all passed / list of failures.

Box test on [e0,e1] x [h0,h1]  (r = 2/3, c3 = 3^{1/3}):
   max( e0^r , c3 e0^{2-r} (3/B(e1,h1))^{1-r} ) >= c3 (1 - A h0)_+^r
Domain: e in [1, sqrt3], eta in [0, 1/A]  (outside: trivial, see eta_cert.py).
"""
import sys, numpy as np
from mpmath import iv, mp

r = 2/3; c3 = 3 ** (1 - r)

def test_float(e0, e1, h0, h1, A, margin=1e-6):
    u1 = np.sqrt(2*e1*(e1 - 1)); B = 3*e1**2 + 2*h1**2 + 4*h1*u1 + 2*u1**2
    lhs = max(e0**r, c3 * e0**(2 - r) * (3/B)**(1 - r))
    rhs = c3 * max(1 - A*h0, 0) ** r
    return lhs - rhs >= margin

def cover(A, min_side=1e-6):
    stack = [(1.0, np.sqrt(3), 0.0, 1/A)]; final = []; bad = []
    while stack:
        e0, e1, h0, h1 = stack.pop()
        if 1 - A*h0 <= 0 or test_float(e0, e1, h0, h1, A): final.append((e0, e1, h0, h1)); continue
        de, dh = e1 - e0, h1 - h0
        if de < min_side and dh < min_side: bad.append((e0, e1, h0, h1)); continue
        if de * 3 >= dh: m = (e0 + e1)/2; stack += [(e0, m, h0, h1), (m, e1, h0, h1)]
        else: m = (h0 + h1)/2; stack += [(e0, e1, h0, m), (e0, e1, m, h1)]
    return final, bad

def test_iv(box, A):
    mp.dps = 25; iv.dps = 25
    e0, e1, h0, h1 = [iv.mpf(x) for x in box]; A = iv.mpf(A); R = iv.mpf(2)/3; C3 = iv.mpf(3)**(1 - R)
    base = 1 - A*h0
    if base.b <= 0: return True
    u1 = iv.sqrt(2*e1*(e1 - 1)); B = 3*e1**2 + 2*h1**2 + 4*h1*u1 + 2*u1**2
    lhs1 = e0**R; lhs2 = C3 * e0**(2 - R) * (3/B)**(1 - R)
    lhs_lo = max(lhs1.a, lhs2.a)
    rhs = C3 * iv.mpf(base.b) ** R
    return lhs_lo > rhs.b

if __name__ == "__main__":
    for A in [float(x) for x in sys.argv[1:]] or [0.28]:
        final, bad = cover(A)
        print(f"A={A}: stage-1 boxes {len(final)}, uncovered {len(bad)}", flush=True)
        if bad: print("  first uncovered:", bad[:3]); continue
        fails = [b for b in final if not test_iv(b, A)]
        print(f"  stage-2 interval check: {len(final) - len(fails)} passed, {len(fails)} failed" + ("  -> Claim(A) PROVED" if not fails else f"  e.g. {fails[:3]}"), flush=True)
