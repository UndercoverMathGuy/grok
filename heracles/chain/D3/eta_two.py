"""Sharper eta-form cost side via TWO chain orders:  cost >= 3^{1/3} (|G| - A eta)_+^{2/3}  at A = 0.26 (V3b had 0.272).
Reduction (fable0923 Lemma 3 + two orders): G = 1, e = sum|c| >= 1, u = sqrt(2e(e-1)); the slot-1-first and slot-2-first chain
matrices M1, M2 are PSD with trace 3e and tr M1^2 + tr M2^2 <= 2[3e^2 + 2eta^2 + 2u(|J1|+|J2|) + 2u^2], |J1|+|J2| <= sqrt2 eta, so
min tr M_i^2 <= B := 3e^2 + 2eta^2 + 2 sqrt2 eta u + 2u^2, and cost >= max(e^{2/3}, 3^{1/3} e^{4/3} (3/B)^{1/3}).
Claim(A): that max >= 3^{1/3}(1 - A eta)_+^{2/3} on e in [1, sqrt3], 0 <= eta <= sqrt3 e.
CORNER: Delta = 3e^4 - (1 - A eta)^2 B; with 2 sqrt2 eta u <= 2 eta^2 + u^2 and u^2 = 2x + 2x^2 (e = 1 + x):
  Delta >= 9x^2 + 6A eta - eta^2 [4 + 3A^2 + 10A^2 x + 7A^2 x^2 + 2A^2 eta^2 + 2 sqrt2 A^2 eta u]   (dropped terms >= 0; sympy check below).
REST: box cover, float search then mpmath interval re-check."""
import sys, numpy as np, sympy as sp
from mpmath import iv
iv.dps = 30
A = float(sys.argv[1]) if len(sys.argv) > 1 else 0.26
X0, H0 = float(sys.argv[2]) if len(sys.argv) > 2 else 0.2, float(sys.argv[3]) if len(sys.argv) > 3 else 0.3
def symbolic():
    x, h, a, u = sp.symbols('x eta A u', nonnegative=True); c = 2 * sp.sqrt(2)
    e = 1 + x; B = 3 * e ** 2 + 2 * h ** 2 + c * h * u + 2 * u ** 2
    P = sp.Poly(sp.expand(3 * e ** 4 - (1 - a * h) ** 2 * B), u)          # d0 + d1 u + d2 u^2 (u kept as an independent symbol)
    d = {m[0]: cf for m, cf in zip(P.monoms(), P.coeffs())}
    D = sp.expand(d.get(0, 0) + d.get(2, 0) * (2 * x + 2 * x ** 2) + d.get(1, 0) * u)
    # D = D0 + D1 * u with D1 = -c h (1 - a h)^2 = -c h + 2 c a h^2 - c a^2 h^3
    D1 = sp.expand(D.coeff(u, 1)); D0 = sp.expand(D - D1 * u)
    assert sp.simplify(D1 - sp.expand(-c * h * (1 - a * h) ** 2)) == 0
    lower = sp.expand(D0 - (2 * h ** 2 + 2 * x + 2 * x ** 2) - c * a ** 2 * h ** 3 * u)   # -c h u -> -(2h^2+u^2); +2c a h^2 u dropped
    claimed = 9 * x ** 2 + 6 * a * h - h ** 2 * (4 + 3 * a ** 2 + 10 * a ** 2 * x + 7 * a ** 2 * x ** 2 + 2 * a ** 2 * h ** 2 + c * a ** 2 * h * u)
    resid = sp.expand(lower - claimed)
    poly = sp.Poly(resid, x, h, u)
    print("residual (lower - claimed) =", resid)
    good = all(all(t >= 0 for t in sp.Poly(cf, a).coeffs()) for cf in poly.coeffs())
    print("all residual coefficients are polynomials in A with nonnegative coefficients:", good)
    assert good
def corner_ok():
    a = iv.mpf(A); x0 = iv.mpf(X0); h0 = iv.mpf(H0); c = iv.sqrt(iv.mpf(8))
    br = 4 + 3 * a ** 2 + 10 * a ** 2 * x0 + 7 * a ** 2 * x0 ** 2 + 2 * a ** 2 * h0 ** 2 + c * a ** 2 * h0 * iv.sqrt(2 * x0 * (1 + x0))
    ok = (6 * a).a > (h0 * br).b
    print(f"corner [1,{1 + X0}]x[0,{H0}]: 6A = {6 * A:.4f} vs eta0*bracket <= {float((h0 * br).b):.4f} ->", "OK" if ok else "FAIL")
    return ok
S3 = np.sqrt(3); C = 2 * np.sqrt(2); K = 3 ** (1 / 3)
def ftest(e0, e1, h0, h1):
    u1 = np.sqrt(2 * e1 * (e1 - 1)); B = 3 * e1 ** 2 + 2 * h1 ** 2 + C * h1 * u1 + 2 * u1 ** 2
    return max(e0 ** (2 / 3), K * e0 ** (4 / 3) * (3 / B) ** (1 / 3)) - K * max(1 - A * h0, 0) ** (2 / 3) > 1e-9
def itest(box):
    e0, e1, h0, h1 = [iv.mpf(float(t)) for t in box]; a = iv.mpf(A); c = iv.sqrt(iv.mpf(8)); t3 = iv.mpf(1) / 3
    if (h0 - iv.sqrt(3) * e1).a > 0: return True
    rb = 1 - a * h0
    if rb.b <= 0: return True
    u1 = iv.sqrt(2 * e1 * (e1 - 1)); B = 3 * e1 ** 2 + 2 * h1 ** 2 + c * h1 * u1 + 2 * u1 ** 2
    k = iv.exp(t3 * iv.log(iv.mpf(3)))
    l1 = iv.exp(2 * t3 * iv.log(e0)); l2 = k * iv.exp(4 * t3 * iv.log(e0)) * iv.exp(t3 * iv.log(3 / B))
    r = k * iv.exp(2 * t3 * iv.log(iv.mpf(rb.b)))
    return max(l1.a, l2.a) > r.b
def cover(dom):
    stack = [dom]; fin = []; bad = []
    while stack:
        e0, e1, h0, h1 = stack.pop()
        if h0 > S3 * e1 or 1 - A * h0 <= 0 or ftest(e0, e1, h0, h1): fin.append((e0, e1, h0, h1)); continue
        if e1 - e0 < 1e-9 and h1 - h0 < 1e-9: bad.append((e0, e1, h0, h1)); continue
        if (e1 - e0) * 3 >= h1 - h0: m = (e0 + e1) / 2; stack += [(e0, m, h0, h1), (m, e1, h0, h1)]
        else: m = (h0 + h1) / 2; stack += [(e0, e1, h0, m), (e0, e1, m, h1)]
    return fin, bad
if __name__ == '__main__':
    symbolic()
    ok = corner_ok()
    tot_fail = 0; nbox = 0
    for dom in [(1.0, 1 + X0, H0, 3.0), (1 + X0, 1.7321, 0.0, 3.0)]:   # 1.7321 > sqrt3: covers the float rounding strip
        fin, bad = cover(dom); fails = sum(not itest(b) for b in fin); nbox += len(fin); tot_fail += fails + len(bad)
        print(f"domain {dom}: boxes {len(fin)}, unresolved {len(bad)}, interval failures {fails}")
    print(f"=> Claim({A}) with two chain orders:", "PROVED" if ok and tot_fail == 0 else "NOT proved", f"({nbox} boxes + corner lemma)")
