"""Certify Claim_PSD(a, kappa):  max(e^{2/3}, 3^{1/3} e^{4/3} (3/B)^{1/3}) >= kappa (1 - a eta)_+^{2/3}
on e in [1, sqrt3], 0 <= eta <= sqrt3 e   (B = 3e^2 + 2eta^2 + 4 eta u + 2u^2, u = sqrt(2e(e-1))).
With fable0923 Lemma 3's reduction (G normalised to 1, e = sum|c| >= 1, every |J_T| <= e so eta <= sqrt3 e; e >= sqrt3 is
covered by the first branch since kappa <= 3^{1/3}), this proves  cost >= kappa (|G| - a eta)_+^{2/3}  for every population.
Box cover: float64 search, then every accepted box re-checked in mpmath interval arithmetic (outward rounding).
Monotone box test: LHS >= max(e0^{2/3}, 3^{1/3} e0^{4/3} (3/B(e1,h1))^{1/3}), RHS <= kappa (1 - a h0)_+^{2/3}."""
import sys, numpy as np
from mpmath import iv
iv.dps = 30
S3 = np.sqrt(3)
CROSS = 4.0   # 4 = fable single chain order; 2 sqrt2 = two chain orders (min <= average)
def lhs_f(e0, e1, h1):
    u1 = np.sqrt(2 * e1 * (e1 - 1)); B = 3 * e1 ** 2 + 2 * h1 ** 2 + CROSS * h1 * u1 + 2 * u1 ** 2
    return max(e0 ** (2 / 3), 3 ** (1 / 3) * e0 ** (4 / 3) * (3 / B) ** (1 / 3))
def cover(a, kap, min_side=1e-9):
    stack = [(1.0, S3, 0.0, 3.0)]; final = []; bad = []
    while stack:
        e0, e1, h0, h1 = stack.pop()
        if h0 > S3 * e1 or 1 - a * h0 <= 0:                      # infeasible (eta > sqrt3 e) or RHS = 0
            final.append((e0, e1, h0, h1)); continue
        if lhs_f(e0, e1, h1) - kap * (1 - a * h0) ** (2 / 3) > 1e-9:
            final.append((e0, e1, h0, h1)); continue
        de, dh = e1 - e0, h1 - h0
        if de < min_side and dh < min_side: bad.append((e0, e1, h0, h1)); continue
        if de * 3 >= dh: m = (e0 + e1) / 2; stack += [(e0, m, h0, h1), (m, e1, h0, h1)]
        else: m = (h0 + h1) / 2; stack += [(e0, e1, h0, m), (e0, e1, m, h1)]
    return final, bad
def iv_check(a, kap, box):
    e0, e1, h0, h1 = [iv.mpf(float(t)) for t in box]
    A = iv.mpf(float(a)); K = iv.mpf(float(kap)); s3 = iv.sqrt(3)
    if (h0 - s3 * e1).a > 0: return True
    rhs_base = 1 - A * h0
    if rhs_base.b <= 0: return True
    cross = iv.mpf(4) if CROSS == 4.0 else iv.sqrt(iv.mpf(8))
    u1 = iv.sqrt(2 * e1 * (e1 - 1)); B = 3 * e1 ** 2 + 2 * h1 ** 2 + cross * h1 * u1 + 2 * u1 ** 2
    third = iv.mpf(1) / 3
    br1 = iv.exp(2 * third * iv.log(e0))
    br2 = iv.exp(third * iv.log(iv.mpf(3))) * iv.exp(4 * third * iv.log(e0)) * iv.exp(third * iv.log(3 / B))
    rb = rhs_base if rhs_base.a > 0 else iv.mpf([0, rhs_base.b])
    rhs = K * iv.exp(2 * third * iv.log(rb)) if rb.a > 0 else K * iv.mpf([0, 1]) * iv.exp(2 * third * iv.log(iv.mpf(rb.b)))
    lhs = br1 if br1.a > br2.a else br2
    return lhs.a > rhs.b
if __name__ == '__main__':
    a = float(sys.argv[1]); kap = float(sys.argv[2])
    if len(sys.argv) > 3 and sys.argv[3] == 'two': CROSS = 2 * np.sqrt(2)
    final, bad = cover(a, kap)
    fails = sum(not iv_check(a, kap, b) for b in final)
    print(f"a={a} kappa={kap}: boxes {len(final)}, unresolved {len(bad)}, interval failures {fails} ->",
          "CERTIFIED" if not bad and not fails else "NOT certified", flush=True)
