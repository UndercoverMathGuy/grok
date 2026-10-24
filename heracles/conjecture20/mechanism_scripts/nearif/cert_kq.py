# Interval certificate for Proposition quadpay(ii) (relations case, n=2):
#   g(d) = phi(d)^s - sqrt(1-d^2) >= k1(s) d^2   for s in [1, 3/2], d in [0, 1],   k1(s) = 0.9 (5+4r)/18,  r = 1/s,
# where phi(d) = 3^(r-1) * ((1+d)^r (3+d)^(1-r) + (1-d)^r (3-d)^(1-r)) / 2   (the P12 cost function, normalised).
# Consequence: |A| - 3^(1-s) Q^s <= |D|^2 / (4 k1 L) = 5/(5+4r) * |D|^2 / L.
# Three regions:
#  R1  d in [0, 0.21]: phi^s >= 1 - s(1-phi) (Bernoulli), 1 - sqrt(1-d^2) = d^2/(1+sqrt(1-d^2)), and
#      1 - phi(d) = -3^(r-1) f''(xi) d^2 / 2 for some |xi| <= d, f(x) = (1+x)^r (3+x)^(1-r); f'' bounded on intervals.
#  R2  d in [0.21, 0.5]: interval evaluation in d at the centre of each s-box, plus (w/2) * sup |dG/ds| (mean value in s).
#  R3  d in [0.5, 1]: direct interval evaluation on (s, d) boxes.
from mpmath import iv
iv.dps = 30
def k1(r): return iv.mpf('0.9') * (5 + 4 * r) / 18
def f2(x, r):
    f = (1 + x)**r * (3 + x)**(1 - r); a = r / (1 + x) + (1 - r) / (3 + x)
    return f * (a**2 - r / (1 + x)**2 - (1 - r) / (3 + x)**2)
def parts(d, s):
    r = 1 / s; Fp = (1 + d)**r * (3 + d)**(1 - r); Fm = (1 - d)**r * (3 - d)**(1 - r); c = iv.mpf(3)**(r - 1)
    phi = c * (Fp + Fm) / 2
    dphi_dr = phi * iv.log(iv.mpf(3)) + c * (Fp * iv.log((1 + d) / (3 + d)) + Fm * iv.log((1 - d) / (3 - d))) / 2
    return phi, dphi_dr
def G(d, s):
    phi, _ = parts(d, s); return phi**s - iv.sqrt(1 - d**2) - k1(1 / s) * d**2
def dG(d, s):
    phi, dr = parts(d, s); return phi**s * (iv.log(phi) - dr / (s * phi)) + iv.mpf('0.2') * d**2 / s**2
fails = 0; worst = {1: 9.0, 2: 9.0, 3: 9.0}; nbox = 0
# R1
NS, ND, D1 = 100, 84, 0.21
for i in range(NS):
    s = iv.mpf([1 + 0.5 * i / NS, 1 + 0.5 * (i + 1) / NS]); r = 1 / s; run = None
    for j in range(ND):
        da, db = D1 * j / ND, D1 * (j + 1) / ND
        for x in (iv.mpf([da, db]), iv.mpf([-db, -da])):
            v = -f2(x, r); run = v.b if run is None else max(run, v.b)
        lb = (1 / (1 + iv.sqrt(iv.mpf(1 - da**2))) - s * iv.mpf(3)**(r - 1) * iv.mpf(run) / 2 - k1(r)).a
        nbox += 1; worst[1] = min(worst[1], float(lb)); fails += lb <= 0
# R2
NS, ND, a0, a1 = 200, 145, 0.21, 0.5
for i in range(NS):
    lo, hi = 1 + 0.5 * i / NS, 1 + 0.5 * (i + 1) / NS; w = hi - lo
    sc = iv.mpf((lo + hi) / 2); sbox = iv.mpf([lo, hi])
    for j in range(ND):
        d = iv.mpf([a0 + (a1 - a0) * j / ND, a0 + (a1 - a0) * (j + 1) / ND])
        der = dG(d, sbox); m = max(abs(der.a), abs(der.b))
        lb = (G(d, sc) - m * w / 2).a
        nbox += 1; worst[2] = min(worst[2], float(lb)); fails += lb <= 0
# R3
NS, ND, a0 = 1000, 200, 0.5
for i in range(NS):
    s = iv.mpf([1 + 0.5 * i / NS, 1 + 0.5 * (i + 1) / NS])
    for j in range(ND):
        da, db = a0 + (1 - a0) * j / ND, a0 + (1 - a0) * (j + 1) / ND; d = iv.mpf([da, db])
        phi, _ = parts(d, s)
        g = phi**s - iv.sqrt(iv.mpf([max(0.0, 1 - db**2), 1 - da**2])) - k1(1 / s) * d**2
        nbox += 1; worst[3] = min(worst[3], float(g.a)); fails += g.a <= 0
print(f"boxes {nbox}, failures {fails}; smallest certified lower bounds: R1 (g/d^2 - k1) {worst[1]:.3g}, "
      f"R2 (g - k1 d^2) {worst[2]:.3g}, R3 {worst[3]:.3g}")
