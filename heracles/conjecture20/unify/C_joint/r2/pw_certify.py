# Interval-arithmetic certificates that PW_C fails (positive-margin class-C networks).
import numpy as np, mpmath as mp
from mpmath import iv
iv.dps = 30
def legendre_cert(p, n, s):
    # PW_C fails iff min(S_+, S_-) > (n/nu)^{1-r} (p-1), S_pm = sum_{m=1}^{p-1} (1 pm chi(m) sin(2 pi m/p))^r ; margin = sqrt(p)-2 > 0 scale-free
    r = iv.mpf(1)/iv.mpf(s); nu = n+1
    qr = set((m*m) % p for m in range(1, p))
    Sp = iv.mpf(0); Sm = iv.mpf(0)
    for m in range(1, p):
        c = 1 if m in qr else -1; sn = iv.sin(2*iv.pi*m/p)
        Sp += (1 + c*sn)**r; Sm += (1 - c*sn)**r
    rhs = (iv.mpf(n)/nu)**(1-r)*(p-1)
    lo = Sp.a if Sp.a < Sm.a else Sm.a
    return lo/rhs.b, (lo > rhs.b)
for (p, n, s) in [(347, 2, 1.5), (383, 2, 1.5), (467, 3, 2.0), (443, 3, 2.0)]:
    ratio, ok = legendre_cert(p, n, s)
    print(f"Legendre single-T network p={p} (=3 mod 4), n={n}, s={s}: certified min_x sum psi / (nu^(r-1) sum Q) >= {iv.nstr(ratio, 8)}; PW_C fails: {ok}; margin = sqrt(p)-2 > 0")
# optimised positive-margin network at p=43 (n=2, s=1.5): phases/amplitudes rounded to rationals, then interval evaluation
x = np.load('pwpos_p43_n2_s1.5.npy'); p = 43; n = 2; F = 21; s = iv.mpf(3)/2; r = 1/s; nu = 3
al = [mp.mpf(str(round(float(v), 15))) for v in np.exp(x[:F])]; be = [mp.mpf(str(round(float(v), 15))) for v in np.exp(x[F:2*F])]
ph = [mp.mpf(str(round(float(v), 15))) for v in x[2*F:]]
al = [a if a > 1e-9*max(be) else mp.mpf(0) for a in al]
cost = sum((iv.mpf(a))**r for a in al if a > 0) + (iv.mpf(n)/nu)**(1-r)*sum(iv.mpf(b)**r for b in be)
lhs_min = None
for w in range(p):
    S = iv.mpf(0)
    for l in range(1, F+1):
        S += (iv.mpf(al[l-1]) + iv.mpf(be[l-1])*(1 + iv.cos(iv.mpf(ph[l-1]) + 2*iv.pi*l*w/p)))**r
    lhs_min = S if lhs_min is None else (S if S.b < lhs_min.b else lhs_min)
# margin h = min_{w,d} sum (al+be) kappa_l(d) + V(w) - V(w-d)
V = [sum(iv.mpf(be[l-1])*iv.cos(iv.mpf(ph[l-1]) + 2*iv.pi*l*w/p) for l in range(1, F+1)) for w in range(p)]
hmin = None
for d in range(1, p):
    cl = sum((iv.mpf(al[l-1]) + iv.mpf(be[l-1]))*(1 - iv.cos(2*iv.pi*l*d/p)) for l in range(1, F+1))
    for w in range(p):
        g = cl + V[w] - V[(w-d) % p]
        hmin = g if hmin is None or g.a < hmin.a else hmin
print(f"p=43 optimised single-T network: certified PW ratio >= {iv.nstr(lhs_min.a/cost.b, 8)} (>1 refutes PW_C); certified margin h >= {iv.nstr(hmin.a, 8)} > 0")
