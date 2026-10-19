"""Referee check of L1's counterexample ingredients (independent code).

(a) p = 3, n = 11 exact certificate (heracles/conjecture20/exact_certificate.py weights): gamma/gamma_IF (exact) and the
    frequency split theta_split (P_0 = constants, P_1 = mean-zero part, per factor): gamma_split/gamma_IF (exact rational).
    Claim checked: mixed (DC) modes raise the margin above the split's (per-network failure of 'mixing never helps').
(b) SPLIT R10 violator (p,n,b) = (5,2,2), viol_p5_n2_b2.npz (weights rounded to 12 sig. digits as in the certificate):
    gamma/gamma_IF, gamma_split/gamma_IF, and h / RHS(D_Q) with
       RHS(D_Q) = nu^{1-s} (C_b P / sum_l Q_l)^s min_delta sum_l Q_l^s kappa_l(delta),  C_b = B_p^{r},
    Q_l = sum_i |beta_il|^r, beta_il = 2 p^{-nu} conj(what_i(l)) prod_t uhat_it(l).  Margins exact (Fractions), rest mpmath 40 digits.
"""
import importlib.util
import itertools
from fractions import Fraction as Fr
import numpy as np
import mpmath as mp
mp.mp.dps = 40
ROOT = "/home/user/grok/heracles/conjecture20/"

# ---------------- (a) p = 3 ----------------
spec = importlib.util.spec_from_file_location("ec", ROOT + "exact_certificate.py")
ec = importlib.util.module_from_spec(spec); spec.loader.exec_module(ec)


def margin_sym(U, W, n, p):
    """slot-symmetric network: margin over count vectors (exact)."""
    m = len(U); h = None
    for counts in itertools.product(range(n + 1), repeat=p - 1):
        if sum(counts) > n:
            continue
        N = list(counts) + [n - sum(counts)]          # N[a] = #slots equal to a, a = 0..p-1 (last one fills)
        N = [N[-1]] + N[:-1]                           # reorder so N[0] is count of a=0
        y = sum(a * N[a] for a in range(p)) % p
        z = []
        for i in range(m):
            v = Fr(1)
            for a in range(p):
                v *= U[i][a] ** N[a]
            z.append(v)
        f = [sum(W[i][c] * z[i] for i in range(m)) for c in range(p)]
        g = min(f[y] - f[c] for c in range(p) if c != y)
        h = g if h is None or g < h else h
    return h


for n in [11, 13]:
    p = 3; nu = n + 1
    net = ec.NETWORKS[n]
    U = [[Fr(a) for a in row] for row in net["u"]]
    W = [[Fr(a) for a in row] for row in net["w"]]
    P2 = sum(n * sum(a * a for a in U[i]) + sum(a * a for a in W[i]) for i in range(len(U)))
    h = margin_sym(U, W, n, p)
    gIF = Fr(3 * nu) / Fr(6 * nu * nu) ** (nu // 2)
    gamma = h / P2 ** (nu // 2)
    # split
    Us, Ws = [], []
    for i in range(len(U)):
        mu_u = sum(U[i]) / 3; mu_w = sum(W[i]) / 3
        Us += [[mu_u] * 3, [a - mu_u for a in U[i]]]
        Ws += [[mu_w] * 3, [a - mu_w for a in W[i]]]
    P2s = sum(n * sum(a * a for a in Us[i]) + sum(a * a for a in Ws[i]) for i in range(len(Us)))
    hs = margin_sym(Us, Ws, n, p)
    print(f"(a) p=3 n={n}: gamma/gamma_IF = {float(gamma / gIF):.6f} (exact > 1: {gamma > gIF}); "
          f"P2_split == P2: {P2s == P2}; gamma_split/gamma_IF = {float(hs / P2s ** (nu // 2) / gIF):.4f}")

# ---------------- (b) (5,2,2) violator ----------------
p, n, b = 5, 2, 2
nu = n + 1; s = mp.mpf(nu) / b; r = 1 / s
d = np.load(ROOT + "unify/split/viol_p5_n2_b2.npz")
U = d["U0"]; W = d["W0"]; m = len(W)
q = lambda x: Fr(float('%.12g' % x))
Uq = [[[q(U[i, t, a]) for a in range(p)] for t in range(n)] for i in range(m)]
Wq = [[q(W[i, a]) for a in range(p)] for i in range(m)]


def margin_gen(Uq, Wq):
    h = None
    for x in itertools.product(range(p), repeat=n):
        y = sum(x) % p
        z = [Uq[i][0][x[0]] * Uq[i][1][x[1]] for i in range(len(Wq))]
        f = [sum(Wq[i][c] * z[i] for i in range(len(Wq))) for c in range(p)]
        g = min(f[y] - f[c] for c in range(p) if c != y)
        h = g if h is None or g < h else h
    return h


def dft(v, k):
    return mp.fsum(mp.mpf(v[a].numerator) / v[a].denominator * mp.e ** (-2j * mp.pi * k * a / p) for a in range(p))


def analyse(Uq, Wq, tag):
    mm = len(Wq)
    h = margin_gen(Uq, Wq)
    norms2 = [sum(a * a for t in range(n) for a in Uq[i][t]) + sum(a * a for a in Wq[i]) for i in range(mm)]
    P = mp.fsum(mp.mpf(x.numerator) / x.denominator for x in norms2)          # b = 2: P = sum ||omega||^2
    Bp = 2 * (2 * p * nu) ** (-mp.mpf(nu) / 2)
    Cb = Bp ** r
    F = range(1, (p - 1) // 2 + 1)
    Q = {}
    for l in F:
        Q[l] = mp.fsum(abs(2 * mp.mpf(p) ** (-nu) * mp.conj(dft(Wq[i], l)) * dft(Uq[i][0], l) * dft(Uq[i][1], l)) ** r for i in range(mm))
    kappa = lambda l, dl: 1 - mp.cos(2 * mp.pi * l * dl / p)
    Gam = min(mp.fsum(Q[l] ** s * kappa(l, dl) for l in F) for dl in range(1, p))
    rhs = nu ** (1 - s) * (Cb * P / mp.fsum(Q.values())) ** s * Gam
    hf = mp.mpf(h.numerator) / h.denominator
    gamma = hf / P ** (mp.mpf(nu) / b)
    Gam_s = 2.5 * mp.mpf(2) ** (-s)          # Gamma_{3/2}(5), equal allocation (all offsets give kappa sum 2.5)
    gIF = nu ** (1 - s) * Bp * Gam_s
    print(f"(b) {tag}: h = {float(hf):.6e} (exact sign {'+' if h > 0 else '-'}), gamma/gamma_IF = {float(gamma / gIF):.6f}, "
          f"h/RHS(D_Q) = {float(hf / rhs):.6f}")


analyse(Uq, Wq, "theta (mixed)")
# split: P_0 and P_k projections per factor (k = 1, 2), rational parts via real DFT projection in exact arithmetic is
# irrational at p = 5, so use high-precision floats -> rationals with 30 digits for the split margin.
def proj(v, k):
    vf = [mp.mpf(a.numerator) / a.denominator for a in v]
    if k == 0:
        mu_ = mp.fsum(vf) / p
        return [mu_] * p
    c = 2 / mp.mpf(p) * mp.fsum(vf[a] * mp.cos(2 * mp.pi * k * a / p) for a in range(p))
    sn = 2 / mp.mpf(p) * mp.fsum(vf[a] * mp.sin(2 * mp.pi * k * a / p) for a in range(p))
    return [c * mp.cos(2 * mp.pi * k * a / p) + sn * mp.sin(2 * mp.pi * k * a / p) for a in range(p)]


toF = lambda x: Fr(mp.nstr(x, 35))
Us, Ws = [], []
for i in range(m):
    for k in [0, 1, 2]:
        Us.append([[toF(v) for v in proj(Uq[i][t], k)] for t in range(n)])
        Ws.append([toF(v) for v in proj(Wq[i], k)])
analyse(Us, Ws, "theta_split (35-digit rational rounding of projections)")
