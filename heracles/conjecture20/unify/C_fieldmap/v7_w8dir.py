# v7: sanity checks for
#  (W8a) directional W8: aligned circular class-C single features, cone half-angle alpha = pi k/p <= alpha* = 36.59 deg,
#        any complex apex z0 = c - i eta:  Sum_T |beta_T|^2 <= k_alpha x^2 |A|^2  (x = 1 - c/|A|)  and  c <= nu^{1-s} Q^s.
#  (K*)  K* = min_{0<u<1} -log(1-u)/u^2 and alpha* = arccos(1/(1+K*))/2.
#  (OB)  the circular two-point construction in a sector of half-angle >= pi/4: (Re A - c)/sigma -> 0.
#  (N2)  n = 2: one coupled class-C feature + IF background with COMPLEX amplitudes satisfies
#        h <= min_d [ sum_{m != l} |A_m| kappa_m(d) + 3^{1-s} Q_l^s kappa_l(d) ]   (Theorem S + R9 / Theorem A).
#  (MP)  mean projection: if sum_l (|A_l|/pi)^r <= sum_l Q_l then h <= pi Gamma (sum Q)^s.
import numpy as np
from scipy.optimize import minimize_scalar, brentq
from fmlib import *
from cjlib import gamma_exact

out = []
def say(*a):
    s = ' '.join(str(x) for x in a); print(s); out.append(s)

# (K*)
f = lambda u: -np.log1p(-u) / u ** 2
res = minimize_scalar(f, bounds=(0.3, 0.95), method='bounded', options={'xatol': 1e-14})
Kstar = res.fun; ustar = res.x
alpha_star = 0.5 * np.arccos(1 / (1 + Kstar))
say(f'(K*) K* = {Kstar:.6f} at u* = {ustar:.6f};  alpha* = {np.degrees(alpha_star):.4f} deg;  k/p <= {alpha_star/np.pi:.5f}')
say(f'     W8 used sqrt5 = {np.sqrt(5):.6f} <= K*: {np.sqrt(5) <= Kstar}')

def best_apex(Phi, cotk):
    g = lambda eta: -np.min(np.real(Phi) - cotk * np.abs(np.imag(Phi) + eta))
    lo, hi = -np.abs(Phi).max() - 1, np.abs(Phi).max() + 1
    r = minimize_scalar(g, bounds=(lo, hi), method='bounded', options={'xatol': 1e-13})
    return -r.fun, r.x

rng = np.random.default_rng(17)
viol_i = 0; viol_ii = 0; ntest = 0; maxratio = 0
for (p, n) in [(5, 2), (7, 2), (11, 2), (13, 2), (5, 3), (7, 3), (11, 3), (5, 4), (17, 2), (19, 3)]:
    X = grid(p, n); nu = n + 1
    ks = [k for k in range(1, (p - 1) // 2 + 1) if np.pi * k / p <= alpha_star]
    for trial in range(120):
        psi = rng.uniform(0, TAU)
        m = int(rng.integers(nu, 2 * nu + 2))
        if trial % 4 != 3:
            sc = [0.01, 0.04, 0.15][trial % 4]
            C0, xi0 = if_atoms(n, 1.0)
            mod = np.concatenate([np.abs(C0), rng.uniform(0, sc, m - nu)]) * np.exp(rng.normal(0, sc, m))
            xi = np.concatenate([xi0, np.exp(1j * rng.uniform(0, TAU, (m - nu, n)))]) * np.exp(1j * rng.normal(0, 2 * sc, (m, n)))
        else:
            mod = rng.uniform(0, 1, m); xi = np.exp(1j * rng.uniform(0, TAU, (m, n)))
        C = mod * np.exp(1j * psi)
        Phi = field(p, X, 1, C, xi); A = Phi.mean(); a = abs(A)
        # sign-flip energy (class C: exact Parseval of the fluctuation)
        E = np.mean(np.abs(Phi - A) ** 2)
        for k in ks:
            cotk = 1 / np.tan(np.pi * k / p); alpha = np.pi * k / p; kal = 1 / np.cos(2 * alpha) - 1
            c, eta = best_apex(Phi, cotk)
            if c <= 0: continue
            x = 1 - c / a
            ntest += 1
            if E > kal * x ** 2 * a ** 2 * (1 + 1e-9) + 1e-12: viol_i += 1
            for b in (2.0, 2.5, nu - 0.25):
                if b < 2: continue
                s = nu / b; r = 1 / s
                Q = np.sum(np.abs(C) ** r)
                ratio = c / (nu ** (1 - s) * Q ** s)
                maxratio = max(maxratio, ratio)
                if ratio > 1 + 1e-9: viol_ii += 1
say(f'(W8a) {ntest} (feature, direction) tests with positive apex: violations of (i) {viol_i}, of c <= nu^(1-s) Q^s {viol_ii}; '
    f'max c/(nu^(1-s)Q^s) = {maxratio:.5f}')

# (OB) circular two-point law in the 45-degree sector
for eps in [0.5, 0.1, 1e-2, 1e-4]:
    M = 1.0; lam = np.sqrt(1 - eps)
    vals = np.array([0, M + 1j * lam * M, M - 1j * lam * M]); prob = np.array([1 - eps, eps / 2, eps / 2])
    Am = np.sum(prob * vals); pv = np.sum(prob * (vals - Am) ** 2); var = np.sum(prob * np.abs(vals - Am) ** 2)
    inside = np.all(np.abs(np.imag(vals)) <= 1.0 * np.real(vals) + 1e-15)
    say(f'(OB) eps={eps:g}: pseudo-variance={abs(pv):.1e}, in 90-degree wedge: {inside}, (Re A - c)/sigma = '
        f'{np.real(Am)/np.sqrt(var/2):.4f}  (= sqrt(eps/(1-eps)) = {np.sqrt(eps/(1-eps)):.4f})')

# (N2) n = 2, one coupled feature + complex IF background
viol = 0; tot = 0; worst = -np.inf
for p in [5, 7, 11, 13]:
    F = (p - 1) // 2; X = grid(p, 2)
    for trial in range(80):
        s = rng.uniform(1.0, 1.5); r = 1 / s
        l = int(rng.integers(1, F + 1))
        m = int(rng.integers(1, 6))
        C = rng.normal(size=m) + 1j * rng.normal(size=m); xi = np.exp(1j * rng.uniform(0, TAU, (m, 2)))
        if trial % 2:
            C0, xi0 = if_atoms(2, 2.0 * np.exp(1j * rng.uniform(0, TAU))); C = np.concatenate([C0, 0.5 * C]); xi = np.concatenate([xi0, xi])
        feats = {l: (C, xi)}; Am = {}
        for mm in range(1, F + 1):
            if mm == l: continue
            Am[mm] = (rng.normal() + 1j * rng.normal()) * rng.choice([0.3, 3, 30])
            feats[mm] = if_atoms(2, Am[mm])
        net = CNet(p, 2, feats, X=X)
        d = np.arange(1, p)
        rhs = (sum(abs(Am[mm]) * kap(p, mm, d) for mm in Am) + price_n2(s) * cost(C, r) ** s * kap(p, l, d)).min()
        tot += 1; worst = max(worst, net.h() - rhs)
        viol += net.h() > rhs + 1e-9
say(f'(N2) {tot} networks (n=2, one coupled feature, complex IF background): violations {viol}, max h - bound = {worst:.3e}')

# (MP) mean projection on random class-C networks
viol = 0; tot = 0
for (p, n) in [(5, 2), (7, 2), (7, 3), (11, 2)]:
    F = (p - 1) // 2
    for trial in range(40):
        s = rng.choice([1.2, 1.5]); r = 1 / s
        pi = price_n2(s) if n == 2 else (n + 1) ** (1 - s)          # nu^{1-s} <= pi: conservative for n >= 3
        feats = {}
        for l in range(1, F + 1):
            m = int(rng.integers(1, 2 * n + 3))
            feats[l] = (rng.normal(size=m) + 1j * rng.normal(size=m), np.exp(1j * rng.uniform(0, TAU, (m, n))))
        net = CNet(p, n, feats); A = net.A(); Q = net.Q(r)
        if sum((abs(A[l]) / pi) ** r for l in A) <= sum(Q.values()):
            G, _ = gamma_exact(p, s); tot += 1
            viol += net.h() > pi * G * sum(Q.values()) ** s + 1e-12
say(f'(MP) aggregate-non-cheap random networks: {tot}, violations of RD bound {viol}')
open('v7_w8dir.out', 'w').write('\n'.join(out) + '\n')
