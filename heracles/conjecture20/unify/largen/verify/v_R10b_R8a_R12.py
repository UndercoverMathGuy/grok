# Independent referee checks of largen R10b (width-6 positive-margin single feature at p=5, n=6), R8a (polydisc payment
# family) and R12 (rate identity).
import json, math, itertools
from fractions import Fraction as Fr
import numpy as np
from mpmath import iv, mp
import sympy as sp
iv.dps = 60; mp.dps = 60
ROOT = '/home/user/grok/heracles/conjecture20/'

# ---------------- R10b -----------------
print("=== R10b: rigorous margin of the Note's saved 1-D network p=5, n=6, width 6 ===")
J = json.load(open(ROOT + 'mechanism_scripts/kkt/oned_p5_n6.json'))
p, n = J['p'], J['n']; m = len(J['lam'])
eps = [1 if a < p / 2 else -1 for a in range(p)]; e = [2 * a if a < p / 2 else 2 * a - p for a in range(p)]
lvmax = max(J['logV'])
# rational network: u_i(a) = eps_a exp(lam_i e_a) rounded to 15 significant digits (exact rationals);
# w_i(c) = Re(V_i) cos(2 pi c/p) - Im(V_i) sin(2 pi c/p) with V_i rational complex -> output EXACTLY in span{cos, sin}
U = [[Fr(f"{eps[a] * math.exp(J['lam'][i] * e[a]):.15e}") for a in range(p)] for i in range(m)]
V = [(Fr(f"{math.exp(J['logV'][i] - lvmax) * math.cos(J['argV'][i]):.15e}"),
      Fr(f"{math.exp(J['logV'][i] - lvmax) * math.sin(J['argV'][i]):.15e}")) for i in range(m)]
def ivq(q): return iv.mpf(q.numerator) / iv.mpf(q.denominator)
cosc = [iv.cos(2 * iv.pi * c / p) for c in range(p)]; sinc = [iv.sin(2 * iv.pi * c / p) for c in range(p)]
W = [[ivq(V[i][0]) * cosc[c] - ivq(V[i][1]) * sinc[c] for c in range(p)] for i in range(m)]
def counts(n, p):
    def rec(prefix, left, k):
        if k == p - 1:
            yield prefix + (left,); return
        for v in range(left + 1):
            yield from rec(prefix + (v,), left - v, k + 1)
    return list(rec((), n, 0))
def margin(sign):
    # sign=+1: w = Re(V rho^c); sign=-1: conjugate convention w = Re(conj(V) rho^c)
    Wl = [[ivq(V[i][0]) * cosc[c] - sign * ivq(V[i][1]) * sinc[c] for c in range(p)] for i in range(m)]
    hmin = None; nlow = 0
    for N in counts(n, p):
        y = sum(a * N[a] for a in range(p)) % p
        z = [ivq(math.prod([U[i][a] ** N[a] for a in range(p)])) for i in range(m)]     # exact rational product
        f = [sum(Wl[i][c] * z[i] for i in range(m)) for c in range(p)]
        for c in range(p):
            if c != y:
                g = f[y] - f[c]
                if hmin is None or g.a < hmin.a: hmin = g
    return hmin
for sgn in [+1, -1]:
    h = margin(sgn)
    print(f"  convention {'Re(V rho^c)' if sgn > 0 else 'Re(conj V rho^c)'}: margin h in [{mp.nstr(h.a, 12)}, {mp.nstr(h.b, 12)}]  -> positive: {h.a > 0}")
print(f"  width m = {m}, nu = n+1 = {n + 1}; inputs: same vector in every slot (slot-symmetric), sign pattern {eps}")
# normalised ratio to the IF single-feature value at b=2, for information (gamma/gamma_IF with Gamma = kappa_min when K*=1)
# outputs are exactly in span{cos,sin}: single-feature in the Companion's sense.

# ---------------- R8a -----------------
print("=== R8a: family beta_i = A/n, xi_it = e^{2 pi i i/n}: P(z) = A(1 + z_1...z_n) ===")
for nn in [2, 3, 4, 5, 6]:
    zs = sp.symbols(f'z1:{nn + 1}'); A = sp.Symbol('A')
    w = sp.exp(2 * sp.pi * sp.I / nn)
    P = sum((A / nn) * sp.prod([1 + w ** i * z for z in zs]) for i in range(nn))
    diff = sp.simplify(sp.expand(sp.expand_complex(sp.expand(P - A * (1 + sp.prod(zs))))))
    print(f"  n={nn}: P - A(1+z1...zn) simplifies to {diff}")
for nn in [8, 12, 20]:
    # numeric check of the moment structure: m_k = sum_i xi_i^k / n = [n | k]
    xi = np.exp(2j * np.pi * np.arange(nn) / nn)
    mk = [abs(np.mean(xi ** k)) for k in range(1, nn + 1)]
    print(f"  n={nn}: max_{{1<=k<n}} |m_k| = {max(mk[:-1]):.1e}, m_n = {mk[-1]:.3f}")
print("  required constant C >= (1-(n/nu)^{s-1}) R^{-n} at R=1/2:")
for nn in [2, 3, 4, 8, 16, 32, 64]:
    nu = nn + 1
    for s in [nu / 2, 1.5, 1.05]:
        if s > nu / 2 + 1e-12 or s <= 1: continue
        g = 1 - (nn / nu) ** (s - 1)
        print(f"    n={nn} s={s:.3f}: gain/A={g:.4f}  C >= {g * 2 ** nn:.4g}  (0.28*2^n = {0.28 * 2 ** nn:.4g})")
# cone status of the family (is it a positive-margin network?): field values A(1+omega), omega in mu_p
for pp in [5, 7]:
    om = np.exp(2j * np.pi * np.arange(pp) / pp); P = 1 + om
    print(f"  p={pp}: min cm of A(1+z1..zn) over mu_p^n = {(P.real - abs(P.imag) / math.tan(math.pi / pp)).min():.4f} * A (negative -> outside the cone)")

# ---------------- R12 -----------------
print("=== R12: nu^{1-s} Q^s == (K_r/nu)^{s-1} L, K_r = Q^{1/(1-r)}/L^{r/(1-r)} ===")
rng = np.random.default_rng(3)
worst = 0
for t in range(2000):
    mm = int(rng.integers(1, 30)); nu = int(rng.integers(3, 40)); s = mp.mpf(rng.uniform(1.05, nu / 2)); r = 1 / s
    M = [mp.e ** (3 * mp.mpf(rng.standard_normal())) for _ in range(mm)]
    Q = mp.fsum([x ** r for x in M]); L = mp.fsum(M); K = Q ** (1 / (1 - r)) / L ** (r / (1 - r))
    lhs = nu ** (1 - s) * Q ** s; rhs = (K / nu) ** (s - 1) * L
    worst = max(worst, abs(lhs / rhs - 1))
    assert 1 - mp.mpf(10) ** -40 <= K <= mm * (1 + mp.mpf(10) ** -40)
print(f"  2000 random (masses, nu, s): max relative deviation {float(worst):.2e}; 1 <= K_r <= width always held")
sym_s = sp.Symbol('s', positive=True); Qs, Ls = sp.symbols('Q L', positive=True); r = 1 / sym_s
K = Qs ** (1 / (1 - r)) / Ls ** (r / (1 - r))
print("  symbolic: log((K/nu)^(s-1) L) - log(nu^(1-s) Q^s) =",
      sp.simplify(sp.expand_log(sp.log((K / sp.Symbol('nu', positive=True)) ** (sym_s - 1) * Ls) - sp.log(sp.Symbol('nu', positive=True) ** (1 - sym_s) * Qs ** sym_s), force=True)))
