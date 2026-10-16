"""Barrier for proofs of Conjecture 20 that use only own-frequency Fourier modes (p = 3, b = 2).

Both earlier proof attempts (the p=3,n=2 four-moment proof and the matrix/gap-cone interval proof) bound the
margin using only (i) the clean and sign-flipped modes of the score table and (ii) the single-neuron Fourier
budget.  Every other mode (anything with a constant component in some slot) is left unconstrained.

In that relaxation the margin of a moment tensor M in (C^2)^{(x)n} is
    h*(M) = max{ h : M - h e_0  in  cone_K{ chi(zeta) : zeta in mu_3^n } },   chi(zeta)_T = prod_{t in T} zeta_t,
K = {c : |arg c| <= pi/3}  (the gap cone: g(x,1), g(x,2) >= h), and a network with atoms (beta_i, xi_i) has
M = sum_i beta_i chi(xi_i) with Fourier budget Q = sum_i |beta_i|^r, r = 2/nu.  Conjecture 20 at p = 3 would
need  h <= nu^{1-s} Q^s  (s = nu/2, units in which the clean average gives h <= Re M_0).

This script builds three slot-symmetric atoms  xi_0 = (1,...,1), xi_pm = (e^{+-i a},...)  and slack
c_1 = kappa e^{i PH} at zeta = w(1,..,1), c_2 = kappa e^{-i PH} at zeta = w^2(1,..,1)  so that
    sum_i beta_i chi(xi_i) = e_0 + c_1 chi(w 1) + c_2 chi(w^2 1)        (exactly, all 2^n coordinates),
hence h*(M) >= 1, and reports h-ratio = 1 / (nu^{1-s} Q^s) > 1.  So no argument confined to own-frequency modes
can prove Conjecture 20 at p = 3, n = 3 or n = 4, b = 2.
"""
import itertools
from mpmath import mp, mpf, mpc, exp, pi, cos, sin, matrix, lu_solve, fabs, diff, findroot

mp.dps = 50
w = exp(2j * pi / 3)


def lin(a, PH):
    # unknowns b0 (real), b+ = x + i y, kappa;   eq_j:  b0 + 2 Re(b+ e^{i j a}) - 2 kappa cos(2 pi j/3 + PH) = [j = 0]
    A = matrix(4, 4); rhs = matrix(4, 1)
    for j in range(4):
        A[j, 0] = 1; A[j, 1] = 2 * cos(j * a); A[j, 2] = -2 * sin(j * a); A[j, 3] = -2 * cos(2 * pi * j / 3 + PH)
        rhs[j] = 1 if j == 0 else 0
    v = lu_solve(A, rhs)
    return v[0], mpc(v[1], v[2]), v[3]


def report(n, a, PH):
    nu = n + 1; r = mpf(2) / nu; s = 1 / r
    b0, bp, k = lin(a, PH)
    atoms = [(b0, mpc(1)), (bp, exp(1j * a)), (bp.conjugate(), exp(-1j * a))]
    slack = {1: k * exp(1j * PH), 2: k * exp(-1j * PH)}
    err = max(abs(sum(b * z ** sum(T) for b, z in atoms)
                  - ((1 if sum(T) == 0 else 0) + sum(c * (w ** g) ** sum(T) for g, c in slack.items())))
              for T in itertools.product([0, 1], repeat=n))
    Q = sum(abs(b) ** r for b, _ in atoms)
    ratio = 1 / (nu ** (1 - s) * Q ** s)
    assert k >= 0 and err < mpf(10) ** -40
    print("n=%d: a = %s turns, kappa = %s (>= 0), |slack args| = 60 deg" % (n, mp.nstr(a / (2 * pi), 12), mp.nstr(k, 10)))
    print("      beta_0 = %s, beta_+ = %s" % (mp.nstr(b0, 10), mp.nstr(bp, 10)))
    print("      moment identity error over all 2^n subsets = %s" % mp.nstr(err, 3))
    print("      h*(M) >= 1,  Q = %s,  h / (nu^{1-s} Q^s) = %s" % (mp.nstr(Q, 12), mp.nstr(ratio, 12)))


# n = 3: one-parameter family in a; minimise Q
f3 = lambda a: sum(abs(b) ** (mpf(1) / 2) for b in (lin(a, -pi / 3)[0], lin(a, -pi / 3)[1], lin(a, -pi / 3)[1]))
a3 = findroot(lambda t: diff(f3, t), mpf('1.46'))
report(3, a3, -pi / 3)


# n = 4: the fifth moment equation fixes a; bracket and bisect
def resid4(a, PH=pi / 3):
    b0, bp, k = lin(a, PH)
    return b0 + 2 * (bp * exp(4j * a)).real - 2 * k * cos(8 * pi / 3 + PH)


lo, hi = 2 * pi * mpf('0.5860'), 2 * pi * mpf('0.5870')
flo = resid4(lo)
for _ in range(200):
    mid = (lo + hi) / 2; fm = resid4(mid)
    if (fm > 0) == (flo > 0):
        lo, flo = mid, fm
    else:
        hi = mid
report(4, (lo + hi) / 2, pi / 3)
