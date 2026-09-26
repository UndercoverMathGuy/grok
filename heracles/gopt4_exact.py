"""gopt4_exact.py -- exact (rational/algebraic) certificate for the junk-relief
coefficient at nu = 3 (n = 2), the case the paper needs.

Claim (Theorem J of GLOBAL_OPT4).  With  r = q/(n+1)  and
    lambda_n(r) = r(1-r) / [ 1 - ((n-1)/(n+1))^2 (1-r) ],   beta = lambda/(n+1),
the second-order junk-relief programme at the (n+1)-block has value exactly
lambda_n(r); i.e. lambda_n(r) is the least lambda >= 0 with  P - lambda Q <= 0.

Certificate, all in exact arithmetic:
  (a)  lambda* Q - P  is positive semidefinite
       (checked by: every coefficient of det(t I + (lambda* Q - P)) is >= 0,
        which for a real symmetric matrix is equivalent to PSD);
  (b)  det(lambda* Q - P) = 0 and a kernel vector u has u^T Q u > 0,
       so no smaller lambda works.

Run:  uv run --with sympy python gopt4_exact.py
"""
import sys
import itertools
import sympy as sp


def build_exact(n, r):
    nb = n + 1
    Ts = [T for T in itertools.combinations(range(n), k)
          for k in ()] if False else \
        [T for k in range(1, n + 1) for T in itertools.combinations(range(n), k)]
    nd = nb * n
    dim = 2 * nb + nd
    w = sp.exp(2 * sp.pi * sp.I / nb)

    L = sp.zeros(len(Ts), dim)
    for t, T in enumerate(Ts):
        m = len(T)
        for l in range(nb):
            ph = sp.simplify(w ** (-l * m))
            L[t, l] += ph / nb
            L[t, nb + l] += sp.I * ph / nb
            for j in T:
                L[t, 2 * nb + l * n + j] += sp.I * ph / nb

    # kernel of  sum_l eps_l = 0  (real and imaginary parts separately)
    Zcols = []
    for l in range(1, nb):                 # x-part: e_l - e_0
        v = sp.zeros(dim, 1)
        v[l] = 1
        v[0] = -1
        Zcols.append(v)
    for l in range(1, nb):                 # y-part
        v = sp.zeros(dim, 1)
        v[nb + l] = 1
        v[nb] = -1
        Zcols.append(v)
    for k in range(nd):                    # delta unconstrained
        v = sp.zeros(dim, 1)
        v[2 * nb + k] = 1
        Zcols.append(v)
    Z = sp.Matrix.hstack(*Zcols)

    P = sp.zeros(dim, dim)
    for l in range(nb):
        P[l, l] = r * (1 - r) / 2
        P[nb + l, nb + l] = -r / 2

    LZ = sp.expand(L * Z)
    Q = sp.re(sp.expand(LZ.conjugate().T * LZ))
    Q = sp.simplify(Q)
    Pz = sp.simplify(Z.T * P * Z)
    return Pz, Q


def certify(n, q):
    r = sp.Rational(q).limit_denominator(1000) / sp.Integer(n + 1)
    kap = sp.Rational((n - 1) ** 2, (n + 1) ** 2)
    lam = sp.simplify(r * (1 - r) / (1 - kap * (1 - r)))
    P, Q = build_exact(n, r)
    M = sp.simplify(lam * Q - P)
    t = sp.symbols('t')
    cp = sp.Poly(sp.expand((t * sp.eye(M.rows) + M).det(method='berkowitz')), t)
    coeffs = [sp.simplify(c) for c in cp.all_coeffs()]
    psd = all(sp.simplify(c) >= 0 for c in coeffs)
    det0 = sp.simplify(coeffs[-1])          # det(M)
    ns = M.nullspace()
    uQu = None
    if ns:
        u = ns[0]
        uQu = sp.simplify((u.T * Q * u)[0, 0])
    print(f"n={n} q={q} r={r}  lambda* = {lam} = {sp.nsimplify(lam)} "
          f"= {float(lam):.12f}")
    print(f"   (a) all coefficients of det(tI + lambda*Q - P) >= 0 : {psd}")
    print(f"   (b) det(lambda* Q - P) = {det0}   kernel dim = {len(ns)}"
          f"   u^T Q u = {uQu}"
          f" ({'> 0' if uQu is not None and sp.simplify(uQu) > 0 else 'n/a'})")
    return bool(psd), det0 == 0


if __name__ == '__main__':
    cells = [(2, sp.Rational(2)), (2, sp.Rational(5, 2)), (2, sp.Rational(7, 3))]
    if len(sys.argv) > 1:
        cells = [(2, sp.Rational(sys.argv[1]))]
    for n, q in cells:
        certify(n, q)
