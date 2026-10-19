"""Check of Lemma F (dual form of the fully free relaxation, all output frequencies j in F carry free modes).

Primal: h(a) = max over fields Phi_j (j in F) with prescribed own coefficients a_j(T) (at 2j1_T) of min_{x,delta} g(x,delta).
Dual (claimed): h(a) = min over mu (probability on Z_p^x) and complex c_{l,T} (l in F, T != {}) with
   K(x,delta) = mu_delta + 2 sum_{l,T} Re( c_{l,T} rho^{ l (delta - 2 s_T(x)) } ) >= 0  for all x, delta
of   sum_delta mu_delta sum_j Re( a_j(()) c_j(delta) )  -  p sum_{l,T} Re( a_l(T) c_{l,T} ).
Both are LPs; we solve both on random own data and compare.
"""
import itertools
import numpy as np
from scipy.optimize import linprog
from mrel import Model, subsets

rng = np.random.default_rng(5)
for (p, n) in [(5, 2), (7, 2), (5, 3)]:
    F = (p - 1) // 2
    M = Model(p, n, list(range(1, F + 1)), mode='free')
    X = M.X; N = len(X); rho = np.exp(2j * np.pi / p)
    Ts = [T for T in subsets(n) if T]
    for trial in range(3):
        a = {j: {T: (rng.normal() + 1j * rng.normal()) * (1.0 if T == () else 0.3) for T in subsets(n)} for j in range(1, F + 1)}
        hp, _ = M.margin(a)
        # dual LP variables: mu (p-1), then for each (l,T): Re c, Im c
        pairs = [(l, T) for l in range(1, F + 1) for T in Ts]
        nv = (p - 1) + 2 * len(pairs)
        cobj = np.zeros(nv)
        for di, d in enumerate(range(1, p)):
            cobj[di] = sum((a[j][()] * (1 - rho ** (-j * d))).real for j in range(1, F + 1))
        for k, (l, T) in enumerate(pairs):
            # - p Re(a c) = -p (Re a Re c - Im a Im c)
            cobj[p - 1 + 2 * k] = -p * a[l][T].real
            cobj[p - 1 + 2 * k + 1] = p * a[l][T].imag
        A = []; b = []
        for di, d in enumerate(range(1, p)):
            for x in X:
                row = np.zeros(nv); row[di] = -1.0
                for k, (l, T) in enumerate(pairs):
                    ph = rho ** (l * (d - 2 * sum(x[list(T)])))
                    # 2 Re(c ph) = 2(Re c Re ph - Im c Im ph)
                    row[p - 1 + 2 * k] = -2 * ph.real
                    row[p - 1 + 2 * k + 1] = 2 * ph.imag
                A.append(row); b.append(0.0)
        Aeq = np.zeros((1, nv)); Aeq[0, :p - 1] = 1.0
        res = linprog(cobj, A_ub=np.array(A), b_ub=np.array(b), A_eq=Aeq, b_eq=[1.0],
                      bounds=[(0, None)] * (p - 1) + [(None, None)] * (2 * len(pairs)), method='highs')
        print(f"p={p} n={n} trial {trial}: primal h = {hp:.10f}   dual = {res.fun:.10f}   diff {abs(hp-res.fun):.1e}")
