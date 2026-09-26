"""Extended separable lower bounds for the flip-enlarged body (per frequency, direction moduli m = (m1, m2, m3), m3 = {1,2}).
Orders:
 (T) transfer eliminations (6), as in transfer_cert.py                      [needs A_s, A*]
 (P) pure-pattern aims: A_s * m_T                                          [3]
 (V) averaged-outer eliminations: max(h_P, h_F2) >= max_a [f1(a) + max_c |g_a(c)|] >= mean_a [...] with a uniform:
     mean f1 = 0 and mean_a |z3 + z2 zeta^{-a}| >= Hol(|z3|, |z2|) where
     Hol(x, w) = max( max(x, w), (x^2+w^2)^{3/2} / sqrt((x^2+w^2)^2 + 2 x^2 w^2) )  (Hoelder: E|X| >= (E|X|^2)^{3/2}/(E|X|^4)^{1/2},
     E|X|^2 = x^2+w^2, E|X|^4 = (x^2+w^2)^2 + 2x^2w^2 for zeta^{-a} uniform on p-th roots, p >= 5).
     outer slot 1 (inner patterns 12 & 2, two-sided via F2): A_s * Hol(m3, m2); outer slot 2: A_s * Hol(m3, m1);
     outer a+b (inner patterns 1 & 2, one-sided): A* * Hol(m1, m2).                                  [3]
c = max_lam min_m sum lam_sigma beta_sigma(m) / |m|  (sphere grid; NUMERICAL here, certify later).
"""
import sys, numpy as np
from scipy.optimize import linprog
from transfer_cert import ORDERS, beta
def hol(x, w):
    s = x * x + w * w
    return max(max(x, w), s ** 1.5 / np.sqrt(s * s + 2 * x * x * w * w)) if s > 0 else 0.0
def all_betas(m, As, Ao):
    b = [float(beta(m, o, As, Ao)) for o in ORDERS]                     # 6 transfer + 3 pure
    b += [As * hol(m[2], m[1]), As * hol(m[2], m[0]), Ao * hol(m[0], m[1])]
    return np.array(b)
def c_value(As, Ao, n=80):
    pts = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            v = np.array([i, j, n - i - j], float)
            if v.any(): pts.append(v / np.linalg.norm(v))
    B = np.array([all_betas(v, As, Ao) for v in pts]); k = B.shape[1]
    r = linprog(np.r_[np.zeros(k), -1], A_ub=np.c_[-B, np.ones(len(B))], b_ub=np.zeros(len(B)),
                A_eq=[np.r_[np.ones(k), 0]], b_eq=[1], bounds=[(0, None)] * k + [(None, None)])
    return -r.fun, np.round(r.x[:k], 3)
if __name__ == '__main__':
    for name, As, Ao in [('p7 dense', 0.425933, 0.160988), ('p11 {1234}', 0.364375, 0.183795), ('p13 {1234}', 0.368428, 0.202498),
                         ('p13 {1235}', 0.395934, 0.228572), ('p11 dense', 0.313, 0.095), ('p13 dense', 0.26, 0.076)]:
        c, lam = c_value(As, Ao)
        print(f"{name:12s} A_s={As:.3f} A*={Ao:.3f}: c ~ {c:.4f}  lam={lam}")
