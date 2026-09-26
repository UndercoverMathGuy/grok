"""NUMERICAL feasibility of the matched population inequality PI(Psi), Psi(j) = sum_sigma lam_sigma beta_sigma(j).
Population normalised so G = sum c_l = 1 (real); e = sum |c_l| >= 1; j = (|J1|, |J2|, |J12|), each <= e.
Matrix bound (fable0923 Lemma 3 / P3_ANALYTIC 1.1, keeping the three junk moduli separate, both chain orders):
  cost >= LHS(e, j) := max( e^{2/3}, 3^{1/3} e^{4/3} (3/B)^{1/3} ),
  B = 3e^2 + 2(j1^2 + j2^2 + j12^2) + 2u^2 + 4u min(j1, j2),  u = sqrt(2e(e-1)).
PI holds at (e, j) iff Psi(j) >= need(e, j) := 1 - (LHS/3^{1/3})^{3/2}.  For e >= sqrt3 it always holds.
LP: maximise t s.t. sum lam beta(j) - N(j) >= t on a grid, N(j) = max_e need(e, j).
"""
import sys, numpy as np
from scipy.optimize import linprog
from transfer2 import all_betas
def LHS(e, j1, j2, j12):
    u = np.sqrt(2 * e * (e - 1)); B = 3 * e ** 2 + 2 * (j1 ** 2 + j2 ** 2 + j12 ** 2) + 2 * u ** 2 + 4 * u * np.minimum(j1, j2)
    return np.maximum(e ** (2 / 3), 3 ** (1 / 3) * e ** (4 / 3) * (3 / B) ** (1 / 3))
def N(j, ne=200):
    e = np.linspace(max(1.0, max(j)), np.sqrt(3), ne)
    if len(e) == 0 or max(j) > np.sqrt(3): return -1.0
    need = 1 - (LHS(e, *j) / 3 ** (1 / 3)) ** 1.5
    return need.max()
def run(As, Ao, n=22):
    grid = np.linspace(0, np.sqrt(3), n); pts, needs = [], []
    for j1 in grid:
        for j2 in grid:
            for j12 in grid:
                j = (j1, j2, j12); v = N(j)
                if v > 1e-12: pts.append(j); needs.append(v)
    B = np.array([all_betas(np.array(j), As, Ao) for j in pts]); k = B.shape[1]; needs = np.array(needs)
    r = linprog(np.r_[np.zeros(k), -1], A_ub=np.c_[-B, np.ones(len(B))], b_ub=-needs,
                A_eq=[np.r_[np.ones(k), 0]], b_eq=[1], bounds=[(0, None)] * k + [(None, None)])
    lam = r.x[:k]; slack = B @ lam - needs; i = int(np.argmin(slack))
    return -r.fun, np.round(lam, 3), pts[i]
if __name__ == '__main__':
    for name, As, Ao in [('eta sanity 0.3476', 0.3476, 0.3476), ('p7 dense', 0.425933, 0.160988), ('p11 {1234}', 0.364375, 0.183795),
                         ('p13 {1235}', 0.395934, 0.228572), ('p13 {1234}', 0.368428, 0.202498), ('p11 dense', 0.298256, 0.091082),
                         ('p13 {12345}', 0.314209, 0.138204), ('p13 dense', 0.260208, 0.075851)]:
        t, lam, worst = run(As, Ao)
        print(f"{name:16s}: min slack {t:+.4f} (>= 0 feasible)  lam={lam}  worst j={np.round(worst, 3)}", flush=True)
