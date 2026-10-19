"""What the min-T refinement would give at (7,2): for a configuration (tau, x1, x2) the adversary also chooses
b_j = min_T |beta_{j,T}|/A_j in [0, x_j] and the slot-set pattern of the maxima; payment uses
F_j >= max(1 - c1 x_j, theta max(1,x_j), theta g(b_j)), g(b) = (((1+b)/2)^r + ((1-b)/2)^r)^s  (two-atom cap with min_T);
the test may detect, for each feature, the max-T (amount x_j, slot-set fixed by the pattern) or any other T
(amount b_j, slot-set chosen by the test).  Prints max over (b, pattern) of min over strategies.
Usage: python brefine72.py s tau x1 x2 [nb]"""
import sys, itertools
import numpy as np
import cvxpy as cp
import rig72

s = float(sys.argv[1]); tau = float(sys.argv[2]); x1 = float(sys.argv[3]); x2 = float(sys.argv[4])
nb = int(sys.argv[5]) if len(sys.argv) > 5 else 6
r = 1 / s; th = 3.0 ** (1 - s); c1 = 1 - th
models = {c: rig72.Model(c, 'pat') for c in rig72.CASES}
MIR = {'1': '2', '2': '1', '12': '12'}


def rep(case):
    return case if case in models else (MIR[case[0]], MIR[case[1]])


def g(b):
    return (((1 + b) / 2) ** r + ((1 - b) / 2) ** r) ** s


def Fpay(x, b):
    v = max(1 - c1 * x, th * max(1.0, x), th * g(min(b, 1.0)))
    return v


def solve(case, d1, d2, F1, F2):
    m = models[rep(case)]
    A1, A2 = 1 - tau, tau
    F = ((F1 * A1) ** r + (F2 * A2) ** r) ** s
    m.cA.value = np.array([A1, A2]) / F
    m.cM.value = np.array([d1 * A1, d2 * A2]) / F
    m.wK.value = 3.0 ** (s - 1)
    m.prob.solve(solver=cp.CLARABEL)
    return m.prob.value / (3.5 * 3.0 ** (-s))


TS = ['1', '2', '12']
worst = (0, None)
for b1 in np.linspace(0, x1, nb):
    for b2 in np.linspace(0, x2, nb):
        F1 = Fpay(x1, b1); F2 = Fpay(x2, b2)
        for pat in itertools.product(TS, TS):
            best = np.inf
            for o1 in (['max'] + TS):
                for o2 in (['max'] + TS):
                    T1 = pat[0] if o1 == 'max' else o1
                    T2 = pat[1] if o2 == 'max' else o2
                    d1 = x1 if (o1 == 'max' or o1 == pat[0]) else b1
                    d2 = x2 if (o2 == 'max' or o2 == pat[1]) else b2
                    best = min(best, solve((T1, T2), d1, d2, F1, F2))
            if best > worst[0]:
                worst = (best, (round(b1, 3), round(b2, 3), pat))
print(f"s={s} tau={tau} x=({x1},{x2}): refined bound (max over b, pattern) = {worst[0]:.5f} at {worst[1]}", flush=True)
