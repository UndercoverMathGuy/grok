"""Coarse scan of the min-T-refined phase-blind bound at (7,2) (float, exploration).
For each (tau, x1, x2) on a grid: max over b_j in {0, x_j/2, min(x_j,1)} and slot-set patterns of
min over detection strategies (max-T with amount x_j, or another T with amount b_j).
Usage: python bscan72.py s"""
import sys, itertools
import numpy as np
import cvxpy as cp
import rig72

s = float(sys.argv[1]); r = 1 / s; th = 3.0 ** (1 - s); c1 = 1 - th
models = {c: rig72.Model(c, 'pat') for c in rig72.CASES}
MIR = {'1': '2', '2': '1', '12': '12'}
TS = ['1', '2', '12']
PATS = rig72.CASES


def g(b):
    return (((1 + b) / 2) ** r + ((1 - b) / 2) ** r) ** s


def Fpay(x, b):
    return max(1 - c1 * x, th * max(1.0, x), th * g(min(b, 1.0)))


cache = {}
def solve(case, tau, d1, d2, F1, F2):
    case = case if case in models else (MIR[case[0]], MIR[case[1]])
    key = (case, round(tau, 6), round(d1, 6), round(d2, 6), round(F1, 6), round(F2, 6))
    if key in cache:
        return cache[key]
    m = models[case]
    A1, A2 = 1 - tau, tau
    F = ((F1 * A1) ** r + (F2 * A2) ** r) ** s
    m.cA.value = np.array([A1, A2]) / F; m.cM.value = np.array([d1 * A1, d2 * A2]) / F
    m.wK.value = 3.0 ** (s - 1)
    m.prob.solve(solver=cp.CLARABEL)
    v = m.prob.value / (3.5 * 3.0 ** (-s))
    cache[key] = v
    return v


def refined(tau, x1, x2):
    worst = 0; arg = None
    for b1 in sorted(set([0.0, x1 / 2, min(x1, 1.0)])):
        for b2 in sorted(set([0.0, x2 / 2, min(x2, 1.0)])):
            F1 = Fpay(x1, b1); F2 = Fpay(x2, b2)
            for pat in PATS:
                best = np.inf
                for o1 in ['max'] + [t for t in TS if t != pat[0]]:
                    for o2 in ['max'] + [t for t in TS if t != pat[1]]:
                        T1 = pat[0] if o1 == 'max' else o1; T2 = pat[1] if o2 == 'max' else o2
                        d1 = x1 if o1 == 'max' else b1; d2 = x2 if o2 == 'max' else b2
                        best = min(best, solve((T1, T2), tau, d1, d2, F1, F2))
                        if best < worst:
                            break
                    if best < worst:
                        break
                if best > worst:
                    worst = best; arg = (b1, b2, pat)
    return worst, arg


best = (0, None)
for tau in np.linspace(0.3, 0.8, 11):
    for x1 in (0.0, 0.4, 0.7, 0.85, 1.0, 1.15, 1.4):
        for x2 in (0.0, 0.4, 0.7, 0.85, 1.0, 1.15, 1.4):
            v, arg = refined(tau, x1, x2)
            if v > best[0]:
                best = (v, (round(tau, 3), x1, x2, arg))
print(f"s={s}: refined scan sup {best[0]:.5f} at {best[1]}", flush=True)
