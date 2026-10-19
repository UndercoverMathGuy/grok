# C_ccfactor c7: quadratic (second-moment, copositive) detection on the whole binding set of programme maximisers.
# For class-C networks and any copositive Q on R^Delta:  sum_ab Q_ab X_a X_b >= -(1/2) sum_l V_l <Q, Psi^(l)>,
# Psi^(l)_ab = Re(c_l(a) conj c_l(b)) = kappa_l(a)+kappa_l(b)-kappa_l(a-b).  Feature l is detected iff <Q,Psi^(l)> < 0.
# (i) numerical check of the lemma on random class-C networks and random entrywise-nonnegative Q;
# (ii) LP: does an entrywise-nonnegative Q exist that detects EVERY support feature of a programme maximiser
#      on its binding set?  Gordan dual: infeasible iff some lambda >= 0, lambda != 0, has sum_l lambda_l Psi^(l)_ab >= 0
#      for all a != b in Delta.
import sys
import numpy as np
from scipy.optimize import linprog
sys.path.insert(0, '/home/user/grok/heracles/conjecture20/unify/C_ccfactor')
sys.path.insert(0, '/home/user/grok/heracles/conjecture20/mechanism_scripts')
import ccf, cclib as C
import prog
E = C.E
rng = np.random.default_rng(3)


def Psi(p, l, offs):
    c = 1 - np.exp(-2j * np.pi * l * np.array(offs) / p)
    return np.real(np.outer(c, np.conj(c)))


print('=== (i) the copositive second-moment lemma on random class-C networks (Q entrywise >= 0, random Delta)')
worst = -np.inf; cnt = 0
for (p, n) in [(5, 2), (7, 2), (11, 2), (7, 3), (13, 2)]:
    F = (p - 1) // 2
    for t in range(20):
        y = rng.uniform(0.2, 1, F)
        net, lab = ccf.perturbed_if(p, n, y, rng, [0.1, 0.3, 0.6][t % 3], extra=int(rng.integers(0, 3)))
        fn = ccf.FeatNet(net, 1.5, lab)
        k = int(rng.integers(2, p)); offs = sorted(rng.choice(np.arange(1, p), size=k, replace=False))
        hD = fn.g[:, [o - 1 for o in offs]].min()
        if hD <= 0: continue
        X = fn.g.mean(0)[[o - 1 for o in offs]] - hD
        for u in range(5):
            Q = rng.exponential(size=(k, k)) * (rng.random((k, k)) < 0.6); Q = Q + Q.T
            lhs = X @ Q @ X
            rhs = -0.5 * sum(fn.V[l - 1] * (Q * Psi(p, l, offs)).sum() for l in range(1, F + 1))
            worst = max(worst, (rhs - lhs) / max(abs(lhs), 1e-300)); cnt += 1
print(f'  {cnt} cases: max (rhs - lhs)/|lhs| = {worst:.2e}  (<= 0 required)')

print('=== (ii) joint quadratic detectability of the support on the binding set (entrywise-nonnegative Q, LP)')
from sympy import primerange
res_rows = []
for p in list(primerange(5, 62)):
    for s in [1.05, 1.2, 1.35, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0]:
        G, y, cov = prog.solve(p, s, restarts=4)
        S = list(np.nonzero(y > 1e-10)[0] + 1); F = (p - 1) // 2
        Bd = list(np.nonzero(cov <= G * (1 + 1e-6))[0] + 1)
        offs = sorted(set(Bd + [p - d for d in Bd]))
        k = len(offs); pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
        if len(S) < 2 or not pairs:
            res_rows.append((p, s, len(S), k, None, None)); continue
        Ps = [Psi(p, l, offs) for l in S]
        # LP: variables q_ij >= 0 (i<j); constraints  sum_ij q_ij * 2 Psi_l[i,j] <= -1  for all l in S
        A = np.array([[2 * Pl[i, j] for (i, j) in pairs] for Pl in Ps])
        r = linprog(np.zeros(len(pairs)), A_ub=A, b_ub=-np.ones(len(S)), bounds=(0, None), method='highs')
        joint = r.status == 0
        indiv = [bool((A[i] < -1e-12).any()) for i in range(len(S))]
        res_rows.append((p, s, len(S), k, joint, all(indiv)))
tot = [r for r in res_rows if r[4] is not None]
print(f'  {len(tot)} (p,s) cases with K* >= 2 (p in [5,61]):')
print(f'    every support feature individually detectable on the binding set: {sum(r[5] for r in tot)}/{len(tot)}')
print(f'    all support features JOINTLY detectable by one entrywise-nonnegative Q:  {sum(r[4] for r in tot)}/{len(tot)}')
print('    jointly undetectable cases (p, s, K*, |Delta|):', [r[:4] for r in tot if not r[4]][:25])
