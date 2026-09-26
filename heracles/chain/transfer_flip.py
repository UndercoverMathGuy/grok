# Transfer lemma with flips.  For direction z = (z1, z2, z3) (patterns {1}, {2}, {1,2}; z3 is the full / a+b pattern),
# eliminating the variable of one pattern ("outer") and maximising the other two inside:
#   bound_sigma = A_in * Re sum conj(d) z_full + A_out * sum | z_outer + A_in d conj(z_coupled) |
# per frequency: beta >= max( A_in x + A_out |y - A_in w| ,  A_out (y + A_in w) - A_in x ),   (x, y, w) = |full|, |outer|, |coupled|
# Outer = slot 1 or slot 2 (4 assignments): inner can be two-sided via the flip (A_in = A_s), outer one-sided (A_out = A*).
# Outer = the a+b pattern (2 assignments): no clean flip, A_in = A_out = A*.
# Depth >= c = max over assignment weights lam of  min over a >= 0, |a| = 1 of sum lam_sigma beta_sigma(a).
import itertools, sys, numpy as np
from scipy.optimize import linprog
def betas(a, As, Ao):
    # a = (m1, m2, m3) moduli of patterns 1, 2, 12. assignments: (outer, full, coupled, A_in, A_out)
    m = dict(zip('12F', a))
    asg = [('1', 'F', '2', As, Ao), ('1', '2', 'F', As, Ao), ('2', 'F', '1', As, Ao), ('2', '1', 'F', As, Ao),
           ('F', '1', '2', Ao, Ao), ('F', '2', '1', Ao, Ao)]
    out = []
    for o, f, c, Ai, Aout in asg:
        x, y, w = m[f], m[o], m[c]
        out.append(max(Ai * x + Aout * abs(y - Ai * w), Aout * (y + Ai * w) - Ai * x))
    return np.array(out)
def depth(As, Ao, n=60):
    # sample the positive octant of the unit sphere finely, then solve the max-min LP over lam
    pts = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            v = np.array([i, j, n - i - j], float) / n
            if v.any(): pts.append(v / np.linalg.norm(v))
    B = np.array([betas(v, As, Ao) for v in pts])                    # rows: points, cols: assignments
    # maximise t s.t. B lam >= t, sum lam = 1, lam >= 0
    k = B.shape[1]
    r = linprog(np.r_[np.zeros(k), -1], A_ub=np.c_[-B, np.ones(len(B))], b_ub=np.zeros(len(B)),
                A_eq=[np.r_[np.ones(k), 0]], b_eq=[1], bounds=[(0, None)] * k + [(None, None)])
    return -r.fun, r.x[:k]
for name, Ao, As in [('fable check A*=A_s=0.40', 0.40, 0.40), ('fable check 0.3476', 0.3476, 0.3476),
                     ('dense p=11', 0.095, 0.313), ('dense p=13', 0.076, 0.260), ('p=59 old', 0.336, 0.352),
                     ('p=59 LADDER', 0.401, 0.429), ('p=113 old', 0.413, 0.502)]:
    c, lam = depth(As, Ao)
    print(f"{name:26s} A*={Ao:.3f} A_s={As:.3f}  ->  depth >= {c:.4f}  (fable formula A(2+A)/3 at A*: {Ao * (2 + Ao) / 3:.4f})  lam={np.round(lam, 2)}")
