# l2-polyball depth of the flip-enlarged body from (A*, A_s) via the flip-enhanced transfer (VERIFY.md "New lemma";
# same computation as ../transfer_flip.py: sphere grid n=60 + LP over assignment weights = NUMERICAL constant),
# and the rigorous fable version A(2+A)/3 at A = min(A*, 0.47).
import numpy as np
from scipy.optimize import linprog
def betas(a, As, Ao):
    m = dict(zip('12F', a))
    asg = [('1', 'F', '2', As, Ao), ('1', '2', 'F', As, Ao), ('2', 'F', '1', As, Ao), ('2', '1', 'F', As, Ao),
           ('F', '1', '2', Ao, Ao), ('F', '2', '1', Ao, Ao)]
    return np.array([max(Ai * m[f] + Aout * abs(m[o] - Ai * m[c]), Aout * (m[o] + Ai * m[c]) - Ai * m[f]) for o, f, c, Ai, Aout in asg])
def flip_transfer(As, Ao, n=60):
    pts = [v / np.linalg.norm(v) for i in range(n + 1) for j in range(n + 1 - i) for v in [np.array([i, j, n - i - j], float)] if v.any()]
    B = np.array([betas(v, As, Ao) for v in pts]); k = B.shape[1]
    r = linprog(np.r_[np.zeros(k), -1], A_ub=np.c_[-B, np.ones(len(B))], b_ub=np.zeros(len(B)),
                A_eq=[np.r_[np.ones(k), 0]], b_eq=[1], bounds=[(0, None)] * k + [(None, None)])
    return -r.fun
if __name__ == '__main__':
    print("A*    A_s   | fable A(2+A)/3 | flip transfer")
    for Ao, As in [(0.3476, 0.3476), (0.30, 0.35), (0.31, 0.351), (0.30, 0.40), (0.25, 0.45), (0.25, 0.5), (0.20, 0.5), (0.216, 0.274), (0.312, 0.351)]:
        f = min(Ao, 0.47); print(f"{Ao:.3f} {As:.3f} | {f * (2 + f) / 3:.4f}         | {flip_transfer(As, Ao):.4f}")
