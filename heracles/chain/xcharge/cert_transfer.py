# EXACT certificate: sum_sigma lam_sigma beta_sigma(a) >= c |a|_2 for all a >= 0 (flip-enhanced transfer, V "New lemma").
# LHS is piecewise linear; on each cell of the arrangement of kink planes it is linear, so LHS - c|a| is concave there and
# its minimum over the simplex {a >= 0, sum a = 1} is attained at a vertex of the arrangement. We enumerate all vertices
# in exact rational arithmetic and check LHS^2 >= c^2 |a|^2 (LHS >= 0 checked too).
import sys, itertools
from fractions import Fraction as Fr
def run(Ao, As, lam, c):
    asg = [('1', 'F', '2', As, Ao), ('1', '2', 'F', As, Ao), ('2', 'F', '1', As, Ao), ('2', '1', 'F', As, Ao),
           ('F', '1', '2', Ao, Ao), ('F', '2', '1', Ao, Ao)]
    idx = {'1': 0, '2': 1, 'F': 2}
    planes = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]                                    # simplex faces
    for (o, f, cc, Ai, Aout), l in zip(asg, lam):
        if l == 0: continue
        n1 = [0, 0, 0]; n1[idx[o]] += 1; n1[idx[cc]] -= Ai; planes.append(tuple(n1))          # a_o = A_in a_c
        n2 = [0, 0, 0]; n2[idx[f]] += 1; n2[idx[cc]] -= Aout; planes.append(tuple(n2))        # a_f = A_out a_c
        n3 = [0, 0, 0]; n3[idx[f]] += Ai; n3[idx[o]] -= Aout; planes.append(tuple(n3))        # A_in a_f = A_out a_o
    def solve(n, m):
        # intersection of n.a = 0, m.a = 0, (1,1,1).a = 1  (Cramer, exact)
        M = [list(n), list(m), [1, 1, 1]]
        det = lambda A: (A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) - A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0])
                         + A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]))
        d = det(M)
        if d == 0: return None
        sol = []
        for j in range(3):
            Mj = [row[:] for row in M]
            for i in range(3): Mj[i][j] = [0, 0, 1][i]
            sol.append(Fr(det(Mj)) / Fr(d))
        return sol
    verts = set()
    for n, m in itertools.combinations(planes, 2):
        s = solve(n, m)
        if s is not None and all(t >= 0 for t in s): verts.add(tuple(s))
    worst = None
    for a in verts:
        L = Fr(0)
        for (o, f, cc, Ai, Aout), l in zip(asg, lam):
            if l == 0: continue
            x, y, w = a[idx[f]], a[idx[o]], a[idx[cc]]
            L += l * max(Ai * x + Aout * abs(y - Ai * w), Aout * (y + Ai * w) - Ai * x)
        n2 = sum(t * t for t in a)
        ok = L >= 0 and L * L >= c * c * n2
        slack = float(L * L - c * c * n2)
        if worst is None or slack < worst[0]: worst = (slack, a, ok)
        if not ok: return False, len(verts), (float(L), [float(t) for t in a])
    return True, len(verts), (worst[0], [float(t) for t in worst[1]])
if __name__ == '__main__':
    lam = [Fr(4, 9), Fr(7, 18), Fr(0), Fr(1, 6), Fr(0), Fr(0)]
    for c in (Fr(2725, 10000), Fr(2900, 10000), Fr(2920, 10000)):
        print(f"A*=1/4, A_s=1/2, c={c}:", run(Fr(1, 4), Fr(1, 2), lam, c))
