"""tri_ineq.py -- the 3-variable inequality behind the transfer lemma (JOURNAL 2.2):

   L(a) := 2(a1+a2+a3) + sum_{i != j} |a_i - A a_j|  >=  2(2+A) ||a||_2 ,   a >= 0, 0 <= A <= A_max.

Part 1: numerical minimum of L/||a||_2 over the simplex, for a range of A (sanity).
Part 2: PROOF by vertex enumeration. On each cell of the arrangement {a_i = A a_j}, {a_i = 0} (restricted
to the simplex sum a = 1), L is linear and ||a||_2 is convex, so L - 2(2+A)||a||_2 is concave and attains its
minimum over the cell at a vertex of the cell. Every cell vertex is the solution of sum a = 1 plus two
independent equations from the list {a_i - A a_j = 0 (6), a_i = 0 (3)}. We enumerate all such points, keep the
nonneg ones, and check L - 2(2+A)||a||_2 >= 0 at each, for A on a fine grid with exact rational arithmetic
(A rational) and an interval-in-A argument: at each candidate point, a is a rational function of A, L is
piecewise-linear in a with the sign pattern determined at that vertex, and we verify the inequality
symbolically over A in [0, A_max] with sympy.
"""
import itertools, sys, numpy as np
import sympy as sp

def L_num(a, A):
    s = 2*a.sum()
    for i in range(3):
        for j in range(3):
            if i != j: s += abs(a[i] - A*a[j])
    return s

def part1():
    rng = np.random.default_rng(0)
    for A in (0.0, 0.2, 0.3, 0.4, 0.45, 0.48, 0.5, 0.55, 0.6):
        pts = np.abs(rng.standard_normal((400000, 3)))
        pts = np.vstack([pts, np.eye(3), np.ones((1,3)), np.array([[1,1,0],[1,0,1],[0,1,1],[1,A,0],[1,A,A],[1,1,A]])])
        r = np.array([L_num(x, A) / np.linalg.norm(x) for x in pts]) / (2*(2+A))
        i = r.argmin(); print(f"A={A:.2f}: min ratio {r.min():.6f} at a={np.round(pts[i]/pts[i].sum(),3)}")

def part2(Amax):
    A = sp.symbols('A', nonnegative=True)
    a = sp.symbols('a1 a2 a3')
    forms = [a[i] - A*a[j] for i in range(3) for j in range(3) if i != j] + list(a)
    cands = []
    for f1, f2 in itertools.combinations(forms, 2):
        sol = sp.solve([f1, f2, sum(a) - 1], a, dict=True)
        if not sol: continue
        s = sol[0]
        if len(s) < 3: continue
        v = [sp.simplify(s[x]) for x in a]
        cands.append(v)
    print(f"{len(cands)} candidate vertices (symbolic in A)")
    Ai = sp.Interval(0, Amax)
    bad = 0; checked = 0
    for v in cands:
        # nonnegativity over A in [0, Amax]: if a coordinate is negative somewhere, the vertex is outside for those A;
        # we only need to check A where all coordinates are >= 0. Split at sign changes numerically fine grid + exact check.
        grid = [sp.Rational(k, 400) * Amax for k in range(401)]
        for Aq in grid:
            vv = [sp.nsimplify(x.subs(A, Aq)) for x in v]
            if any(x < 0 for x in vv): continue
            Lv = 2*sum(vv) + sum(abs(vv[i] - Aq*vv[j]) for i in range(3) for j in range(3) if i != j)
            rhs2 = (2*(2+Aq))**2 * sum(x**2 for x in vv)
            checked += 1
            if Lv**2 < rhs2:                       # both sides >= 0, compare squares exactly
                bad += 1; print("VIOLATION at A=", Aq, "a=", vv, float(Lv), float(sp.sqrt(rhs2)))
    print(f"checked {checked} (vertex, A) pairs on a 401-point rational grid in A in [0,{Amax}], violations: {bad}")
    return cands

if __name__ == "__main__":
    part1()
    Amax = sp.Rational(sys.argv[1]) if len(sys.argv) > 1 else sp.Rational(1, 2)
    cands = part2(Amax)
    # print the candidate vertices for the write-up
    for v in cands: print("  ", v)
