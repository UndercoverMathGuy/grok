"""tri_ineq2.py -- symbolic PROOF of the 3-variable inequality of the transfer lemma on A in [0, Amax]:

   L(a) := 2(a1+a2+a3) + sum_{i != j} |a_i - A a_j|  >=  2(2+A) sqrt(a1^2+a2^2+a3^2),   a >= 0.

Both sides are positively homogeneous of degree 1, so restrict to the simplex sum a = 1. The 6 planes
{a_i = A a_j} and the 3 faces {a_i = 0} cut the simplex into convex cells on each of which L is LINEAR
(fixed sign pattern) while 2(2+A)||a||_2 is convex, so L - RHS is concave on each cell and its minimum is
at a cell vertex. Cell vertices are the solutions of {sum a = 1} plus two independent equations from the
9 linear forms. For each such vertex v(A) (a rational function of A) and each of the 6 forms we locate the
sign changes of v_i - A v_j on [0, Amax] exactly (polynomial roots), so that on each sub-interval L(v(A)) is
a rational function of A; we then verify L^2 - RHS^2 >= 0 there by exact root counting (sympy Sturm) plus a
sign check at an interior rational point. Vertices with a negative coordinate on a sub-interval are skipped
(they are not in the simplex there).
"""
import itertools, sys
import sympy as sp

A = sp.symbols('A', real=True)
a = sp.symbols('a1 a2 a3', real=True)

def roots_in(expr, lo, hi):
    """real roots of a polynomial/rational numerator in [lo, hi] (exact, via sympy)"""
    num = sp.numer(sp.together(expr))
    P = sp.Poly(sp.expand(num), A)
    if P.degree() <= 0: return []
    rs = []
    for r_ in sp.real_roots(P):
        if lo <= r_ <= hi: rs.append(r_)
    return sorted(set(rs))

def nonneg_on(expr, lo, hi):
    """is the rational function expr >= 0 on the OPEN interval (lo,hi)? exact: no numerator/denominator root inside, sign at midpoint"""
    expr = sp.together(expr)
    num, den = sp.numer(expr), sp.denom(expr)
    inside = [r_ for r_ in roots_in(num, lo, hi) if lo < r_ < hi] + [r_ for r_ in roots_in(den, lo, hi) if lo < r_ < hi]
    if inside: return None   # sign may change; caller splits
    mid = (lo + hi) / 2
    return sp.sign(expr.subs(A, mid)) >= 0

def prove(Amax):
    forms = [a[i] - A*a[j] for i in range(3) for j in range(3) if i != j] + list(a)
    verts = {}
    for f1, f2 in itertools.combinations(forms, 2):
        sol = sp.solve([f1, f2, sum(a) - 1], a, dict=True)
        if not sol or len(sol[0]) < 3: continue
        v = tuple(sp.simplify(sol[0][x]) for x in a)
        verts[v] = True
    verts = list(verts); print(f"{len(verts)} distinct symbolic vertices")
    nchecks = 0; ok = True
    for v in verts:
        # breakpoints: roots of coordinates and of the 6 difference forms
        bps = {sp.Integer(0), Amax}
        exprs = list(v) + [v[i] - A*v[j] for i in range(3) for j in range(3) if i != j]
        for ex in exprs:
            for r_ in roots_in(ex, 0, Amax): bps.add(r_)
            for r_ in roots_in(sp.denom(sp.together(ex)), 0, Amax): bps.add(r_)
        bps = sorted(bps)
        for lo, hi in zip(bps[:-1], bps[1:]):
            mid = (lo + hi) / 2
            if any(sp.sign(x.subs(A, mid)) < 0 for x in v): continue     # outside the simplex on this piece
            # fixed signs of the difference forms on (lo,hi)
            L = 2*sum(v)
            for i in range(3):
                for j in range(3):
                    if i == j: continue
                    g = v[i] - A*v[j]; s = sp.sign(g.subs(A, mid))
                    L += s*g if s != 0 else 0
            rhs2 = 4*(2 + A)**2 * sum(x**2 for x in v)
            diff = sp.together(L**2 - rhs2)
            res = nonneg_on(diff, lo, hi)
            nchecks += 1
            # closed endpoints: check by continuity (diff is continuous where den != 0); check endpoints directly
            end_ok = all(sp.sign(diff.subs(A, e)) >= 0 for e in (lo, hi) if sp.denom(diff).subs(A, e) != 0)
            if res is None:
                # split at interior roots of diff and re-check each piece
                rs = [r_ for r_ in roots_in(sp.numer(diff), lo, hi) if lo < r_ < hi]
                pts = [lo] + rs + [hi]; res = all(nonneg_on(diff, p0, p1) for p0, p1 in zip(pts[:-1], pts[1:]))
            if not (res and end_ok):
                ok = False; print("FAIL at vertex", v, "on", (float(lo), float(hi)), "min diff sample", float(diff.subs(A, mid)))
    print(f"pieces checked: {nchecks}; PROVED on [0,{Amax}]" if ok else "NOT proved")
    return ok

if __name__ == "__main__":
    Amax = sp.Rational(sys.argv[1]) if len(sys.argv) > 1 else sp.Rational(47, 100)
    prove(Amax)
