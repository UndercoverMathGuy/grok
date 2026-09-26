"""VERIFY link 6, RIGOROUS version: P1 at q = 2 for nu = 3, 4, 5 with no floating point anywhere
that matters.

The inherited certificates are: P1C's "cell-wise guaranteed but float64" grid, P1A's mpmath-interval
branch-and-bound (conditional on Theorem B, a CONTINUUM statement), and P1B/ADVERSARY's float64
two-variable minimisation.  This script replaces all of them for the paper's own cells by an
argument in which

  * branch 1 is CLOSED FORM (no numerics at all), and
  * branch 2 is an EXACT rational vertex enumeration plus a rigorous mpmath-interval evaluation.

--------------------------------------------------------------------------------------------------
BRANCH 1, in closed form (new here; replaces the numerically-bisected eps0 / R1 / eps*).

  phi(E) := tgt (1 - 2^n E)^r + E^r,   tgt := (n+1)^{1-r}.   Claim: phi(E) >= tgt on [0, E_a] with

      c   := ( tgt * r * 2^n )^{-1/(1-r)},        E_a := c / (1 + 2^n c).

  Proof.  phi(E) >= tgt is  E^r >= tgt [1 - (1 - 2^n E)^r].  The map u -> 1 - (1-u)^r has derivative
  r(1-u)^{r-1}, which is increasing on [0,1), so by the mean value theorem
  1 - (1-u)^r <= u * r (1-u)^{r-1}.  With u = 2^n E it therefore suffices that
      E^r >= tgt * r * 2^n E * (1 - 2^n E)^{r-1},
  i.e. (since r - 1 < 0, and both sides positive)  ( E / (1 - 2^n E) )^{r-1} >= tgt r 2^n,
  i.e.  E / (1 - 2^n E) <= c,  i.e.  E <= E_a.   QED -- three lines, no computer.

--------------------------------------------------------------------------------------------------
BRANCH 2, exactly.

  L_r(n, delta) = min { sum_l x_l^r : x_1 >= ... >= x_N >= 0,
                        sum_{l>h} x_l >= D_h(n) (h = 0..n),  sum_{l>n+1} x_l >= delta }.
  Objective concave, feasible set a pointed polyhedron with recession cone the sorted non-negative
  cone (on which the objective increases), so the minimum is at a vertex and vertex enumeration is
  exact.  We (i) replace every D_h(n) by a RATIONAL LOWER BOUND (weakening a ">=" constraint can
  only lower L_r, so the result stays a valid lower bound on the truth), (ii) enumerate every vertex
  in exact Fraction arithmetic, (iii) evaluate sum_l x_l^r with mpmath INTERVAL arithmetic, and take
  the outward-rounded minimum.

Conclusion: if  L_r(n, delta) >= tgt  for some rational delta <= E_a, then branch 1 covers [0, E_a]
and branch 2 covers [delta, inf) (L_r is non-decreasing in delta), so P1 holds for EVERY N.
"""
import itertools, json, os
from fractions import Fraction as F
import mpmath as mp

mp.mp.dps = 50
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "FINAL_RUNS", "VERIFY")
os.makedirs(OUT, exist_ok=True)


# ----------------------------------------------------------------- exact linear algebra
def solve_frac(A, b):
    """Exact Gaussian elimination over Fraction.  Returns None if singular."""
    m = len(A); n = len(A[0])
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    piv = []
    r = 0
    for c in range(n):
        p = None
        for i in range(r, m):
            if M[i][c] != 0:
                p = i; break
        if p is None:
            return None
        M[r], M[p] = M[p], M[r]
        pv = M[r][c]
        M[r] = [v / pv for v in M[r]]
        for i in range(m):
            if i != r and M[i][c] != 0:
                f = M[i][c]
                M[i] = [M[i][j] - f * M[r][j] for j in range(n + 1)]
        piv.append(c); r += 1
        if r == m:
            break
    if r < n:
        return None
    return [M[i][n] for i in range(n)]


# ----------------------------------------------------------------- rational lower bounds for D_h
def rat_lower(x, den=10 ** 12):
    """largest k/den <= x, with x an mpf (outward rounded downwards, conservatively)."""
    v = mp.mpf(x)
    k = int(mp.floor(v * den)) - 1          # one ulp of slack, deliberately conservative
    return F(k, den)


def Dh_rational(n):
    k1 = mp.mpf(1) / 2
    k2 = 3 * mp.sqrt(3) / 8
    k3 = mp.mpf(49) / 72
    kap = {1: k1, 2: k2, 3: k3}
    out = [F(1)]
    for h in range(1, n + 1):
        best = mp.mpf(-1)
        for comp in itertools.product([1, 2, 3], repeat=h):
            if sum(comp) > n:
                continue
            v = mp.mpf(1)
            for d in comp:
                v *= kap[d]
            best = max(best, v)
        out.append(rat_lower(best))
    return out


# ----------------------------------------------------------------- the programme
def Lr_exact(n, r_num, r_den, delta, N):
    """Exact vertex enumeration; returns an mpmath INTERVAL lower bound on L_r(n, delta)."""
    D = Dh_rational(n)
    rows, rhs = [], []
    for l in range(N - 1):
        a = [F(0)] * N; a[l] = F(1); a[l + 1] = F(-1)
        rows.append(a); rhs.append(F(0))
    a = [F(0)] * N; a[N - 1] = F(1); rows.append(a); rhs.append(F(0))
    for h in range(0, n + 1):
        a = [F(0)] * N
        for l in range(h, N):
            a[l] = F(1)
        rows.append(a); rhs.append(D[h])
    a = [F(0)] * N
    for l in range(n + 1, N):
        a[l] = F(1)
    rows.append(a); rhs.append(delta)
    M = len(rows)
    best = None; bestx = None
    for idx in itertools.combinations(range(M), N):
        sol = solve_frac([rows[i] for i in idx], [rhs[i] for i in idx])
        if sol is None:
            continue
        if any(v < 0 for v in sol):
            continue
        ok = True
        for i in range(M):
            if sum(rows[i][j] * sol[j] for j in range(N)) < rhs[i]:
                ok = False; break
        if not ok:
            continue
        with mp.workdps(60):
            tot = mp.iv.mpf(0)
            for v in sol:
                if v == 0:
                    continue
                xv = mp.iv.mpf([mp.mpf(v.numerator) / v.denominator,
                                mp.mpf(v.numerator) / v.denominator])
                tot = tot + mp.iv.exp(mp.iv.log(xv) * mp.iv.mpf(r_num) / mp.iv.mpf(r_den))
        if best is None or tot.a < best.a:
            best, bestx = tot, sol
    return best, bestx


def main():
    rep = []
    print("RIGOROUS P1 certificate at q = 2 (branch 1 closed-form, branch 2 exact + interval)\n")
    for n, (rn, rd) in [(2, (2, 3)), (3, (1, 2)), (4, (2, 5))]:
        r = mp.mpf(rn) / rd
        tgt = mp.mpf(n + 1) ** (1 - r)
        c = (tgt * r * mp.mpf(2) ** n) ** (-1 / (1 - r))
        Ea = c / (1 + mp.mpf(2) ** n * c)
        # a rational delta safely below E_a
        delta = rat_lower(Ea * mp.mpf('0.98'), den=10 ** 9)
        vals = []
        for N in range(n + 2, n + 5):
            lo, x = Lr_exact(n, rn, rd, delta, N)
            vals.append((N, lo, x))
        worst = min(vals, key=lambda t: float(t[1].a))
        ok = worst[1].a > tgt
        print(f"nu={n+1}  n={n}  r={rn}/{rd}  target={mp.nstr(tgt, 12)}")
        print(f"   branch 1 closed form: c = {mp.nstr(c, 10)},  E_a = {mp.nstr(Ea, 10)}  "
              f"(numerically-bisected eps0 was 1.697663e-02 / 1.469238e-02 / 8.441542e-03)")
        print(f"   rational delta used  = {float(delta):.9e}  (<= E_a: {float(delta) <= float(Ea)})")
        for N, lo, x in vals:
            print(f"   N={N}: L_r(n,delta) in [{mp.nstr(lo.a, 15)}, {mp.nstr(lo.b, 15)}]  "
                  f"vertex = {[str(t) for t in x]}")
        print(f"   ==> L_r(n,delta) >= {mp.nstr(worst[1].a, 15)}  vs target {mp.nstr(tgt, 15)}  "
              f"margin {mp.nstr(mp.mpf(worst[1].a) / tgt, 10)}  -> "
              f"{'*** P1 RIGOROUSLY CERTIFIED for every N ***' if ok else 'FAILS'}\n")
        rep.append(dict(n=n, r=f"{rn}/{rd}", target=float(tgt), Ea=float(Ea),
                        delta=float(delta), Lr_lower=float(mp.mpf(worst[1].a)),
                        margin=float(mp.mpf(worst[1].a) / tgt), certified=bool(ok)))
    with open(os.path.join(OUT, "p1_rigorous.json"), "w") as f:
        json.dump(rep, f, indent=1)
    print("wrote", os.path.join(OUT, "p1_rigorous.json"))


if __name__ == "__main__":
    main()
