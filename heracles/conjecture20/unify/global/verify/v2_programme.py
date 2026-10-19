# Independent high-precision programme computations (mpmath, 40 digits), for R6/R7/R12 numbers.
#   Coverage polytope P_p = {v >= 0 : sum_j v_j kappa_j(delta) >= 1, delta = 1..F};
#   Gamma^{(S)}_s = max over vertices v of P_p with spt v in S of (sum_j v_j^{1/s})^{-s}   (concave minimisation).
#   delta_min(s) = min over NONEMPTY S containing no optimal support of 1 - Gamma^{(S)}/Gamma; eps_tol = delta_min/(1-delta_min).
import itertools, sys
import mpmath as mp

mp.mp.dps = 40


def kap(p, j, d):
    return 1 - mp.cos(2 * mp.pi * j * d / p)


def vertices(p):
    F = (p - 1) // 2
    rows = [[kap(p, j, d) for j in range(1, F + 1)] for d in range(1, F + 1)] + \
           [[mp.mpf(1) if j == k else mp.mpf(0) for j in range(F)] for k in range(F)]
    rhs = [mp.mpf(1)] * F + [mp.mpf(0)] * F
    V = []
    for act in itertools.combinations(range(2 * F), F):
        M = mp.matrix([rows[a] for a in act]); bb = mp.matrix([rhs[a] for a in act])
        if abs(mp.det(M)) < mp.mpf(10) ** -25:
            continue
        v = mp.lu_solve(M, bb)
        v = [v[i] if abs(v[i]) > mp.mpf(10) ** -30 else mp.mpf(0) for i in range(F)]
        if min(v) < -mp.mpf(10) ** -30:
            continue
        ok = all(mp.fsum(rows[d][j] * v[j] for j in range(F)) >= 1 - mp.mpf(10) ** -30 for d in range(F))
        if not ok:
            continue
        if any(max(abs(a - b) for a, b in zip(v, w)) < mp.mpf(10) ** -25 for w in V):
            continue
        V.append(v)
    return V


def val(v, s):
    return mp.fsum([x ** (1 / s) for x in v if x > 0]) ** (-s)


def supp(v):
    return tuple(j + 1 for j, x in enumerate(v) if x > 0)


def analyse(p, V, s):
    F = (p - 1) // 2
    vals = [(val(v, s), supp(v)) for v in V]
    G = max(x for x, _ in vals)
    opt = [S for x, S in vals if x >= G * (1 - mp.mpf(10) ** -25)]
    worst = None
    for k in range(1, F + 1):
        for S in itertools.combinations(range(1, F + 1), k):
            if any(set(T) <= set(S) for T in opt):
                continue
            GS = max([x for x, T in vals if set(T) <= set(S)] + [mp.mpf(0)])
            if worst is None or GS > worst[0]:
                worst = (GS, S)
    if worst is None:
        return G, opt, mp.inf, None
    dmin = 1 - worst[0] / G
    return G, opt, dmin, worst[1]


if __name__ == "__main__":
    grid = [mp.mpf(x) for x in "1.05 1.2 1.35 1.5 1.7 1.85 1.86 2.0 2.3 2.5".split()]
    allV = {}
    for p in [5, 7, 11, 13]:
        V = vertices(p); allV[p] = V
        print(f"p={p}: {len(V)} vertices", flush=True)
        for s in grid:
            G, opt, dmin, S = analyse(p, V, s)
            eps = dmin / (1 - dmin) if dmin != mp.inf else mp.inf
            print(f"  s={mp.nstr(s,4)}: Gamma={mp.nstr(G,8)} K*={sorted(set(len(T) for T in opt))} delta_min={mp.nstr(dmin,6)} eps_tol={mp.nstr(eps,6)} worst S={S}", flush=True)

    # transitions: K*(s) changes; bisection on max over support sizes
    def kstar(p, s):
        vals = [(val(v, s), len(supp(v))) for v in allV[p]]; G = max(x for x, _ in vals)
        return sorted(set(k for x, k in vals if x >= G * (1 - mp.mpf(10) ** -30)))
    for p in [5, 7, 11, 13]:
        prev = None; s = mp.mpf(1)
        while s <= 3.0:
            ks = kstar(p, s)
            if prev is not None and ks != prev[1]:
                lo, hi = prev[0], s
                for _ in range(100):
                    mid = (lo + hi) / 2
                    if kstar(p, mid) == prev[1]: lo = mid
                    else: hi = mid
                print(f"p={p}: transition K* {prev[1]} -> {ks} at s_c = {mp.nstr((lo+hi)/2, 15)}", flush=True)
            prev = (s, ks); s += mp.mpf('0.005')

    # p=5 dense threshold (R6(a) / Note Thm dense): 3^{s-1} Gamma^{({1})} < Gamma_s  <=>  6^s < 7.5/kappa_min
    km = kap(5, 1, 1)
    sd = mp.log(mp.mpf('7.5') / km) / mp.log(6)
    print("p=5,n=2 dense threshold s_dense =", mp.nstr(sd, 12), "  (REPORT/Note quote 1.331 / 1.3309)")
    for s in [sd - mp.mpf('1e-6'), mp.mpf('1.3309'), mp.mpf('1.331')]:
        print(f"   s={mp.nstr(s,10)}: 3^(s-1) * kappa_min / Gamma_s = {mp.nstr(3 ** (s - 1) * km / (mp.mpf('2.5') * 2 ** (-s)), 12)}  (<1 needed for dense)")
    # R12 slack at (5,2): 1+eps_tol = Gamma_s/kappa_min on (s_dense, 1.5]: decreasing in s, min at 1.5
    for s in [sd, mp.mpf('1.4'), mp.mpf('1.5')]:
        print(f"(5,2) s={mp.nstr(s,8)}: Gamma_s/kappa_min = {mp.nstr(mp.mpf('2.5') * 2 ** (-s) / km, 10)}")
    # R12 (7,2): 1+eps_tol = Gamma/Gamma^{(S)} for 2-sets at b=2, 2.2, 2.5 (s = 3/b), and the dense threshold at p=7
    V7 = allV[7]
    for b in ['2', '2.2', '2.5', '2.6', '2.8']:
        s = 3 / mp.mpf(b); G = max(val(v, s) for v in V7)
        G12 = max(val(v, s) for v in V7 if set(supp(v)) <= {1, 2})
        G13 = max(val(v, s) for v in V7 if set(supp(v)) <= {1, 3})
        G23 = max(val(v, s) for v in V7 if set(supp(v)) <= {2, 3})
        print(f"(7,2) b={b} s={mp.nstr(s,6)}: Gamma/Gamma^(12)={mp.nstr(G/G12,6)} Gamma/Gamma^(13)={mp.nstr(G/G13,6)} Gamma/Gamma^(23)={mp.nstr(G/G23,6)}  dense needs 3^(s-1) < min ratio: 3^(s-1)={mp.nstr(3**(s-1),6)}")
    # rigorous-ish: K* = F on [1,1.5] at p=5,7: V non-increasing in s, so on [a,b] uniform >= V_unif(b) and others <= V_v(a)
    for p in [5, 7]:
        F = (p - 1) // 2; V = allV[p]
        unif = [v for v in V if len(supp(v)) == F]
        assert len(unif) == 1
        others = [v for v in V if len(supp(v)) < F]
        N = 3000; ok = True; minratio = mp.inf
        for k in range(N):
            a = 1 + mp.mpf(k) / N * mp.mpf('0.5'); bb = 1 + mp.mpf(k + 1) / N * mp.mpf('0.5')
            r = val(unif[0], bb) / max(val(v, a) for v in others)
            minratio = min(minratio, r)
            if r <= 1: ok = False
        print(f"p={p}: uniform vertex strictly optimal on [1,1.5] (monotone interval bound, {N} cells): {ok}, min ratio {mp.nstr(minratio,8)}")
