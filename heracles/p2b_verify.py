"""P2B step 0: independent hand-check of every inherited claim P2B leans on.

Checks (all float64 unless stated):

  V1  the CHAIN IDENTITY            log[(n+1)^{n+1} prod_l |c_l|] = (n+1) H + Delta
      H     = mean of log|h| over the (n+1)! kill points
      Delta = (n+1)log(n+1) - (1/n) sum_j log V_j^2      (Hadamard defect, >= 0)
      Claimed exact for EVERY c and v (no constraint).  Tested on random data.

  V2  the KILL-POINT MOMENTS        m_T := mean of u^T over the kill points,
      closed form  m_T = (-1)^{|T|} (n-t)!(n+1-t)/((n+1) n!) * Inj_T,
      Inj_T = sum over injections tau: T -> atoms of prod_{j in T} conj(v_{tau(j),j}),
      and in particular m_{{j}} = -conj(S_j)/(n+1),
      m_{{1,2}} = (conj(S_1)conj(S_2) - Z)/6 at n = 2, Z = sum_l conj(v_l1 v_l2).

  V3  the SCALAR INEQUALITY used in Theorem P2-2:
      log lam <= (lam - 1) - (lam - 1)^2/(2 Lam)  on (0, Lam].
      P2's stated reason ("both sides and their first derivatives agree at lam = 1 and
      at lam = Lam") is FALSE at lam = Lam; the inequality is nevertheless TRUE, by
      psi'(lam) = (lam-1)(Lam-lam)/(lam Lam).  Checked numerically here.

  V4  THEOREM P2-2 itself, and the EXACT n = 2 Vandermonde identity
      V^2 = 27 - 6|P_1|^2 - 3|P_2|^2 + 2 Re(conj(P_1)^2 P_2),
      which is sharp where Theorem P2-2 loses a factor 3.

  V5  PROP P2-4's constant c_n = (n+1)^2/(4n), recomputed from scratch
      (independent code path from p2_secord.py).

  V6  the COLLAPSE THRESHOLD: if any atom vanishes then eta >= 1/sqrt(2^n - 1).
      (Explains P2's measured "Pi hits 0 near eta = 0.58" at n = 2: 1/sqrt3 = 0.5774.)
"""
import itertools, json, os
import numpy as np

OUT = "FINAL_RUNS/P2B"


# ----------------------------------------------------------------- basic objects
def masks(n):
    """All 2^n sign patterns T as boolean rows, T = () first."""
    return np.array([[bool(k & (1 << j)) for j in range(n)] for k in range(2 ** n)])


def CT(c, v):
    """C_T = sum_l c_l prod_{j in T} v_lj for every T, in the order of masks(n)."""
    n = v.shape[1]
    M = masks(n)
    out = np.empty(2 ** n, dtype=complex)
    for t, m in enumerate(M):
        out[t] = np.sum(c * np.prod(v[:, m], axis=1)) if m.any() else np.sum(c)
    return out


def hval(c, v, u):
    """h(u) = sum_l c_l prod_j (1 + u_j v_lj)."""
    return np.sum(c * np.prod(1.0 + u[None, :] * v, axis=1))


def kill_points(v):
    """The (n+1)! kill points u^{l0,sigma}: u_j = -conj(v_{sigma(j),j}).

    l0 runs over the n+1 atoms, sigma over the n! bijections {slots} -> {atoms != l0}.
    At such a u every atom l != l0 is annihilated and only l0 survives.
    """
    N, n = v.shape
    pts = []
    for l0 in range(N):
        others = [l for l in range(N) if l != l0]
        for perm in itertools.permutations(others):
            u = np.array([-np.conj(v[perm[j], j]) for j in range(n)])
            pts.append((l0, perm, u))
    return pts


def vander_logs(v):
    """log V_j^2 for every slot j, V_j = prod_{l<m} |v_lj - v_mj|."""
    N, n = v.shape
    out = np.empty(n)
    for j in range(n):
        s = 0.0
        for l in range(N):
            for m in range(l + 1, N):
                s += 2.0 * np.log(abs(v[l, j] - v[m, j]))
        out[j] = s
    return out


def rand_cfg(n, rng):
    N = n + 1
    v = np.exp(2j * np.pi * rng.random((N, n)))
    c = rng.normal(size=N) + 1j * rng.normal(size=N)
    return c, v


# ----------------------------------------------------------------- V1 chain identity
def check_chain(n, trials, rng):
    worst = 0.0
    for _ in range(trials):
        c, v = rand_cfg(n, rng)
        pts = kill_points(v)
        H = np.mean([np.log(abs(hval(c, v, u))) for _, _, u in pts])
        lv = vander_logs(v)
        Delta = (n + 1) * np.log(n + 1) - lv.sum() / n
        lhs = (n + 1) * np.log(n + 1) + np.sum(np.log(np.abs(c)))
        worst = max(worst, abs(lhs - ((n + 1) * H + Delta)))
    return worst


def check_hadamard_nonneg(n, trials, rng):
    """Delta >= 0 (Hadamard) on random phase configurations."""
    worst = np.inf
    for _ in range(trials):
        v = np.exp(2j * np.pi * rng.random((n + 1, n)))
        lv = vander_logs(v)
        worst = min(worst, (n + 1) * np.log(n + 1) - lv.sum() / n)
    return worst


# ----------------------------------------------------------------- V2 moments
def moments_bruteforce(v):
    N, n = v.shape
    M = masks(n)
    pts = kill_points(v)
    out = np.zeros(2 ** n, dtype=complex)
    for t, m in enumerate(M):
        vals = [np.prod(u[m]) if m.any() else 1.0 for _, _, u in pts]
        out[t] = np.mean(vals)
    return out


def moments_closed(v):
    """m_T = (-1)^t (n-t)!(n+1-t)/((n+1) n!) * sum over injections tau: T -> atoms."""
    from math import factorial
    N, n = v.shape
    M = masks(n)
    out = np.zeros(2 ** n, dtype=complex)
    for t_i, m in enumerate(M):
        slots = [j for j in range(n) if m[j]]
        t = len(slots)
        if t == 0:
            out[t_i] = 1.0
            continue
        s = 0.0 + 0j
        for tau in itertools.permutations(range(N), t):
            s += np.prod([np.conj(v[tau[k], slots[k]]) for k in range(t)])
        coef = (-1) ** t * factorial(n - t) * (n + 1 - t) / ((n + 1) * factorial(n))
        out[t_i] = coef * s
    return out


def moments_n2_special(v):
    """The two formulas P2 states explicitly, at n = 2."""
    S1, S2 = v[:, 0].sum(), v[:, 1].sum()
    Z = np.sum(np.conj(v[:, 0] * v[:, 1]))
    return -np.conj(S1) / 3.0, -np.conj(S2) / 3.0, (np.conj(S1) * np.conj(S2) - Z) / 6.0


# ----------------------------------------------------------------- V3/V4 Hadamard stability
def scalar_ineq_worst(Lam, npts=200001):
    """max over lam in (0,Lam] of  log lam - [(lam-1) - (lam-1)^2/(2 Lam)]  (should be <= 0)."""
    lam = np.linspace(1e-9, Lam, npts)
    return float(np.max(np.log(lam) - ((lam - 1) - (lam - 1) ** 2 / (2 * Lam))))


def p22_ratio(n, trials, rng):
    """max over random slots of V^2 / [ (n+1)^{n+1} exp(-(n+1)^{-3} sum_k (n+1-k)|P_k|^2) ]."""
    N = n + 1
    worst = 0.0
    for _ in range(trials):
        z = np.exp(2j * np.pi * rng.random(N))
        V2 = 1.0
        for l in range(N):
            for m in range(l + 1, N):
                V2 *= abs(z[l] - z[m]) ** 2
        Pk = np.array([np.sum(z ** k) for k in range(1, n + 1)])
        bound = N ** N * np.exp(-(N ** -3) * np.sum((N - np.arange(1, n + 1)) * np.abs(Pk) ** 2))
        worst = max(worst, V2 / bound)
    return worst


def v2_exact_n2(trials, rng):
    """V^2 = 27 - 6|P1|^2 - 3|P2|^2 + 2 Re(conj(P1)^2 P2) for three points on the circle."""
    worst = 0.0
    for _ in range(trials):
        z = np.exp(2j * np.pi * rng.random(3))
        V2 = abs(z[0] - z[1]) ** 2 * abs(z[0] - z[2]) ** 2 * abs(z[1] - z[2]) ** 2
        P1, P2 = z.sum(), np.sum(z ** 2)
        f = 27 - 6 * abs(P1) ** 2 - 3 * abs(P2) ** 2 + 2 * np.real(np.conj(P1) ** 2 * P2)
        worst = max(worst, abs(V2 - f))
    return worst


# ----------------------------------------------------------------- V5 c_n
def cn_independent(n):
    """c_n = min{lam : lam Q - P PSD}, built from scratch (real coordinates).

    Perturbation of the block:  c_l = (1 + x_l + i y_l)/(n+1),  v_lj = w^{-l} e^{i d_lj}.
    Objective (2nd order):  -log[(n+1)^{n+1} prod|c_l|] = (sum x^2 - sum y^2)/2.
    Constraint C_empty = 1 <=> sum_l (x_l + i y_l) = 0.
    Junk:  C_T = (1/(n+1)) sum_l w^{-l|T|} (x_l + i y_l + i sum_{j in T} d_lj).
    """
    N = n + 1
    w = np.exp(2j * np.pi / N)
    M = masks(n)
    dim = 2 * N + N * n

    def cvec(z):
        x, y = z[:N], z[N:2 * N]
        d = z[2 * N:].reshape(N, n)
        eps = x + 1j * y
        rows = []
        for t in range(1, 2 ** n):
            tt = int(M[t].sum())
            rows.append(np.sum(w ** (-np.arange(N) * tt) * (eps + 1j * d[:, M[t]].sum(axis=1))) / N)
        return np.array(rows)

    A = np.zeros((2, dim))
    A[0, :N] = 1.0
    A[1, N:2 * N] = 1.0
    B = np.linalg.svd(A)[2][2:].T
    k = B.shape[1]
    Pm = np.zeros((k, k))
    Qm = np.zeros((k, k))
    for i in range(k):
        for j in range(i, k):
            zi, zj = B[:, i], B[:, j]
            Pm[i, j] = Pm[j, i] = 0.5 * (zi[:N] @ zj[:N] - zi[N:2 * N] @ zj[N:2 * N])
            Qm[i, j] = Qm[j, i] = float(np.real(np.vdot(cvec(zi), cvec(zj))))
    lo, hi = 0.0, 1.0
    while np.linalg.eigvalsh(hi * Qm - Pm).min() < -1e-12:
        hi *= 2
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if np.linalg.eigvalsh(mid * Qm - Pm).min() < -1e-12:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ----------------------------------------------------------------- V6 collapse threshold
def collapse_threshold(n, trials, rng):
    """If c_{l0} = 0 for some l0 then N-1 = n atoms remain; n slots kill them all, so
    h(u) = 0 at that torus point, while |h| >= 1 - sqrt(2^n-1) eta everywhere.
    Hence eta >= 1/sqrt(2^n-1).  Numerically: check h(u) = 0 at the killing point."""
    N = n + 1
    worst = 0.0
    for _ in range(trials):
        v = np.exp(2j * np.pi * rng.random((N, n)))
        c = rng.normal(size=N) + 1j * rng.normal(size=N)
        c[N - 1] = 0.0                      # drop the last atom
        u = np.array([-np.conj(v[j, j]) for j in range(n)])   # kills atoms 0..n-1
        worst = max(worst, abs(hval(c, v, u)))
    return worst


# ----------------------------------------------------------------- main
def main():
    rng = np.random.default_rng(20260921)
    res = {}
    print("== V1  chain identity  log[(n+1)^{n+1} prod|c_l|] = (n+1)H + Delta ==")
    res["V1_chain_max_abs_err"] = {}
    for n in (1, 2, 3, 4):
        e = check_chain(n, 60 if n < 4 else 12, rng)
        res["V1_chain_max_abs_err"][n] = e
        print(f"   n={n}: max |lhs-rhs| = {e:.3e}")

    print("== V1b Delta >= 0 (Hadamard) on random configurations ==")
    res["V1b_min_Delta"] = {}
    for n in (1, 2, 3, 4):
        m = check_hadamard_nonneg(n, 4000 if n < 4 else 800, rng)
        res["V1b_min_Delta"][n] = m
        print(f"   n={n}: min Delta over random v = {m:.6f}  (must be >= 0)")

    print("== V2  kill-point moments: brute force vs closed form ==")
    res["V2_moment_max_abs_err"] = {}
    for n in (1, 2, 3, 4):
        worst = 0.0
        for _ in range(40 if n < 4 else 8):
            v = np.exp(2j * np.pi * rng.random((n + 1, n)))
            worst = max(worst, np.max(np.abs(moments_bruteforce(v) - moments_closed(v))))
        res["V2_moment_max_abs_err"][n] = worst
        print(f"   n={n}: max |m_T(brute) - m_T(closed)| = {worst:.3e}")
    worst = 0.0
    for _ in range(200):
        v = np.exp(2j * np.pi * rng.random((3, 2)))
        mb = moments_bruteforce(v)          # order: (), {1}, {2}, {1,2}
        m1, m2, m12 = moments_n2_special(v)
        worst = max(worst, abs(mb[1] - m1), abs(mb[2] - m2), abs(mb[3] - m12))
    res["V2_n2_special_max_abs_err"] = worst
    print(f"   n=2 P2's explicit formulas: max abs err = {worst:.3e}")

    print("== V3  scalar inequality log lam <= (lam-1) - (lam-1)^2/(2 Lam) on (0,Lam] ==")
    res["V3_scalar_worst"] = {}
    for Lam in (2, 3, 4, 5, 7):
        wv = scalar_ineq_worst(Lam)
        res["V3_scalar_worst"][Lam] = wv
        print(f"   Lam={Lam}: max(lhs-rhs) = {wv:.3e}  (must be <= 0)")
    # P2's stated reason is false: check the two sides do NOT agree at lam = Lam.
    gaps = {}
    for Lam in (2, 3, 4, 5, 7):
        gaps[Lam] = float(((Lam - 1) - (Lam - 1) ** 2 / (2 * Lam)) - np.log(Lam))
    res["V3_gap_at_Lam"] = gaps
    print("   gap rhs-lhs AT lam=Lam (P2 claims 0; it is not):",
          {k: round(v, 6) for k, v in gaps.items()})

    print("== V4  Theorem P2-2, and the exact n=2 Vandermonde identity ==")
    res["V4_p22_max_ratio"] = {}
    for n in (1, 2, 3, 4, 5):
        r = p22_ratio(n, 20000, rng)
        res["V4_p22_max_ratio"][n] = r
        print(f"   n={n}: max V^2/bound = {r:.9f}  (must be <= 1)")
    e = v2_exact_n2(20000, rng)
    res["V4_v2_exact_n2_max_err"] = e
    print(f"   n=2 exact V^2 identity: max abs err = {e:.3e}")

    print("== V5  Prop P2-4:  c_n = (n+1)^2/(4n) ==")
    res["V5_cn"] = {}
    for n in range(1, 7):
        v = cn_independent(n)
        pred = (n + 1) ** 2 / (4.0 * n)
        res["V5_cn"][n] = {"measured": v, "pred": pred, "ratio": v / pred}
        print(f"   n={n}: c_n = {v:.10f}  pred = {pred:.10f}  ratio = {v/pred:.10f}")

    print("== V6  collapse: a vanishing atom forces h = 0 at a torus point ==")
    res["V6_max_abs_h"] = {}
    for n in (1, 2, 3, 4):
        e = collapse_threshold(n, 2000, rng)
        res["V6_max_abs_h"][n] = e
        print(f"   n={n}: max |h(kill-all point)| = {e:.3e}   => eta >= 1/sqrt(2^n-1) "
              f"= {1/np.sqrt(2**n-1):.6f}")

    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(f"{OUT}/verify.json", "w"), indent=1, default=float)
    print(f"wrote {OUT}/verify.json")


if __name__ == "__main__":
    main()
