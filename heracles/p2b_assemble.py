"""P2B step 4: the assembled Quadratic Product Lemma at n = 2, and its constant.

THE CHAIN (every step proved in P2B.md; this script checks each one numerically and then
computes the constant).

 (1) CHAIN IDENTITY          Phi := -log[27 prod_l |c_l|] = -3 H - Delta.
 (2) SCALAR LOG BOUND        log|1+g| >= Re g - Cdel |g|^2 for |g| <= delta,
                             Cdel = (-log(1-delta) - delta)/delta^2, delta = sqrt3 eta.
 (3) MOMENTS + GRAM          H >= -eta ||m|| - Cdel ||G|| eta^2, so
                             Phi <= 3 eta ||m|| + 3 Cdel ||G|| eta^2 - Delta.
 (4) CROSS-SLOT LEMMA        |Z| <= (|Q_1|+|Q_2|)/2 + F,  F <= sum_l 2 sin^2(psi_l) <= 6|a|^2,
                             psi_l = arg(1 - a conj(x_l)),  a = the Moebius parameter,
                             eps = |a|/sqrt(2+|a|^2)  =>  |a|^2 = 2 eps^2/(1-eps^2).
 (5) PER-SLOT LEMMA (PS_rho) 8|S_j|^2 + |Q_j|^2 <= rho * 36 Delta_j   (rho = 1.15;
                             certified separately in p2b_cert_ps.py).
 (6) ASSEMBLY                ||m||^2 <= B(Delta, F) := max over U of
                             [ U/9 + (U + min(6, sqrt(2(K-8U))) + 2F)^2/144 ],  K = 36 rho Delta,
                             U in [0, min(18, K/8)];   and  ||G|| <= Gb(Delta).
 (7) CONSTANT                C(eta0) := sup over 0 < eta <= eta0, Delta >= 0 of
                             [3 eta sqrt(B) + 3 Cdel(eta) Gb eta^2 - Delta]/eta^2,
                             and then  27 prod|c_l| >= exp(-C(eta0) eta^2)  for eta <= eta0.
"""
import argparse, itertools, json, os
import numpy as np

OUT = "FINAL_RUNS/P2B"
RHO = 1.15                        # certified upper bound for the (PS) ratio
SQ3 = np.sqrt(3.0)


# ---------------------------------------------------------------- basic n=2 objects
def CT(c, x, y):
    return np.array([c.sum(), np.sum(c * x), np.sum(c * y), np.sum(c * x * y)])


def kill_pts(x, y):
    pts = []
    for l0 in range(3):
        o = [l for l in range(3) if l != l0]
        for p in itertools.permutations(o):
            pts.append((np.array([-np.conj(x[p[0]]), -np.conj(y[p[1]])]), l0))
    return pts


def phase_quant(x, y):
    S1, S2 = x.sum(), y.sum()
    Q1, Q2 = np.sum(x * x), np.sum(y * y)
    Y = np.sum(np.conj(x) * y)
    Z = np.sum(np.conj(x * y))
    V2a = 27 - 6 * abs(S1) ** 2 - 3 * abs(Q1) ** 2 + 2 * np.real(np.conj(S1) ** 2 * Q1)
    V2b = 27 - 6 * abs(S2) ** 2 - 3 * abs(Q2) ** 2 + 2 * np.real(np.conj(S2) ** 2 * Q2)
    V2a, V2b = max(V2a, 0.0), max(V2b, 0.0)
    Delta = np.inf if (V2a <= 0 or V2b <= 0) else -0.5 * (np.log(V2a / 27) + np.log(V2b / 27))
    Wc = np.conj(S1 * S2) - Z
    m = np.array([-np.conj(S1) / 3, -np.conj(S2) / 3, Wc / 6])
    G = np.array([[1.0 + 0j, (np.conj(S1) * S2 - Y) / 6, -S2 / 3],
                  [np.conj((np.conj(S1) * S2 - Y) / 6), 1.0 + 0j, -S1 / 3],
                  [np.conj(-S2 / 3), np.conj(-S1 / 3), 1.0 + 0j]])
    t = x * y
    L3 = -((x[0] - x[2]) * (y[1] - y[2]) - (x[1] - x[2]) * (y[0] - y[2]))
    cols = np.stack([np.ones(3, dtype=complex), x, y, t], axis=1)
    L = np.array([(-1) ** k * np.linalg.det(np.delete(cols, k, axis=1)) for k in range(4)])
    tail = np.linalg.norm(L[1:])
    eps = abs(L[0]) / tail if tail > 1e-13 else 0.0
    return dict(S1=S1, S2=S2, Q1=Q1, Q2=Q2, Z=Z, Delta=Delta, m=m, G=G, eps=eps,
                mnorm=float(np.linalg.norm(m)),
                Gnorm=float(np.linalg.eigvalsh((G + G.conj().T) / 2).max()))


# ---------------------------------------------------------------- the explicit bounds
def cdel(eta):
    d = SQ3 * eta
    return (-np.log(1 - d) - d) / d ** 2


def Bfun(Delta, F, nU=4000):
    """max over U in [0, min(18, K/8)] of U/9 + (U + min(6,sqrt(2(K-8U))) + 2F)^2/144."""
    K = 36.0 * RHO * Delta
    Umax = min(18.0, K / 8.0)
    if Umax <= 0:
        return min(3.0, (2 * F) ** 2 / 144.0)
    U = np.linspace(0.0, Umax, nU)
    W = np.minimum(6.0, np.sqrt(np.maximum(2 * (K - 8 * U), 0.0)))
    val = U / 9.0 + (U + W + 2 * F) ** 2 / 144.0
    return float(min(3.0, val.max()))


def Gbfun(Delta):
    U = min(18.0, 4.5 * RHO * Delta)
    s = np.sqrt(U)
    r12 = (U / 2 + 3) / 6.0
    return float(min(3.0, 1.0 + max(r12 + s / 3.0, np.sqrt(2 * U) / 3.0)))


def Ffun(eta):
    """F <= 6|a|^2 and |a|^2 = 2 eps^2/(1-eps^2) <= 2 eta^2/(1-eta^2)."""
    return 12.0 * eta ** 2 / (1.0 - eta ** 2)


def Cconst(eta0, neta=600, ndel=4000):
    """C(eta0) = sup over eta<=eta0, Delta>=0 of [3 eta sqrt(B) + 3 cdel Gb eta^2 - Delta]/eta^2."""
    best, arg = -np.inf, None
    etas = np.linspace(eta0 / neta, eta0, neta)
    for eta in etas:
        cd = cdel(eta)
        F = Ffun(eta)
        # only Delta <= 3 sqrt3 eta can contribute (||m|| <= sqrt3)
        Dmax = 3 * SQ3 * eta + 1e-9
        Ds = np.concatenate([np.linspace(0, Dmax, ndel), [0.0]])
        for D in Ds:
            v = (3 * eta * np.sqrt(Bfun(D, F, nU=400)) + 3 * cd * Gbfun(D) * eta ** 2 - D) / eta ** 2
            if v > best:
                best, arg = v, (float(eta), float(D))
    return best, arg


# ---------------------------------------------------------------- numeric checks
NULLB = np.array([[1.0, 1.0], [-1.0, 0.0], [0.0, -1.0]], dtype=complex)  # basis of sum c_l = 0


def rand_admissible(rng, eta, amax_frac=0.9):
    """Random admissible configuration: C_empty = 1 and junk energy <= eta^2.

    Phases are built from the Moebius picture (Lemma M3) with |a| small enough that
    eps(v) <= amax_frac * eta, so feasible atoms exist; then c is the minimum-junk atom
    vector perturbed inside the affine set {sum_l c_l = 1} until the junk is <= eta.
    """
    e_t = amax_frac * eta
    amax = e_t * np.sqrt(2.0) / np.sqrt(max(1 - e_t ** 2, 1e-12))
    for _ in range(200):
        th = rng.uniform(0, 2 * np.pi)
        al = rng.uniform(0, amax) * np.exp(1j * rng.uniform(0, 2 * np.pi))
        x = np.exp(1j * rng.uniform(0, 2 * np.pi, 3))
        y = np.exp(1j * th) * (x - al) / (1 - np.conj(al) * x)
        x, y = x / x[2], y / y[2]                       # gauge x_3 = y_3 = 1
        M = np.stack([np.ones(3, dtype=complex), x, y, x * y], axis=0)   # 4 x 3
        cp = np.array([1.0 + 0j, 0, 0])
        A = M[1:] @ NULLB                               # 3 x 2
        b = -(M[1:] @ cp)
        s0, *_ = np.linalg.lstsq(A, b, rcond=None)      # minimum-junk point
        for _ in range(60):
            s = s0 + (rng.normal(size=2) + 1j * rng.normal(size=2)) * eta * rng.random()
            c = cp + NULLB @ s
            C = M @ c
            if abs(C[0] - 1) > 1e-12:
                continue
            j = float(np.linalg.norm(C[1:]))
            if 1e-6 < j <= eta and np.min(np.abs(c)) > 1e-12:
                return c, x, y, j
    raise RuntimeError("no admissible configuration found")


def check_master(rng, trials, eta):
    worst_gap = np.inf
    worst_ratio = -np.inf
    for _ in range(trials):
        c, x, y, e = rand_admissible(rng, eta)
        q = phase_quant(x, y)
        Phi = -(np.log(27) + np.sum(np.log(np.abs(c))))
        d = SQ3 * e
        if d >= 1 or e <= 0:
            continue
        rhs = 3 * e * q["mnorm"] + 3 * cdel(e) * q["Gnorm"] * e ** 2 - q["Delta"]
        worst_gap = min(worst_gap, rhs - Phi)
        worst_ratio = max(worst_ratio, Phi / e ** 2)
    return worst_gap, worst_ratio


def check_cross(rng, trials, amax):
    """|Z| <= (|Q1|+|Q2|)/2 + sum 2 sin^2(psi_l) <= (|Q1|+|Q2|)/2 + 6|a|^2."""
    w1 = w2 = -np.inf
    for _ in range(trials):
        th = rng.uniform(0, 2 * np.pi)
        a = rng.uniform(0, amax) * np.exp(1j * rng.uniform(0, 2 * np.pi))
        x = np.exp(1j * rng.uniform(0, 2 * np.pi, 3))
        y = np.exp(1j * th) * (x - a) / (1 - np.conj(a) * x)
        Q1, Q2 = np.sum(x * x), np.sum(y * y)
        Z = np.sum(np.conj(x * y))
        psi = np.angle(1 - a * np.conj(x))
        F = abs(Z) - 0.5 * (abs(Q1) + abs(Q2))
        w1 = max(w1, F - np.sum(2 * np.sin(psi) ** 2))
        w2 = max(w2, F - 6 * abs(a) ** 2)
    return w1, w2


def check_B(rng, trials, eta):
    """||m||^2 <= B(Delta, F(eta)) on admissible configurations."""
    worst = -np.inf
    for _ in range(trials):
        c, x, y, e = rand_admissible(rng, eta)
        q = phase_quant(x, y)
        if not np.isfinite(q["Delta"]) or q["eps"] > e + 1e-12:
            continue
        b = Bfun(q["Delta"], Ffun(e), nU=800)
        worst = max(worst, q["mnorm"] ** 2 - b)
    return worst


def check_qpl(rng, trials, eta, C):
    """the end statement: 27 prod|c_l| >= exp(-C eta^2)."""
    worst = np.inf
    for _ in range(trials):
        c, x, y, e = rand_admissible(rng, eta)
        Phi = -(np.log(27) + np.sum(np.log(np.abs(c))))
        worst = min(worst, C * e ** 2 - Phi)
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=3000)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(31415)
    res = {"rho": RHO}

    print("== check (4): cross-slot lemma ==", flush=True)
    res["cross"] = {}
    for amax in (0.05, 0.2, 0.5, 0.9):
        w1, w2 = check_cross(rng, 60000, amax)
        res["cross"][f"{amax:.2f}"] = {"vs_sum2sin2": w1, "vs_6a2": w2}
        print(f"   |a|<={amax:.2f}:  max[F - sum 2sin^2 psi] = {w1:.3e}   "
              f"max[F - 6|a|^2] = {w2:.3e}   (both must be <= 0)", flush=True)

    print("\n== check (3): the master inequality Phi <= 3 eta||m|| + 3 Cdel ||G|| eta^2 - Delta ==",
          flush=True)
    res["master"] = {}
    for eta in (0.05, 0.1, 0.2, 0.3):
        g, r = check_master(rng, a.trials, eta)
        res["master"][f"{eta:.2f}"] = {"min_slack": g, "max_Phi_over_eta2": r}
        print(f"   eta<={eta:.2f}: min (rhs - Phi) = {g:.6f} (must be >= 0);  "
              f"max Phi/eta^2 seen = {r:.4f}", flush=True)

    print("\n== check (6): ||m||^2 <= B(Delta, F) ==", flush=True)
    res["Bcheck"] = {}
    for eta in (0.05, 0.1, 0.2):
        w = check_B(rng, a.trials, eta)
        res["Bcheck"][f"{eta:.2f}"] = w
        print(f"   eta<={eta:.2f}: max[||m||^2 - B] = {w:.3e}  (must be <= 0)", flush=True)

    print("\n== (7) the constant C(eta0), and the thresholds it buys ==", flush=True)
    rho_need = {"nu3_q2": 0.2440169, "nu3_q2.5": 0.1138872}
    res["C"] = {}
    for eta0 in (0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30):
        C, arg = Cconst(eta0, neta=220, ndel=900)
        thr = 3 * rho_need["nu3_q2"] / C
        jt = 3.0 / (2 * SQ3 * C)
        res["C"][f"{eta0:.3f}"] = {"C": C, "argmax_eta_Delta": arg,
                                   "chord_threshold": thr, "JT_threshold_times_K": jt,
                                   "valid": bool(thr <= eta0)}
        print(f"   eta0={eta0:.3f}: C = {C:.6f}  (argmax eta={arg[0]:.4f}, Delta={arg[1]:.5f})"
              f"   chord thr = {thr:.5f} {'<= eta0 OK' if thr <= eta0 else '> eta0'}"
              f"   (JT) thr = {jt:.5f}/K", flush=True)

    # end-to-end sanity on the best cell
    best = max((v["C"] for k, v in res["C"].items()))
    print("\n== check: 27 prod|c_l| >= exp(-C eta^2) on random admissible configurations ==",
          flush=True)
    res["qpl_end"] = {}
    for eta0 in (0.05, 0.1, 0.2):
        C = res["C"][f"{eta0:.3f}"]["C"]
        w = check_qpl(rng, a.trials, eta0, C)
        res["qpl_end"][f"{eta0:.2f}"] = w
        print(f"   eta0={eta0:.2f}, C={C:.4f}: min (C eta^2 - Phi) = {w:.6f} (must be >= 0)",
              flush=True)

    json.dump(res, open(f"{OUT}/assemble.json", "w"), indent=1, default=float)
    print(f"\nwrote {OUT}/assemble.json", flush=True)


if __name__ == "__main__":
    main()
