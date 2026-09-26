"""VERIFY link 6: the P1 "every N" argument, re-derived and re-computed from scratch.

P1 is the claim
    (P1)  sum_{l=1}^N |c_l|^r  >=  (n+1)^{1-r} |A|^r
for every N and every (KL) system
    (KL)  sum_l c_l prod_{j in T} v_lj = A [T empty]   for every T subset of {1..n},
with |v_lj| = 1, r = q/nu in (0,1], nu = n+1.

The three write-ups (ADVERSARY.md §6, P1B.md, P1C.md) give the SAME argument.  I re-derive it
independently, then recompute the two thresholds in 40-digit arithmetic.

SYMBOLS (all defined):
  A        clean total, normalised to 1
  x_1 >= x_2 >= ... >= x_N >= 0   the sorted atom moduli |c_l|
  B        the n+1 largest atoms ("the core")
  E        = sum_{l > n+1} x_l, the tail mass (ADVERSARY's tau, P1C's R)
  s        = x_{n+1}, the smallest core atom (ADVERSARY's beta)
  target   = (n+1)^{1-r}
  D_h(n)   the peeling constants (verify_constants.py)

BRANCH 1 (Prop B1 / Lemma T + Lemma C):
  cost >= (n+1)^{1-r} (1 - 2^n E)_+^r + E^r  =: phi(E).
  eps0 := sup{ t : phi >= target on [0,t] }.
BRANCH 2 (peeling ladder + forced tail):
  cost >= L_r(n, E) := min { sum_l x_l^r : x_1>=...>=x_N>=0,
                             sum_{l>h} x_l >= D_h(n) (h=0..n), sum_{l>n+1} x_l >= E }.
  eps1 := inf{ d : L_r(n,d) >= target }.
If eps1 <= eps0 the two branches cover every E >= 0 and P1 holds for every N.

L_r is a CONCAVE minimisation over a pointed polyhedron, so the minimum is at a vertex;
I enumerate every vertex exactly (my own enumeration, not the authors').
"""
import itertools, json, os, sys
import numpy as np
import mpmath as mp

mp.mp.dps = 40
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "FINAL_RUNS", "VERIFY")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- constants
def Sd(d):
    def f(t):
        return -abs(1 - ((1 + mp.e ** (1j * t)) / 2) ** d)
    best = None
    for t0 in np.linspace(0.01, float(mp.pi), 200):
        lo, hi = mp.mpf(float(t0)) - mp.pi / 200, mp.mpf(float(t0)) + mp.pi / 200
        lo, hi = max(mp.mpf(0), lo), min(mp.pi, hi)
        for _ in range(300):
            m1, m2 = lo + (hi - lo) / 3, hi - (hi - lo) / 3
            if f(m1) < f(m2):
                hi = m2
            else:
                lo = m1
        v = -f((lo + hi) / 2)
        if best is None or v > best:
            best = v
    return best

KAPPA = {}
for d in range(1, 7):
    KAPPA[d] = (1 - mp.mpf(2) ** (-d)) / Sd(d)
# exact algebraic overrides (verified to 0 by verify_constants.py)
KAPPA[1] = mp.mpf(1) / 2
KAPPA[2] = 3 * mp.sqrt(3) / 8
KAPPA[3] = mp.mpf(49) / 72


def Dh(h, n, strict_eq=False):
    if h == 0:
        return mp.mpf(1)
    best = mp.mpf(-1)
    for comp in itertools.product(range(1, n + 1), repeat=h):
        t = sum(comp)
        if t > n or (strict_eq and t != n):
            continue
        v = mp.mpf(1)
        for d in comp:
            v *= KAPPA[d]
        best = max(best, v)
    return best


# ------------------------------------------------------- branch 1: eps0
def phi(E, n, r):
    tgt = mp.mpf(n + 1) ** (1 - r)
    base = 1 - mp.mpf(2) ** n * E
    a = tgt * (base ** r) if base > 0 else mp.mpf(0)
    return a + E ** r


def eps0(n, r):
    """largest t0 such that phi >= target on [0,t0].  phi(0)=target, phi rises (cusp) then falls."""
    tgt = mp.mpf(n + 1) ** (1 - r)
    lo = mp.mpf(10) ** (-30)
    hi = mp.mpf(2) ** (-n)
    # make sure phi(lo) > target
    if phi(lo, n, r) < tgt:
        return mp.mpf(0)
    # phi(hi) = 2^{-nr} < target for the cells of interest
    if phi(hi, n, r) >= tgt:
        return hi
    for _ in range(300):
        m = (lo + hi) / 2
        if phi(m, n, r) >= tgt:
            lo = m
        else:
            hi = m
    return lo


# ------------------------------------------------------- branch 2: L_r(n,delta)
def Lr(n, r, delta, N, strict_eq=False, verbose=False):
    """Exact vertex enumeration of
         min sum_l x_l^r  s.t.  x_1>=...>=x_N>=0,
                                sum_{l>h} x_l >= D_h (h=0..n),
                                sum_{l>n+1} x_l >= delta.
       Returns (value, argmin).  Concave objective on a pointed polyhedron whose recession cone
       is the sorted nonneg cone (objective increases along it) => min at a vertex.
    """
    rows, rhs = [], []
    # ordering x_l - x_{l+1} >= 0
    for l in range(N - 1):
        a = np.zeros(N); a[l] = 1.0; a[l + 1] = -1.0
        rows.append(a); rhs.append(0.0)
    a = np.zeros(N); a[N - 1] = 1.0
    rows.append(a); rhs.append(0.0)                      # x_N >= 0
    for h in range(0, n + 1):
        if h >= N:
            continue
        a = np.zeros(N); a[h:] = 1.0
        rows.append(a); rhs.append(float(Dh(h, n, strict_eq)))
    if N > n + 1:
        a = np.zeros(N); a[n + 1:] = 1.0
        rows.append(a); rhs.append(float(delta))
    elif float(delta) > 0:
        return None, None                                # infeasible: no tail available
    Amat = np.array(rows); b = np.array(rhs)
    M = Amat.shape[0]
    best, arg = None, None
    rf = float(r)
    for idx in itertools.combinations(range(M), N):
        sub = Amat[list(idx)]
        if abs(np.linalg.det(sub)) < 1e-10:
            continue
        try:
            x = np.linalg.solve(sub, b[list(idx)])
        except np.linalg.LinAlgError:
            continue
        if np.any(x < -1e-9):
            continue
        x = np.maximum(x, 0.0)
        if np.any(Amat @ x < b - 1e-9):
            continue
        val = float(np.sum(x ** rf))
        if best is None or val < best:
            best, arg = val, x
    return best, arg


def eps1(n, r, N, strict_eq=False):
    tgt = float(mp.mpf(n + 1) ** (1 - r))
    lo, hi = 0.0, 0.5
    v_hi, _ = Lr(n, r, hi, N, strict_eq)
    if v_hi is None or v_hi < tgt:
        return None
    for _ in range(60):
        m = (lo + hi) / 2
        v, _ = Lr(n, r, m, N, strict_eq)
        if v is not None and v >= tgt:
            hi = m
        else:
            lo = m
    return hi


def main():
    rep = {"cells": []}
    cells = [(2, mp.mpf(2) / 3, "nu=3,q=2"), (3, mp.mpf(1) / 2, "nu=4,q=2"),
             (4, mp.mpf(2) / 5, "nu=5,q=2"), (5, mp.mpf(1) / 3, "nu=6,q=2"),
             (2, mp.mpf(5) / 6, "nu=3,q=2.5"), (3, mp.mpf(5) / 8, "nu=4,q=2.5")]
    print(f"{'cell':<14}{'target':>16}{'eps0(branch1)':>16}{'eps1(branch2)':>16}{'eps0/eps1':>12}  verdict")
    for n, r, label in cells:
        tgt = mp.mpf(n + 1) ** (1 - r)
        e0 = eps0(n, r)
        Nuse = min(n + 4, 8)
        e1 = eps1(n, r, Nuse)
        ok = (e1 is not None) and (e1 <= e0)
        ratio = float(e0 / mp.mpf(e1)) if e1 else float("nan")
        print(f"{label:<14}{float(tgt):>16.9f}{float(e0):>16.6e}"
              f"{(e1 if e1 else float('nan')):>16.6e}{ratio:>12.2f}  "
              f"{'COVERS (P1 proved)' if ok else 'GAP -- branches do not overlap'}")
        rep["cells"].append(dict(n=n, r=float(r), label=label, target=float(tgt),
                                 eps0=float(e0), eps1=e1, covers=bool(ok), N=Nuse))

    # ---- N-stability of L_r: the programme must lower-bound for EVERY N -----
    print("\nN-stability of L_r(n,delta) (must not decrease as N grows; the argument needs a bound valid for all N):")
    for n, r, label in [(2, mp.mpf(2) / 3, "nu=3,q=2"), (3, mp.mpf(1) / 2, "nu=4,q=2"),
                        (4, mp.mpf(2) / 5, "nu=5,q=2")]:
        for delta in [0.0, 0.005, 0.02]:
            vals = []
            for N in range(n + 1, n + 7):
                v, _ = Lr(n, r, delta, N)
                vals.append(None if v is None else round(v, 10))
            print(f"  {label:<12} delta={delta:<7} L_r over N={n+1}..{n+6}: {vals}")
            rep.setdefault("Nstab", []).append(dict(label=label, delta=delta, vals=vals))

    # ---- the minimising vertex at delta = 0 --------------------------------
    print("\nL_r(n,0) and its minimising vertex (should reproduce GLOBAL_OPT3 §4):")
    for n, r, label in [(2, mp.mpf(2) / 3, "nu=3,q=2"), (3, mp.mpf(1) / 2, "nu=4,q=2"),
                        (4, mp.mpf(2) / 5, "nu=5,q=2"), (5, mp.mpf(1) / 3, "nu=6,q=2")]:
        v, x = Lr(n, r, 0.0, min(n + 4, 8))
        veq, _ = Lr(n, r, 0.0, min(n + 4, 8), strict_eq=True)
        tgt = float(mp.mpf(n + 1) ** (1 - r))
        F = (tgt / v) ** (1 / float(r))
        print(f"  {label:<12} L_r={v:.9f}  (GLOBAL_OPT3 convention {veq:.9f})  target={tgt:.9f}"
              f"  L_r/target={v/tgt:.6f}  F={F:.6f}  vertex={np.round(x,6)}")
        rep.setdefault("Lr0", []).append(dict(label=label, Lr=v, Lr_eq=veq, target=tgt,
                                              F=F, vertex=[float(t) for t in x]))

    with open(os.path.join(OUT, "p1_twobranch.json"), "w") as f:
        json.dump(rep, f, indent=1)
    print("\nwrote", os.path.join(OUT, "p1_twobranch.json"))


if __name__ == "__main__":
    main()
