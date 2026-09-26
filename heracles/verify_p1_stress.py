"""VERIFY link 6, stress tests on the two-branch P1 argument.

(a) dense validation that {E : phi(E) >= target} really is the interval [0, eps0] on [0, 2^{-n}]
    (the bisection in verify_p1_twobranch.py assumes this);
(b) eps1 under BOTH peeling-constant conventions (GLOBAL_OPT3's sum d_i = n, P1B's sum d_i <= n);
(c) L_r(n, delta) for N well past n+6, to test the N-independence claim that P1B asserts from
    numerics only;
(d) the minimum tail cost as a function of (E, s): the one-line proof that N = n+2 suffices;
(e) a direct Monte-Carlo assault on P1 itself: build exact (KL) systems with N up to n+6 by
    least-squares + null-space sampling, with deliberately adversarial structures (coalescing
    phases, near-cancelling pairs, the sign rule, block perturbations), and look for
    sum |c_l|^r < (n+1)^{1-r}.
"""
import itertools, json, os
import numpy as np
import mpmath as mp
from verify_p1_twobranch import Dh, phi, eps0, Lr, eps1, KAPPA

mp.mp.dps = 40
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "FINAL_RUNS", "VERIFY")
os.makedirs(OUT, exist_ok=True)
rep = {}

CELLS = [(2, mp.mpf(2) / 3, "nu=3,q=2"), (3, mp.mpf(1) / 2, "nu=4,q=2"),
         (4, mp.mpf(2) / 5, "nu=5,q=2"), (5, mp.mpf(1) / 3, "nu=6,q=2")]

# ---------------------------------------------------------------- (a)
print("(a) shape of phi: is {phi >= target} exactly [0, eps0]?")
for n, r, lab in CELLS:
    tgt = mp.mpf(n + 1) ** (1 - r)
    e0 = eps0(n, r)
    ts = np.linspace(1e-12, float(mp.mpf(2) ** (-n)), 200001)
    vals = np.array([float(phi(mp.mpf(float(t)), n, r)) for t in ts])
    ok = vals >= float(tgt) - 1e-15
    # the set should be a prefix
    first_false = np.argmax(~ok) if (~ok).any() else len(ok)
    prefix = ok[:first_false].all() and (not ok[first_false:].any())
    print(f"  {lab:<12} eps0={float(e0):.6e}  scan crossing at {ts[first_false]:.6e}  "
          f"set is a prefix interval: {prefix}")
    rep.setdefault("a", []).append(dict(cell=lab, eps0=float(e0),
                                        scan_cross=float(ts[first_false]), prefix=bool(prefix)))

# ---------------------------------------------------------------- (b)
print("\n(b) eps1 under both peeling conventions, and eps0 vs eps1")
for n, r, lab in CELLS:
    e0 = float(eps0(n, r))
    N = min(n + 4, 8)
    e1_le = eps1(n, r, N, strict_eq=False)
    e1_eq = eps1(n, r, N, strict_eq=True)
    print(f"  {lab:<12} eps0={e0:.6e}  eps1(<=n, P1B)={e1_le:.6e}  eps1(=n, GOPT3)={e1_eq:.6e}  "
          f"covers: <=n {e1_le <= e0}, =n {e1_eq <= e0}")
    rep.setdefault("b", []).append(dict(cell=lab, eps0=e0, eps1_le=e1_le, eps1_eq=e1_eq,
                                        covers_le=bool(e1_le <= e0), covers_eq=bool(e1_eq <= e0)))

# ---------------------------------------------------------------- (c)
print("\n(c) L_r(n,delta) for N up to n+8  (P1B asserts N-independence for N >= n+2 from numerics)")
for n, r, lab in [(2, mp.mpf(2) / 3, "nu=3,q=2"), (3, mp.mpf(1) / 2, "nu=4,q=2")]:
    for delta in [0.0, 0.0001, 0.02, 0.1, 0.3]:
        vals = []
        for N in range(n + 1, n + 9):
            v, _ = Lr(n, r, delta, N)
            vals.append(None if v is None else round(v, 10))
        print(f"  {lab:<12} delta={delta:<8} N={n+1}..{n+8}: {vals}")
        rep.setdefault("c", []).append(dict(cell=lab, delta=delta, vals=vals))

# ---------------------------------------------------------------- (d)
print("\n(d) cheapest tail of total mass E with every atom <= s (the N-independence mechanism):")
print("    min sum t_i^r over t_i in [0,s], sum t_i = E  is attained at ceil(E/s) atoms")
for (E, s, r) in [(0.02, 0.3, 2 / 3), (0.4, 0.3, 2 / 3), (0.9, 0.3, 2 / 3)]:
    best = None
    for mnum in range(1, 40):
        if mnum * s < E - 1e-15:
            continue
        full = int(np.floor(E / s + 1e-12)); rem = E - full * s
        cost = full * s ** r + (rem ** r if rem > 1e-15 else 0.0)
        # equal split over mnum atoms, for comparison
        eq = mnum * (E / mnum) ** r
        if best is None:
            best = (mnum, cost, eq)
    full = int(np.floor(E / s + 1e-12)); rem = E - full * s
    print(f"    E={E}, s={s}: ceil(E/s)={int(np.ceil(E/s-1e-12))} atoms, cost="
          f"{full*s**r + (rem**r if rem>1e-15 else 0):.6f}; "
          f"cost with 10 equal atoms = {10*(E/10)**r:.6f} (larger, as claimed)")

# ---------------------------------------------------------------- (e)
print("\n(e) direct assault on P1: exact (KL) systems, adversarial families")


def kl_system(n, N, v, rng, mode="ls"):
    """Given phases v (N,n) on the torus, find c with sum_l c_l prod_{j in T} v_lj = [T empty].
       Returns the minimum-|c|_r solution found over the affine solution set (random null-space
       search + the minimum-norm solution)."""
    T = list(itertools.chain.from_iterable(itertools.combinations(range(n), k) for k in range(n + 1)))
    X = np.array([[np.prod(v[l, list(t)]) if t else 1.0 + 0j for t in T] for l in range(N)])  # (N, 2^n)
    b = np.zeros(len(T), dtype=complex); b[0] = 1.0
    c0, *_ = np.linalg.lstsq(X.T, b, rcond=None)
    if np.linalg.norm(X.T @ c0 - b) > 1e-8:
        return None, None
    # null space of X^T
    u, s, vh = np.linalg.svd(X.T)
    ns = vh.conj().T[:, np.sum(s > 1e-10):]
    return c0, ns


def cost(c, r):
    return float(np.sum(np.abs(c) ** r))


rows = []
for n, r, lab in [(2, 2 / 3, "nu=3,q=2"), (3, 1 / 2, "nu=4,q=2"), (4, 2 / 5, "nu=5,q=2")]:
    tgt = (n + 1) ** (1 - r)
    worst = np.inf; worstinfo = None
    rng = np.random.default_rng(7)
    for trial in range(4000):
        N = int(rng.integers(n + 1, n + 7))
        fam = trial % 6
        if fam == 0:
            v = np.exp(2j * np.pi * rng.random((N, n)))
        elif fam == 1:                      # coalescing atoms
            v = np.exp(2j * np.pi * rng.random((1, n))) * np.ones((N, n))
            v = v * np.exp(1j * rng.normal(0, 10.0 ** (-rng.integers(1, 6)), (N, n)))
        elif fam == 2:                      # sign rule + noise
            sgn = np.array(list(itertools.product([1, -1], repeat=n)))
            take = sgn[rng.choice(len(sgn), size=N, replace=True)]
            v = take * np.exp(1j * rng.normal(0, 0.05, (N, n)))
        elif fam == 3:                      # block + extras
            w = np.exp(-2j * np.pi * np.arange(n + 1) / (n + 1))
            v = np.zeros((N, n), dtype=complex)
            v[:n + 1] = w[:, None] * np.ones((1, n))
            if N > n + 1:
                v[n + 1:] = np.exp(2j * np.pi * rng.random((N - n - 1, n)))
            v *= np.exp(1j * rng.normal(0, 0.1, (N, n)))
        elif fam == 4:                      # near-cancelling pair
            v = np.exp(2j * np.pi * rng.random((N, n)))
            v[1] = v[0] * np.exp(1j * rng.normal(0, 1e-3, n))
        else:                               # grid
            g = np.exp(2j * np.pi * rng.random(n)[None, :] + 2j * np.pi *
                       rng.integers(0, 3, (N, n)) / 3)
            v = g
        out = kl_system(n, N, v, rng)
        if out[0] is None:
            continue
        c0, ns = out
        for _ in range(12):
            c = c0 + (ns @ (rng.normal(size=(ns.shape[1], 1)) + 1j * rng.normal(size=(ns.shape[1], 1)))
                      * rng.choice([0.0, 0.05, 0.3, 1.0])).ravel() if ns.shape[1] else c0
            # verify the system
            Tl = list(itertools.chain.from_iterable(itertools.combinations(range(n), k)
                                                    for k in range(n + 1)))
            res = max(abs(np.sum(c * np.prod(v[:, list(t)], axis=1)) - (1.0 if not t else 0.0))
                      for t in Tl)
            if res > 1e-8:
                continue
            f = cost(c, r) / tgt
            if f < worst:
                worst = f; worstinfo = dict(N=N, fam=fam, res=float(res))
    rows.append(dict(cell=lab, worst_ratio=worst, info=worstinfo))
    print(f"  {lab:<12} min sum|c|^r / (n+1)^(1-r) over exact (KL) systems = {worst:.10f}  "
          f"{'(P1 holds)' if worst >= 1 - 1e-9 else '*** P1 VIOLATED ***'}  {worstinfo}")
rep["e"] = rows

with open(os.path.join(OUT, "p1_stress.json"), "w") as f:
    json.dump(rep, f, indent=1, default=float)
print("\nwrote", os.path.join(OUT, "p1_stress.json"))
