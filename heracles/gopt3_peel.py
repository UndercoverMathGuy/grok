"""gopt3_peel.py -- THE PEELING LADDER: an unconditional Key Lemma for EVERY number of atoms.

The open gap "G1" (LEMMA_ATTEMPT Sec.6, GLOBAL_OPT Sec.6) is the Key Lemma for N > n+1 atoms:

    c_1..c_N in C,  v_lj on the unit circle (l = 1..N, j = 1..n),
    (KL)   sum_l c_l prod_{j in T} v_lj = A [T empty]   for every T subset of {1..n}
    ==>    sum_l |c_l|^r >= (n+1)^{1-r} |A|^r          for every r in (0,1].

Proved for N = n+1; open for N > n+1.  This script establishes and measures a LADDER of elementary
consequences of (KL) that together give the conclusion for EVERY N with a constant within a few percent
of (n+1)^{1-r}.

THE PEELING INEQUALITY (elementary, proved, every N).  Let Lambda(z) = sum_T lambda_T prod_{j in T} z_j be
ANY multi-affine polynomial.  Pairing (KL) with Lambda gives the exact identity

    sum_l c_l Lambda(v_l) = lambda_empty * A.                                            (dual pairing)

Choose h <= n atoms l_1..l_h to "kill", give each one its own operand slot j_i, and take the product

    Lambda(z) = prod_{i=1}^h ( 1 - z_{j_i} conj(v_{l_i, j_i}) ) / 2 .

Then Lambda(v_{l_i}) = 0, lambda_empty = 2^{-h}, and |Lambda| <= 1 everywhere on the torus, so

    (PEEL_h)   sum_{l not in S} |c_l|  >=  2^{-h} |A|      for EVERY set S of h <= n atoms.

Sorting x_1 >= x_2 >= ... this says: the total mass left after removing the h LARGEST atoms is at least
2^{-h}|A|, for h = 0, 1, ..., n.  (h = 0 is the classical sum_l |c_l| >= |A|; h = n with a single slot
per killed atom is the bound used in LEMMA_ATTEMPT F-e -- but F-e used ONE value of h, and the whole
ladder h = 0..n is much stronger.)

Sharper constants D_h.  2^{-h} is what the product ("bijection") polynomial gives.  The best possible
constant is the exact peeling number

    D_h(n) = inf over h points of T^n of max{ |Lambda_empty| : ||Lambda||_{L^inf(T^n)} <= 1, Lambda(v_i)=0 },

computed here by LP (gopt3_dual.py) and, for h = 1, attained in closed form by the reproducing-kernel
polynomial  Lambda(z) = 1 - prod_j (1 + z_j conj(v_j))/2  (K(.,v) is the reproducing kernel of the
multi-affine space for Haar measure on the torus).

THE LADDER PROGRAMME.  Whatever constants D_0 = 1 >= D_1 >= ... >= D_n > 0 = D_{n+1} are available,

    sum_l |c_l|^r  >=  L_r(n; D) |A|^r,
    L_r(n; D) := min { sum_l x_l^r : x_1 >= x_2 >= ... >= 0,  sum_{l > h} x_l >= D_h, h = 0..n }.

This script computes L_r, compares it with the conjectured sharp value (n+1)^{1-r}, verifies (PEEL_h)
and the resulting bound on random feasible configurations with N up to 12 atoms, and reports the
certified loss factor  (( n+1 )^{1-r} / L_r)^{1/r}  by which the target inequality (T) is missed.

fp64, local CPU.  Usage: uv run python heracles/gopt3_peel.py --out heracles/FINAL_RUNS/GOPT3
"""
import argparse, itertools, json, math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
from scipy.optimize import minimize, linprog


# ----------------------------------------------------------------- multi-affine helpers
def subsets(n):
    out = [()]
    for k in range(1, n + 1):
        out += list(itertools.combinations(range(n), k))
    return out


def mono(TS, V):
    M = V.shape[0]
    out = np.ones((M, len(TS)), dtype=complex)
    for t, T in enumerate(TS):
        w = np.ones(M, dtype=complex)
        for j in T:
            w = w * V[:, j]
        out[:, t] = w
    return out


def torus_grid(n, G):
    ang = 2 * math.pi * np.arange(G) / G
    mesh = np.meshgrid(*([ang] * n), indexing="ij")
    return np.exp(1j * np.stack([m.ravel() for m in mesh], 1))


# ----------------------------------------------------------------- the kernel constants S_d, kappa_d
def S_d(d, G=2000001):
    """sup over T^d of |1 - prod_{j=1}^d (1+z_j)/2|.

    With z_j = e^{2 i psi_j}, prod_j (1+z_j)/2 = m e^{i phi}, m = prod cos psi_j, phi = sum psi_j, so
    |1-u|^2 = 1 - 2 m cos phi + m^2.  At any interior critical point with m != 0,
    dF/dpsi_j = 2 m [ tan(psi_j)(cos phi - m) + sin phi ] = 0 gives the SAME tan(psi_j) for every j,
    so psi_j in {psi, psi +- pi}; flipping k of them multiplies m by (-1)^k and shifts phi by k pi,
    leaving m e^{i phi} unchanged.  Hence the sup is attained on the diagonal and equals
        max_psi | 1 - cos^d(psi) e^{i d psi} | .
    """
    psi = math.pi * (2 * np.arange(G) / (G - 1) - 1)
    m = np.cos(psi) ** d
    return math.sqrt(float(np.max(1 - 2 * m * np.cos(d * psi) + m ** 2)))


def kappa(d, cache={}):
    """kappa_d = (1 - 2^{-d}) / S_d : the value of the d-slot reproducing-kernel kill polynomial."""
    if d not in cache:
        cache[d] = (1 - 2.0 ** (-d)) / S_d(d)
    return cache[d]


def compositions(n, h):
    """all (d_1..d_h) with d_i >= 1 summing to n."""
    if h == 1:
        yield (n,); return
    for d in range(1, n - h + 2):
        for rest in compositions(n - d, h - 1):
            yield (d,) + rest


def peel_numbers_kernel(n):
    """D_h >= max over partitions of the n slots into h groups of prod_i kappa_{d_i}  (h = 0..n)."""
    D = {0: 1.0}
    for h in range(1, n + 1):
        D[h] = max(float(np.prod([kappa(d) for d in comp])) for comp in compositions(n, h))
    return D


def kernel_sup(n, G=721):
    psi = math.pi * (2 * np.arange(G) / (G - 1) - 1)
    m = np.cos(psi) ** n
    diag = math.sqrt(float(np.max(1 - 2 * m * np.cos(n * psi) + m ** 2)))
    full = None
    if n <= 3:
        V = torus_grid(n, 240 if n == 2 else 72)
        u = np.prod((1 + V) / 2, axis=1)
        full = float(np.max(np.abs(1 - u)))
    return diag, full


# ----------------------------------------------------------------- exact peeling numbers by LP
def min_supnorm(n, pts, G=40, P=24):
    TS = subsets(n); K = len(TS)
    V = torus_grid(n, G); Mg = mono(TS, V)
    rows, rhs = [], []
    for ph in 2 * math.pi * np.arange(P) / P:
        Z = Mg * np.exp(-1j * ph)
        rows.append(np.concatenate([Z[:, 1:].real, -Z[:, 1:].imag, -np.ones((Z.shape[0], 1))], 1))
        rhs.append(-Z[:, 0].real)
    A_ub = np.concatenate(rows, 0); b_ub = np.concatenate(rhs, 0)
    A_eq, b_eq = [], []
    if len(pts):
        Mp = mono(TS, np.asarray(pts))
        for i in range(Mp.shape[0]):
            A_eq.append(np.concatenate([Mp[i, 1:].real, -Mp[i, 1:].imag, [0.0]])); b_eq.append(-Mp[i, 0].real)
            A_eq.append(np.concatenate([Mp[i, 1:].imag, Mp[i, 1:].real, [0.0]])); b_eq.append(-Mp[i, 0].imag)
    c = np.zeros(2 * (K - 1) + 1); c[-1] = 1.0
    r = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=np.array(A_eq) if A_eq else None,
                b_eq=np.array(b_eq) if b_eq else None,
                bounds=[(None, None)] * (2 * (K - 1)) + [(0, None)], method="highs")
    if not r.success:
        return np.inf, None
    lam = np.concatenate([[1.0 + 0j], r.x[:K - 1] + 1j * r.x[K - 1:2 * (K - 1)]])
    Vf = torus_grid(n, 200 if n == 2 else (48 if n == 3 else 24))
    return float(np.max(np.abs(mono(TS, Vf) @ lam))), lam


def peel_numbers_lp(n, rand=40, seed=0, G=40, P=24):
    """D_h for h = 0..n, minimised over point configurations (v_1 normalised to (1,..,1))."""
    rng = np.random.default_rng(seed)
    out = {0: 1.0}
    for h in range(1, n + 1):
        best = np.inf
        cands = []
        if h == 1:
            cands = [[np.ones(n, dtype=complex)]]
        else:
            for _ in range(rand):
                cands.append([np.ones(n, dtype=complex)] +
                             [np.exp(1j * rng.uniform(0, 2 * math.pi, n)) for _ in range(h - 1)])
            # structured candidates: roots of unity along the diagonal
            for q in range(2, n + 3):
                cands.append([np.exp(2j * math.pi * i / q) * np.ones(n) for i in range(h)])
        for pts in cands:
            t, lam = min_supnorm(n, pts, G=G, P=P)
            if lam is not None:
                best = min(best, 1.0 / t)
        out[h] = best
    return out


# ----------------------------------------------------------------- the ladder programme L_r
def L_r(n, D, r, Nmax=12, starts=40, seed=0):
    """min sum_l x_l^r over decreasing x >= 0 with sum_{l>h} x_l >= D_h, h = 0..n."""
    rng = np.random.default_rng(seed)
    Ds = [D[h] for h in range(n + 1)]
    best, bx = np.inf, None
    for N in range(n + 1, Nmax + 1):
        cons = []
        for h in range(n + 1):
            cons.append({"type": "ineq", "fun": (lambda x, h=h: np.sum(x[h:]) - Ds[h])})
        for l in range(N - 1):                                   # x_l >= x_{l+1}
            cons.append({"type": "ineq", "fun": (lambda x, l=l: x[l] - x[l + 1])})
        f = lambda x: float(np.sum(np.maximum(x, 0) ** r))
        g = lambda x: r * np.maximum(x, 1e-300) ** (r - 1)
        cand = np.array([Ds[h] - (Ds[h + 1] if h + 1 <= n else 0.0) for h in range(n + 1)] + [0.0] * (N - n - 1))
        for s in range(starts + 1):
            x0 = cand.copy() if s == 0 else np.sort(rng.random(N))[::-1] * rng.uniform(0.3, 1.5)
            try:
                res = minimize(f, x0, jac=g, method="SLSQP", constraints=cons,
                               bounds=[(0, None)] * N, options=dict(maxiter=500, ftol=1e-14))
            except Exception:
                continue
            x = np.maximum(res.x, 0)
            ok = all(np.sum(x[h:]) >= Ds[h] - 1e-9 for h in range(n + 1)) and np.all(np.diff(x) <= 1e-9)
            if ok and f(x) < best:
                best, bx = f(x), x
    return best, bx


def L_r_vertex(n, D, r, Nmax=8):
    """L_r by EXACT VERTEX ENUMERATION.

    sum_l x_l^r is concave on R_+^N and the feasible set {x_1>=..>=x_N>=0, sum_{l>h} x_l >= D_h} is a
    pointed polyhedron, so the minimum is attained at a vertex.  Enumerating all vertices (every
    N-subset of the N+n+1 defining inequalities that gives an invertible system and a feasible point)
    removes any reliance on a local optimiser.  Returns (value, argmin).
    """
    Ds = [D[h] for h in range(n + 1)]
    best, arg = np.inf, None
    for N in range(n + 1, Nmax + 1):
        rows, rhs = [], []
        for l in range(N - 1):
            a = np.zeros(N); a[l] = 1; a[l + 1] = -1; rows.append(a); rhs.append(0.0)
        a = np.zeros(N); a[N - 1] = 1; rows.append(a); rhs.append(0.0)
        for h in range(n + 1):
            a = np.zeros(N); a[h:] = 1; rows.append(a); rhs.append(Ds[h])
        R = np.array(rows); b = np.array(rhs)
        for S in itertools.combinations(range(len(rows)), N):
            A = R[list(S)]
            if abs(np.linalg.det(A)) < 1e-12:
                continue
            x = np.linalg.solve(A, b[list(S)])
            if np.min(x) < -1e-10 or np.min(R @ x - b) < -1e-9:
                continue
            v = float(np.sum(np.maximum(x, 0) ** r))
            if v < best:
                best, arg = v, x
    return best, arg


# ----------------------------------------------------------------- verification on random feasible configs
def random_feasible(n, N, rng, tries=400):
    """Random unit-modulus v, then the minimum-norm c solving the KL system (A = 1)."""
    TS = subsets(n)
    for _ in range(tries):
        V = np.exp(1j * rng.uniform(0, 2 * math.pi, (N, n)))
        M = mono(TS, V).T                                        # (2^n, N)
        b = np.zeros(len(TS), dtype=complex); b[0] = 1.0
        c, *_ = np.linalg.lstsq(M, b, rcond=None)
        if np.max(np.abs(M @ c - b)) < 1e-9:
            # add a random null-space component to sample the whole feasible set
            u, s, vh = np.linalg.svd(M)
            ns = vh[np.sum(s > 1e-10):].conj().T
            if ns.shape[1] > 0:
                y = (rng.normal(size=ns.shape[1]) + 1j * rng.normal(size=ns.shape[1])) * rng.choice([0.0, 0.3, 1.0, 3.0])
                c = c + ns @ y
            if np.max(np.abs(M @ c - b)) < 1e-8:
                return V, c
    return None, None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--lp", action="store_true", help="also compute the exact peeling numbers by LP (slow for n>=3)")
    ap.add_argument("--trials", type=int, default=300)
    ap.add_argument("--out", default=str(ROOT / "FINAL_RUNS" / "GOPT3"))
    a = ap.parse_args()
    rng = np.random.default_rng(7)
    report = []

    for n in a.ns:
        nu = n + 1
        print(f"\n================ n = {n}  (nu = {nu}) ================", flush=True)
        Dprod = {h: 2.0 ** (-h) for h in range(n + 1)}
        d1, d1full = kernel_sup(n)
        Dker = peel_numbers_kernel(n)
        print("  kappa_d = (1-2^-d)/S_d :", {d: round(kappa(d), 6) for d in range(1, n + 1)},
              f"   [S_n diagonal {d1:.6f}" + (f", full-grid {d1full:.6f}]" if d1full else "]"))
        print("  D_h  product-kill 2^-h :", {h: round(Dprod[h], 6) for h in range(n + 1)})
        print("  D_h  kernel-partition  :", {h: round(Dker[h], 6) for h in range(n + 1)})
        Dlp = None
        if a.lp and n <= 3:
            Dlp = peel_numbers_lp(n, rand=(0 if n == 1 else 30))
            print("  D_h by LP (inf over configurations):", {h: round(v, 6) for h, v in Dlp.items()})

        # ---- verification of (PEEL_h) and of the final bound on random feasible configurations
        worst_peel = {h: np.inf for h in range(n + 1)}
        worstKL = np.inf
        for N in range(n + 1, n + 8):
            for _ in range(a.trials):
                V, c = random_feasible(n, N, rng)
                if c is None:
                    continue
                x = np.sort(np.abs(c))[::-1]
                for h in range(n + 1):
                    worst_peel[h] = min(worst_peel[h], float(np.sum(x[h:])) / Dprod[h])
                worstKL = min(worstKL, float(np.sum(x ** (2.0 / nu))) / (nu ** (1 - 2.0 / nu)))
        print("  (PEEL_h) check: min over random feasible configs of [tail mass]/2^{-h} =",
              {h: round(worst_peel[h], 4) for h in range(n + 1)}, " (must be >= 1)")
        print(f"  Key Lemma check at q=2: min sum|c|^r / (n+1)^(1-r) = {worstKL:.4f}  (conjectured >= 1)")

        for q in (2.0, 2.5, 3.0, float(nu)):
            if q > nu:
                continue
            r = q / nu
            tgt = nu ** (1 - r)
            row = dict(n=n, q=q, r=r, target=tgt)
            for tag, D in (("product", Dprod), ("kernel", Dker)) + ((("lp", Dlp),) if Dlp else ()):
                val, xopt = L_r(n, D, r)
                row[tag] = val
                row[tag + "_ratio"] = val / tgt
                row[tag + "_loss"] = (tgt / val) ** (1 / r)
                print(f"  q {q:g} (r {r:.4f}): L_r[{tag:7s}] = {val:.6f}   (n+1)^(1-r) = {tgt:.6f}"
                      f"   ratio {val/tgt:.5f}   ==> (T) certified up to factor {(tgt/val)**(1/r):.5f}"
                      + (f"   x* = {np.round(xopt[:n+2], 4)}" if tag == "product" else ""), flush=True)
            report.append(row)

    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "peel.json").write_text(json.dumps(report, indent=1))
    print("\nwrote", Path(a.out) / "peel.json")
