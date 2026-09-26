"""ladder_core.py -- EXACT ladders: gamma_con(K) and obj*(K) for constructed junk-free nets.

Extends fvf_obj.py from (q = 2, equal amplitudes) to (any 2 <= q <= nu, SHARE-WEIGHTED blocks),
for both exact families (gated chain `prod` and the x^n family `pow`).

SYMBOLS (every one defined; see LADDER.md for the plain-language reading)
  p         prime modulus; task = full table of a_1 + ... + a_n = c (mod p), all p^n rows.
  n         number of operands.               nu = n + 1 = homogeneity degree of the net.
  nf        (p-1)/2 = number of positive frequencies k = 1..nf.
  q         penalty exponent: P_q(theta) = sum_i ||theta_i||^q, neuron-wise Euclidean norms.
            q = 2 is plain weight decay; q = nu is Morwani's L_{2,nu}.
  s         = nu/q, the PROGRAMME EXPONENT ("how superlinearly amplitude is priced").
  gamma_q   = margin / P_q^{nu/q}, the normalized margin (scale-invariant because the net is
            nu-homogeneous and P_q is q-homogeneous).
  g_k(D)    = 1 - cos(2 pi k D / p), the "coverage" of the wrong class at distance D by frequency k.
  Gamma_s(p)= max over shares b_k >= 0 with sum_k b_k = 1 of  min_{D != 0} sum_k b_k^s g_k(D).
            b_k = share of the l_q budget on frequency k.  K* = #{k : b_k > 0} at the optimum.
  val(S,b)  = min_{D != 0} sum_{k in S} b_k^s g_k(D), the programme's value at one support/share vector.
  B         = 2 (2 p (n+1))^{-nu/2}, the constant of the unrestricted upper bound gamma_q <= B*Gamma_s
            for the chain (times n! for x^n, whose clean coefficient carries the multinomial n!).
  block     N neurons per frequency with phases arranged so that ALL junk cancels exactly; the
            chain's block has N = n+1 (THEORY_nsum sec 8), the x^n block is searched for here.
  mu        clean-coefficient multiplier of the family: 1 (chain), n! (x^n).

THE IDENTITY THIS FILE IS BUILT ON (derived in LADDER.md sec 2, checked numerically below):
    a junk-free block on support S with shares b has
        gamma_q  =  mu * N^{-(nu-q)/q} * B * val(S, b),
    and its whole logit table depends only on D = (sum_j a_j) - c, so BOTH the margin and the
    cross-entropy are functions of the p-1 numbers  gap(D) = -sum_k A_k g_k(D),  D != 0.

Local CPU, fp64.  No training.
"""
import itertools
import json
import math
import sys
from math import comb
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from kstar import G  # noqa: E402  (G(p)[k-1, D-1] = 1 - cos(2 pi k D / p))
from nu4_program import solve as prog_solve  # noqa: E402  (free-amplitude SLSQP on one support)

OUT = ROOT / "FINAL_RUNS" / "LADDER"


# ----------------------------------------------------------------- the programme Gamma_s
def gmat(p):
    """Rows k = 1..nf, columns D = 1..p-1 of g_k(D) = 1 - cos(2 pi k D / p)."""
    return G(p)


def val_of(g, S, b, s):
    """val(S,b) = min_{D != 0} sum_{k in S} b_k^s g_k(D).  S is 1-based frequency list."""
    idx = np.asarray(S) - 1
    return float(((np.asarray(b) ** s) @ g[idx]).min())


def solve_support(g, S, s, restarts=8, seed=0):
    """max over shares b on the FIXED support S of val(S,b).  Returns (value, b)."""
    idx = np.asarray(S) - 1
    K = len(idx)
    if K == 1:
        return float(g[idx[0]].min()), np.ones(1)
    rng = np.random.default_rng(seed)
    starts = [np.full(K, 1.0 / K)] + [rng.dirichlet(np.full(K, 3.0)) for _ in range(restarts)]
    best, bb = -1.0, None
    for b0 in starts:
        v, b = prog_solve(g[idx], s, b0)
        if v > best:
            best, bb = v, b
    return best, bb


def coverage_topK(g, K, shortlist=300, exhaustive_limit=3_000_000, restarts=200, seed=0):
    """Candidate K-supports ranked by the EQUAL-share coverage M(S) = min_D sum_{k in S} g_k(D).

    WLOG 1 in S: multiplying every frequency by u^{-1} mod p permutes the columns D, so the
    programme value is invariant under S -> uS; every orbit contains a set containing 1.
    Returns (list of supports (1-based), how) with how in {"exhaustive", "local"}.
    """
    nf = g.shape[0]
    if K == 1:
        return [[1]], "exhaustive"
    others = np.arange(1, nf)
    if comb(nf - 1, K - 1) <= exhaustive_limit:
        cands, scores = [], []
        it = itertools.combinations(others, K - 1)
        while True:
            block = list(itertools.islice(it, 100_000))
            if not block:
                break
            idx = np.array(block)
            m = (g[0][None, :] + g[idx].sum(1)).min(1)
            order = np.argsort(-m)[:shortlist]
            for j in order:
                cands.append([1] + [int(x) + 1 for x in idx[j]])
                scores.append(float(m[j]))
        o = np.argsort(-np.array(scores))[:shortlist]
        return [cands[i] for i in o], "exhaustive"
    rng = np.random.default_rng(seed)
    seen = {}
    for _ in range(restarts):
        S = [0] + rng.choice(others, K - 1, replace=False).tolist()
        cur = g[S].sum(0).min()
        improved = True
        while improved:
            improved = False
            for i in range(1, K):
                base = g[[x for j, x in enumerate(S) if j != i]].sum(0)
                cand = (base[None, :] + g).min(1)
                cand[S] = -1
                j = int(cand.argmax())
                if cand[j] > cur + 1e-12:
                    S[i] = j
                    cur = cand[j]
                    improved = True
        key = tuple(sorted(int(x) + 1 for x in S))
        seen[key] = float(cur)
    order = sorted(seen, key=lambda t: -seen[t])[:shortlist]
    return [list(t) for t in order], "local"


def best_K_support(g, K, s, shortlist=200, polish=True, seed=0, restarts=4):
    """Best K-support for the FREE-amplitude programme at exponent s.

    Two stages: (1) shortlist by equal-share coverage (exhaustive over all K-sets containing 1
    when C(nf-1, K-1) <= 3e6, else swap-local-search), (2) free-amplitude solve on each
    shortlisted support, (3) optional swap polish ON THE FREE VALUE from the stage-2 winner.
    Returns dict(value, S, b, how, polished).
    """
    nf = g.shape[0]
    cands, how = coverage_topK(g, K, shortlist=shortlist, seed=seed)
    best = (-1.0, None, None)
    for S in cands:
        v, b = solve_support(g, S, s, restarts=restarts, seed=seed)
        if v > best[0]:
            best = (v, S, b)
    polished = False
    if polish and K > 1:
        v, S, b = best
        improved = True
        while improved:
            improved = False
            for i in range(K):
                for knew in range(1, nf + 1):
                    if knew in S:
                        continue
                    S2 = list(S)
                    S2[i] = knew
                    v2, b2 = solve_support(g, sorted(S2), s, restarts=2, seed=seed)
                    if v2 > v + 1e-12:
                        v, S, b, improved, polished = v2, sorted(S2), b2, True, True
                        break
                if improved:
                    break
        best = (v, S, b)
    v, S, b = best
    return dict(value=v, S=list(map(int, S)), b=[float(x) for x in b], how=how, polished=polished)


# ----------------------------------------------------------------- junk-free blocks
def chain_block_phases(n):
    """The chain block (THEORY_nsum sec 8): N = n+1 neurons, operand phase pi*l/(n+1) on EVERY
    operand, readout phase n*pi*l/(n+1).  Returns (N, phi[l][j], psi[l]) in radians."""
    N = n + 1
    phi = [[math.pi * l / (n + 1)] * n for l in range(N)]
    psi = [n * math.pi * l / (n + 1) for l in range(N)]
    return N, phi, psi


def xn_terms(n):
    """Every Fourier term of one x^n neuron paired with its readout, as integer coefficient rows.

    Neuron l has operand phases phi_{l,j} and readout phase psi_l.  Expanding
    h = (sum_j cos(theta_j + phi_j))^n  and multiplying by cos(theta_c + psi) gives terms indexed by
      m  = multi-exponent, sum_j m_j = n            (which monomial of the n-th power)
      c_j in {-m_j, -m_j+2, ..., m_j}               (which harmonic of operand j)
      delta = +-1                                   (readout sign)
    whose l-dependent phase is exp(i[ sum_j c_j phi_{l,j} + delta psi_l ]).
    Returns (rows, tags): rows[t] = (c_1..c_n, delta) for every NON-clean term (the clean term is
    m = (1..1), c = (1..1), delta = -1 and its conjugate)."""
    rows, tags = [], []
    for m in itertools.product(range(n + 1), repeat=n):
        if sum(m) != n:
            continue
        ranges = [range(-mj, mj + 1, 2) if mj > 0 else [0] for mj in m]
        for c in itertools.product(*ranges):
            for delta in (+1, -1):
                clean = (list(m) == [1] * n and (list(c) == [1] * n and delta == -1
                                                 or list(c) == [-1] * n and delta == +1))
                if not clean:
                    rows.append(list(c) + [delta])
                    tags.append((m, c, delta))
    return np.array(rows, dtype=np.int64), tags


def find_xn_block(n, Nmax=60):
    """Smallest cyclic junk-free x^n block: N neurons l = 0..N-1 with operand phases
    2 pi l alpha_j / N and readout phase 2 pi l (sum_j alpha_j) / N.  The population kills a term
    iff its l-phase exponent u = sum_j c_j alpha_j + delta * sum_j alpha_j  is != 0 mod N.
    Returns (N, alpha) for the smallest N (ties: lexicographically smallest alpha), else (None, None)."""
    rows, _ = xn_terms(n)
    C = rows[:, :n] + rows[:, n:n + 1]        # u = (c_j + delta) . alpha
    C = np.unique(C, axis=0)
    for N in range(n + 1, Nmax + 1):
        alphas = np.array(list(itertools.combinations_with_replacement(range(1, N), n)), dtype=np.int64)
        U = (alphas @ C.T) % N                 # (n_alpha, n_terms)
        ok = (U != 0).all(1)
        if ok.any():
            return N, alphas[np.argmax(ok)].tolist()
    return None, None


def product_block(n):
    """Fallback exactly-junk-free x^n population: the full product grid l in Z_N^n, N = n+2,
    operand phase 2 pi l_j / N on operand j only, readout phase 2 pi (sum_j l_j) / N.
    Summing over each l_j independently kills every term unless c_j + delta = 0 mod N for all j,
    which (N >= n+2 > max|c_j - 1|) leaves only the clean term and its conjugate.  N^n neurons."""
    N = n + 2
    phi, psi = [], []
    for l in itertools.product(range(N), repeat=n):
        phi.append([2 * math.pi * lj / N for lj in l])
        psi.append(2 * math.pi * sum(l) / N)
    return N ** n, phi, psi


XN_BLOCK_CACHE = {}


def xn_block_phases(n):
    """x^n block: N neurons, operand phases 2*pi*l*alpha_j/N, readout phase 2*pi*l*(sum alpha)/N."""
    if n not in XN_BLOCK_CACHE:
        N, alpha = find_xn_block(n)
        if N is None:
            XN_BLOCK_CACHE[n] = ("product",) + product_block(n)
        else:
            psi = sum(alpha)
            XN_BLOCK_CACHE[n] = ("cyclic", N,
                                 [[2 * math.pi * l * a / N for a in alpha] for l in range(N)],
                                 [2 * math.pi * l * psi / N for l in range(N)])
    kind, N, phi, ps = XN_BLOCK_CACHE[n]
    return N, phi, ps


def block_spec(family, n):
    """(N, phi, psi, mu) for the family.  mu = clean-coefficient multiplier (1 chain, n! for x^n)."""
    if family == "chain":
        N, phi, psi = chain_block_phases(n)
        return N, phi, psi, 1.0
    N, phi, psi = xn_block_phases(n)
    return N, phi, psi, float(math.factorial(n))


def build_net(family, p, n, S, b, q):
    """The SHARE-WEIGHTED block net as explicit tensors.  E: (n, m, p), W: (p, m), m = |S|*N.

    Frequency k's N neurons all have radius r_k with r_k^q proportional to the share b_k, which
    is exactly what makes b the l_q share:  ||theta_i||^2 = (n+1) r_k^2 p/2 for every neuron of
    frequency k, so the frequency's slice of P_q is N ((n+1)p/2)^{q/2} r_k^q  ∝  b_k.
    """
    N, phi, psi, _ = block_spec(family, n)
    a = np.arange(p) * 2 * np.pi / p
    m = len(S) * N
    E = np.zeros((n, m, p))
    W = np.zeros((p, m))
    i = 0
    for k, bk in zip(S, b):
        r = float(bk) ** (1.0 / q)
        for l in range(N):
            for j in range(n):
                E[j, i] = r * np.cos(k * a + phi[l][j])
            W[:, i] = r * np.cos(k * a + psi[l])
            i += 1
    return E, W


def clean_spectrum(family, p, n, S, b, q):
    """Amplitudes A_k of the block's logit table  z(row, c) = sum_k A_k cos(2 pi k D / p),
    D = (sum_j a_j) - c, together with the penalty P_q.  Exact, from the construction."""
    N, _, _, mu = block_spec(family, n)
    nu = n + 1
    r = np.asarray(b, float) ** (1.0 / q)                  # radius per frequency
    A = N * mu * 2.0 ** (-n) * r ** nu                      # clean amplitude of frequency k
    Pq = N * ((nu * p / 2.0) ** (q / 2.0)) * (r ** q).sum()  # sum_i ||theta_i||^q
    return A, float(Pq)


def gaps_from_spectrum(p, S, A):
    """gap(D) = z(c) - z(y) for the wrong class at distance D = 1..p-1 (<= 0 for a correct net).
    z depends only on D, so this p-1 vector IS the whole table."""
    D = np.arange(1, p)
    g = np.zeros(p - 1)
    for k, Ak in zip(S, A):
        g -= Ak * (1 - np.cos(2 * np.pi * k * D / p))
    return g


def margin_of_gaps(gp):
    return float(-gp.max())


def ce_of_gaps(gp, t, nu):
    """CE when every weight is multiplied by e^t: logits scale by e^{nu t}.  Same for every row."""
    d = gp * math.exp(nu * t)
    mx = d.max()
    return float(np.log1p(math.exp(mx) * np.exp(d - mx).sum()))   # log(1 + sum_D e^{d_D})


def obj_star_gaps(gp, Pq, lam, nu, q, lo=-8.0, hi=8.0, iters=200):
    """min over the global scale e^t of  CE + lam * e^{q t} * P_q.  Golden section (the function
    is unimodal in t: CE is decreasing convex-ish, the penalty increasing convex)."""
    f = lambda t: ce_of_gaps(gp, t, nu) + lam * math.exp(q * t) * Pq
    gr = (math.sqrt(5) - 1) / 2
    c, d = hi - gr * (hi - lo), lo + gr * (hi - lo)
    fc, fd = f(c), f(d)
    for _ in range(iters):
        if fc < fd:
            hi, d, fd = d, c, fc
            c = hi - gr * (hi - lo)
            fc = f(c)
        else:
            lo, c, fc = c, d, fd
            d = lo + gr * (hi - lo)
            fd = f(d)
    t = (lo + hi) / 2
    ce = ce_of_gaps(gp, t, nu)
    pen = lam * math.exp(q * t) * Pq
    return dict(obj=ce + pen, ce=ce, pen=pen, scale=math.exp(t))


def B_const(p, n):
    nu = n + 1
    return 2.0 * (2.0 * p * (n + 1)) ** (-nu / 2.0)


def gamma_con(family, p, n, S, b, q):
    """Exact normalized margin of the share-weighted block, computed from the clean spectrum."""
    nu = n + 1
    A, Pq = clean_spectrum(family, p, n, S, b, q)
    gp = gaps_from_spectrum(p, S, A)
    mg = margin_of_gaps(gp)
    return mg / Pq ** (nu / q), mg, Pq, A, gp


def gamma_con_predicted(family, p, n, s_val, q):
    """mu * N^{-(nu-q)/q} * B * val(S,b) -- the identity the evaluator must reproduce."""
    nu = n + 1
    N, _, _, mu = block_spec(family, n)
    return mu * N ** (-(nu - q) / q) * B_const(p, n) * s_val


# ----------------------------------------------------------------- brute-force verification
def brute_table(family, E, W, p, n):
    """Exact p^n x p logit table by explicit enumeration (small p only)."""
    rows = p ** n
    idx = np.stack([(np.arange(rows) // p ** j) % p for j in range(n)], 1)
    H = np.ones((rows, E.shape[1])) if family == "chain" else None
    if family == "chain":
        for j in range(n):
            H = H * E[j][:, idx[:, j]].T
    else:
        x = np.zeros((rows, E.shape[1]))
        for j in range(n):
            x = x + E[j][:, idx[:, j]].T
        H = x ** n
    z = H @ W.T
    y = idx.sum(1) % p
    return z, y, idx


def verify_block(family, p, n, S, b, q, verbose=True):
    """Check (a) EXACT junk-freeness: the logit table depends only on D = sum a - c;
    (b) the identity gamma_q = mu N^{-(nu-q)/q} B val(S,b);  (c) the margin from the p-1 gap
    vector equals the brute-force margin over the full table."""
    nu = n + 1
    E, W = build_net(family, p, n, S, b, q)
    z, y, idx = brute_table(family, E, W, p, n)
    s_pn = np.add.reduce(idx, 1) % p
    D = (s_pn[:, None] - np.arange(p)[None, :]) % p
    A, Pq = clean_spectrum(family, p, n, S, b, q)
    zz = np.zeros_like(z)
    for k, Ak in zip(S, A):
        zz += Ak * np.cos(2 * np.pi * k * D / p)
    junk = float(np.abs(z - zz).max()) / max(float(np.abs(z).max()), 1e-300)
    n2 = (E ** 2).sum((0, 2)) + (W ** 2).sum(0)
    Pq_b = float((n2 ** (q / 2)).sum())
    zy = z[np.arange(len(y)), y]
    z2 = z.copy()
    z2[np.arange(len(y)), y] = -np.inf
    mg_b = float((zy - z2.max(1)).min())
    gam_b = mg_b / Pq_b ** (nu / q)
    g = gmat(p)
    v = val_of(g, S, b, nu / q)
    gam_p = gamma_con_predicted(family, p, n, v, q)
    gam_c, mg_c, Pq_c, _, _ = gamma_con(family, p, n, S, b, q)
    out = dict(junk_rel=junk, Pq_rel=abs(Pq_b / Pq - 1), margin_rel=abs(mg_b / mg_c - 1),
               gamma_brute=gam_b, gamma_spec=gam_c, gamma_pred=gam_p,
               id_rel=abs(gam_c / gam_p - 1), brute_rel=abs(gam_b / gam_p - 1))
    if verbose:
        print(f"  verify {family} p{p} n{n} q{q:g} S={S}: junk {junk:.3e}  P_q {out['Pq_rel']:.3e}  "
              f"margin {out['margin_rel']:.3e}  identity {out['id_rel']:.3e}  brute {out['brute_rel']:.3e}")
    return out


if __name__ == "__main__":
    print("x^n cyclic junk-free block search (N, alpha):")
    for n in (2, 3, 4):
        print("  n =", n, find_xn_block(n), flush=True)
    print("blocks used:")
    for n in (2, 3, 4):
        N, phi, psi, mu = block_spec("xn", n)
        print(f"  x^{n}: N = {N} neurons/frequency, mu = {mu:g}, kind = {XN_BLOCK_CACHE[n][0]}")
        N2, _, _, mu2 = block_spec("chain", n)
        print(f"  chain n={n}: N = {N2}, mu = {mu2:g}")
