"""verify_eq_block.py -- INDEPENDENT referee check of the EQUALITY CASE of target (T).

Written from scratch: imports nothing from the authors' modules (fvf_obj, kstar, lemma_*, gopt_*, adv_*).

Setting (every symbol defined).
  p    odd prime, the modulus.
  n    number of operands.  nu = n + 1 = homogeneity degree of the net.
  rows  a = (a_1..a_n) in Z_p^n, all p^n of them; label y(a) = (a_1 + ... + a_n) mod p.
  net  ("gated chain")  z(a, c) = sum_i W[c, i] * prod_{j=1..n} E[j, i, a_j].
       E has shape (n, m, p) (one embedding matrix per operand), W has shape (p, m).
  margin = min over all p^n rows a and all wrong classes c != y(a) of  z(a, y(a)) - z(a, c).
  ||theta_i||^2 = sum_j ||E[j,i,:]||^2 + ||W[:,i]||^2   (all of neuron i's weights).
  P_q = sum_i ||theta_i||^q ;  gamma_q = margin / P_q^{nu/q}   (scale-invariant).
  B = 2 * (2 p (n+1))^{-nu/2}.
  g_k(D) = 1 - cos(2 pi k D / p).
  Gamma_s(p) = max over shares b (b_k >= 0, sum_k b_k = 1, k = 1..(p-1)/2)
                 of  min_{D != 0} sum_k b_k^s g_k(D).            s = nu/q.
  TARGET (T):   gamma_q <= (n+1)^{-(nu-q)/q} * B * Gamma_{nu/q}(p),
  claimed equality at the SHARE-WEIGHTED (n+1)-neuron common-phase block.

Block (THEORY_nsum §8 + ADVERSARY's share weighting).  For each frequency k in the support S,
n+1 neurons l = 0..n with amplitude r_k = b_k^{1/q}:
    E[j, (k,l), a] = r_k * cos(2 pi k a / p + pi l /(n+1))   for every operand j,
    W[c,   (k,l)]  = r_k * cos(2 pi k c / p + n * pi l /(n+1)).

Usage:  uv run python heracles/verify_eq_block.py [--quick]
"""
import argparse, itertools, json, math, time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

OUT = Path(__file__).resolve().parent / "FINAL_RUNS" / "VERIFY"
rng = np.random.default_rng(20260921)


# ---------------------------------------------------------------- the programme Gamma_s(p)
def gmat(p):
    """G[k-1, D-1] = 1 - cos(2 pi k D / p),  k = 1..nf,  D = 1..p-1.  nf = (p-1)/2."""
    nf = (p - 1) // 2
    k = np.arange(1, nf + 1)[:, None]
    D = np.arange(1, p)[None, :]
    return 1.0 - np.cos(2.0 * np.pi * k * D / p)


def maxmin_on_support(G, S, s, starts=40, seed=0):
    """max over shares b on the support S (a list of frequency indices, 1-based) of min_D sum b_k^s G[k,D].

    Epigraph SLSQP:  variables (b_1..b_K, t); maximise t s.t. sum_k b_k^s G[k,D] >= t for all D,
    sum_k b_k = 1, b >= 0.  Many Dirichlet restarts.  Returns (value, b)."""
    K = len(S)
    Gs = G[[k - 1 for k in S], :]          # (K, p-1)
    r = np.random.default_rng(seed)

    def unpack(x):
        return x[:K], x[K]

    def negt(x):
        return -x[K]

    def negt_grad(x):
        g = np.zeros(K + 1); g[K] = -1.0; return g

    cons = [
        {"type": "eq", "fun": lambda x: x[:K].sum() - 1.0,
         "jac": lambda x: np.concatenate([np.ones(K), [0.0]])},
        {"type": "ineq",
         "fun": lambda x: (x[:K] ** s) @ Gs - x[K],
         "jac": lambda x: np.concatenate(
             [(s * np.maximum(x[:K], 1e-300) ** (s - 1))[None, :] * Gs.T,
              -np.ones((Gs.shape[1], 1))], axis=1)},
    ]
    bnds = [(0.0, 1.0)] * K + [(0.0, None)]
    best_v, best_b = -np.inf, None
    for t in range(starts):
        b0 = np.ones(K) / K if t == 0 else r.dirichlet(np.ones(K) * (0.4 if t % 2 else 3.0))
        x0 = np.concatenate([b0, [float(((b0 ** s) @ Gs).min())]])
        try:
            res = minimize(negt, x0, jac=negt_grad, bounds=bnds, constraints=cons,
                           method="SLSQP", options=dict(maxiter=400, ftol=1e-14))
        except Exception:
            continue
        b = np.clip(res.x[:K], 0.0, None)
        if b.sum() <= 0:
            continue
        b = b / b.sum()
        v = float(((b ** s) @ Gs).min())
        if v > best_v:
            best_v, best_b = v, b
    return best_v, best_b


def canon_support(S, p):
    """Canonical form of a support under the multiplicative symmetry k -> c*k (which permutes D too)."""
    nf = (p - 1) // 2
    best = None
    for c in range(1, p):
        T = tuple(sorted(min((c * k) % p, p - (c * k) % p) for k in S))
        if best is None or T < best:
            best = T
    return best


def gamma_s(p, s, kmax=None, starts=40, verbose=False):
    """Global Gamma_s(p) by EXHAUSTIVE enumeration of supports (up to multiplicative symmetry)
    with a multistart epigraph solve on each.  Returns dict with value, support, shares."""
    nf = (p - 1) // 2
    G = gmat(p)
    kmax = nf if kmax is None else min(kmax, nf)
    seen, best = set(), dict(value=-np.inf, S=None, b=None)
    for K in range(1, kmax + 1):
        for S in itertools.combinations(range(1, nf + 1), K):
            c = canon_support(S, p)
            if c in seen:
                continue
            seen.add(c)
            v, b = maxmin_on_support(G, list(S), s, starts=starts, seed=hash(c) % (2 ** 31))
            if v > best["value"]:
                best = dict(value=float(v), S=list(S), b=b.tolist())
                if verbose:
                    print(f"    new best K={K} S={list(S)} v={v:.12f} b={np.round(b,6).tolist()}")
    return best


# ---------------------------------------------------------------- the block, exactly
def build_block(p, n, S, b, q, equal_amplitude=False):
    """Share-weighted (n+1)-neuron common-phase block.  Returns E (n, m, p), W (p, m)."""
    K = len(S)
    m = K * (n + 1)
    E = np.zeros((n, m, p)); W = np.zeros((p, m))
    a = 2.0 * np.pi * np.arange(p) / p
    i = 0
    for t, k in enumerate(S):
        rk = 1.0 if equal_amplitude else float(b[t]) ** (1.0 / q)
        for l in range(n + 1):
            ph = np.pi * l / (n + 1)
            for j in range(n):
                E[j, i] = rk * np.cos(k * a + ph)
            W[:, i] = rk * np.cos(k * a + n * ph)
            i += 1
    return E, W


def logit_table(E, W, p, n):
    """z[row, c] over ALL p^n rows.  row index = a_1 + p a_2 + ... (base-p).  Brute force."""
    m = E.shape[1]
    idx = np.arange(p ** n)
    ops = np.stack([(idx // p ** j) % p for j in range(n)], axis=1)      # (p^n, n)
    h = np.ones((p ** n, m))
    for j in range(n):
        h = h * E[j][:, ops[:, j]].T                                     # (p^n, m)
    z = h @ W.T                                                          # (p^n, p)
    y = ops.sum(axis=1) % p
    return z, y, ops


def exact_margin(z, y):
    """min over rows and wrong classes of z[row, y] - z[row, c].  Pure brute force, no shortcuts."""
    worst = np.inf
    p = z.shape[1]
    for i in range(z.shape[0]):
        zy = z[i, y[i]]
        for c in range(p):
            if c == y[i]:
                continue
            d = zy - z[i, c]
            if d < worst:
                worst = d
    return float(worst)


def Pq_of(E, W, q):
    nrm2 = (E ** 2).sum(axis=(0, 2)) + (W ** 2).sum(axis=0)              # ||theta_i||^2 per neuron
    return float((nrm2 ** (q / 2)).sum())


def junk_spectrum(z, ops, y, p, n):
    """Full (n+1)-dim DFT of the logit table z(a_1..a_n, c).  Clean modes are (k,..,k,-k).
    Returns (max |clean|, max |junk|)."""
    T = np.zeros((p,) * (n + 1))
    for i in range(p ** n):
        T[tuple(ops[i]) ] = 0  # placeholder; filled below
    T = z.reshape((p,) * n + (p,))          # index order (a_1, a_2, ..., a_n, c) since ops[:,j] = (idx//p^j)%p
    # careful: idx = a_1 + p a_2 + ...; reshape of a length-p^n axis in C order gives (a_n, ..., a_1).
    # Build explicitly instead, to avoid any ordering doubt:
    T = np.zeros((p,) * (n + 1))
    for i in range(p ** n):
        T[tuple(ops[i])] = z[i]
    F = np.fft.fftn(T) / (p ** (n + 1))
    clean, junk = 0.0, 0.0
    it = np.ndindex(*([p] * (n + 1)))
    for mm in it:
        v = abs(F[mm])
        k = mm[0]
        is_clean = all(mm[j] == k for j in range(n)) and mm[n] == (-k) % p
        if is_clean:
            clean = max(clean, v)
        else:
            junk = max(junk, v)
    return clean, junk


def target_rhs(p, n, q, Gam):
    nu = n + 1
    B = 2.0 * (2.0 * p * (n + 1)) ** (-nu / 2.0)
    return (n + 1) ** (-(nu - q) / q) * B * Gam


# ---------------------------------------------------------------- main
def cell(p, n, q, starts=40, kmax=None, label="", exact_margin_brute=True):
    nu = n + 1
    s = nu / q
    t0 = time.time()
    prog = gamma_s(p, s, kmax=kmax, starts=starts)
    Gam, S, b = prog["value"], prog["S"], np.array(prog["b"])
    E, W = build_block(p, n, S, b, q)
    z, y, ops = logit_table(E, W, p, n)
    mg = exact_margin(z, y) if exact_margin_brute else float(
        np.min(np.where(np.eye(p)[y] > 0, np.inf, z[np.arange(len(y)), y][:, None] - z)))
    Pq = Pq_of(E, W, q)
    gam = mg / Pq ** (nu / q)
    rhs = target_rhs(p, n, q, Gam)
    clean, junk = junk_spectrum(z, ops, y, p, n)
    # equal-amplitude block on the SAME support
    E2, W2 = build_block(p, n, S, b, q, equal_amplitude=True)
    z2, y2, _ = logit_table(E2, W2, p, n)
    mg2 = exact_margin(z2, y2)
    gam2 = mg2 / Pq_of(E2, W2, q) ** (nu / q)
    row = dict(label=label, p=p, n=n, nu=nu, q=q, s=s, Gamma=Gam, S=S, b=b.tolist(),
               b_spread=float(b.max() - b.min()), margin=mg, Pq=Pq, gamma=gam, rhs=rhs,
               ratio=gam / rhs, junk_max=junk, clean_max=clean,
               gamma_equal_amp=gam2, ratio_equal_amp=gam2 / rhs, secs=time.time() - t0)
    return row


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    cells = [
        dict(p=11, n=2, q=2.0, starts=60, kmax=None, label="n2 p11 q2"),
        dict(p=13, n=2, q=2.0, starts=60, kmax=None, label="n2 p13 q2"),
        dict(p=23, n=2, q=2.0, starts=40, kmax=7,    label="n2 p23 q2"),
        dict(p=11, n=3, q=2.0, starts=60, kmax=None, label="n3 p11 q2"),
    ]
    if a.quick:
        cells = cells[:1]
    for c in cells:
        print(f"== {c['label']}", flush=True)
        r = cell(**c)
        rows.append(r)
        print(f"   Gamma_{r['s']:g}({r['p']}) = {r['Gamma']:.12f}  support {r['S']}")
        print(f"   optimal shares b = {[round(x,9) for x in r['b']]}   (spread {r['b_spread']:.3e})")
        print(f"   block: margin {r['margin']:.12g}  P_q {r['Pq']:.12g}  gamma {r['gamma']:.12e}")
        print(f"   RHS of (T)      {r['rhs']:.12e}")
        print(f"   RATIO gamma/RHS = {r['ratio']:.15f}")
        print(f"   junk: max |junk Fourier coeff| {r['junk_max']:.3e}   (max clean {r['clean_max']:.6e})")
        print(f"   equal-amplitude block on same support: ratio {r['ratio_equal_amp']:.12f}")
        print(f"   [{r['secs']:.1f}s]", flush=True)
    (OUT / "eq_block.json").write_text(json.dumps(rows, indent=1))
    print("\nwrote", OUT / "eq_block.json")
