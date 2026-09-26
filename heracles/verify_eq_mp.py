"""verify_eq_mp.py -- the equality case of (T) at 50-digit precision, end to end, in mpmath.

Everything (block weights, the full p^n x p logit table, the brute-force margin over every row and
every wrong class, P_q, gamma_q, Gamma_s from the KKT system, and the RHS of (T)) is recomputed in
mpmath at dps = 50, so the printed ratio gamma_q / RHS is not a float64 artefact.

Also prints the referee's own ALGEBRAIC derivation of the ratio, which says it must be exactly 1
for every (p, n, q, support S, share vector b):

  one frequency-k neuron of the block has all n+1 of its vectors equal to r_k cos(k. + phase),
  each of squared norm r_k^2 p/2, so ||theta||^2 = (n+1) r_k^2 p / 2 and, with r_k = b_k^{1/q},
      P_q = sum_{k in S} (n+1) * ((n+1) p / 2)^{q/2} * b_k  =  (n+1) ((n+1) p/2)^{q/2}.
  The block's logit table is exactly (junk-free lemma, THEORY_nsum §8)
      z(a,c) = sum_k (n+1) 2^{-n} r_k^{n+1} cos(2 pi k (s(a) - c)/p),   s(a) = sum_j a_j,
  so  margin = min_{D != 0} (n+1) 2^{-n} sum_k b_k^{nu/q} (1 - cos(2 pi k D/p))
             = (n+1) 2^{-n} * val(S, b),  and at the optimal (S,b) that is (n+1) 2^{-n} Gamma_{nu/q}.
  Hence gamma_q = 2^{-n} (n+1)^{1 - nu/q} ((n+1)p/2)^{-nu/2} Gamma, while
        RHS     = (n+1)^{1 - nu/q} * 2 * (2 p (n+1))^{-nu/2} Gamma,
  and the two prefactors differ by 2^{-n} 2^{nu/2} / (2 . 2^{-nu/2}) = 2^{nu - n - 1} = 2^0 = 1.

Usage: uv run python heracles/verify_eq_mp.py
"""
import json
from pathlib import Path

import mpmath as mp

OUT = Path(__file__).resolve().parent / "FINAL_RUNS" / "VERIFY"
DPS = 50


def kkt_gamma(p, s, S, A, b0):
    """High-precision Gamma_s value and shares on support S with active offsets A (each once)."""
    s = mp.mpf(s)
    K, m = len(S), len(A)
    g = [[1 - mp.cos(2 * mp.pi * k * D / p) for D in A] for k in S]

    def F(*x):
        b = list(x[:K]); mu = list(x[K:K + m]); lam = x[K + m]; t = x[K + m + 1]
        e = [s * b[i] ** (s - 1) * sum(mu[d] * g[i][d] for d in range(m)) - lam for i in range(K)]
        e += [sum(b[i] ** s * g[i][d] for i in range(K)) - t for d in range(m)]
        e += [sum(b) - 1, sum(mu) - 1]
        return e

    t0 = min(sum(mp.mpf(b0[i]) ** s * g[i][d] for i in range(K)) for d in range(m))
    x0 = [mp.mpf(v) for v in b0] + [mp.mpf(1) / m] * m + [mp.mpf(1), t0]
    sol = mp.findroot(F, x0, solver="mdnewton", tol=mp.mpf(10) ** -40)
    b = [sol[i] for i in range(K)]
    t = sol[K + m + 1]
    # independent re-evaluation of the inner min over ALL D != 0 (not just the active ones)
    full = min(sum(b[i] ** s * (1 - mp.cos(2 * mp.pi * S[i] * D / p)) for i in range(K))
               for D in range(1, p))
    return t, b, full


def block_mp(p, n, S, b, q):
    """(n+1)-neuron common-phase block, share-weighted, in mpmath.  E[j][i][a], W[c][i]."""
    m = len(S) * (n + 1)
    E = [[[mp.mpf(0)] * p for _ in range(m)] for _ in range(n)]
    W = [[mp.mpf(0)] * m for _ in range(p)]
    i = 0
    for t, k in enumerate(S):
        rk = b[t] ** (mp.mpf(1) / q)
        for l in range(n + 1):
            ph = mp.pi * l / (n + 1)
            for a in range(p):
                v = rk * mp.cos(2 * mp.pi * k * a / p + ph)
                for j in range(n):
                    E[j][i][a] = v
            for c in range(p):
                W[c][i] = rk * mp.cos(2 * mp.pi * k * c / p + n * ph)
            i += 1
    return E, W


def margin_mp(E, W, p, n):
    """Brute force: min over all p^n rows and all p-1 wrong classes of z(a,y) - z(a,c)."""
    m = len(W[0])
    worst = None
    for idx in range(p ** n):
        a = [(idx // p ** j) % p for j in range(n)]
        y = sum(a) % p
        h = [mp.mpf(1)] * m
        for j in range(n):
            for i in range(m):
                h[i] *= E[j][i][a[j]]
        z = [sum(W[c][i] * h[i] for i in range(m)) for c in range(p)]
        for c in range(p):
            if c == y:
                continue
            d = z[y] - z[c]
            if worst is None or d < worst:
                worst = d
    return worst


def Pq_mp(E, W, p, n, q):
    m = len(W[0])
    tot = mp.mpf(0)
    for i in range(m):
        s2 = mp.mpf(0)
        for j in range(n):
            s2 += sum(E[j][i][a] ** 2 for a in range(p))
        s2 += sum(W[c][i] ** 2 for c in range(p))
        tot += s2 ** (mp.mpf(q) / 2)
    return tot


CELLS = [
    # p, n, q, support, active offsets (each once), float64 starting shares
    dict(p=11, n=2, q=2.0, S=[1, 3], A=[1, 4],
         b0=["0.43054540683", "0.56945459317"]),
    dict(p=13, n=2, q=2.0, S=[1, 2, 5], A=[1, 2, 5],
         b0=["0.332437246572", "0.287857825801", "0.379704927627"]),
    dict(p=23, n=2, q=2.0, S=[1, 2, 7, 10], A=[2, 3, 7, 10],
         b0=["0.151865737065", "0.285662736317", "0.280975936694", "0.281495589923"]),
    dict(p=11, n=3, q=2.0, S=[1, 3], A=[1, 4],
         b0=["0.447761401537", "0.552238598463"]),
]

if __name__ == "__main__":
    mp.mp.dps = DPS
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for C in CELLS:
        p, n, q, S, A, b0 = C["p"], C["n"], C["q"], C["S"], C["A"], C["b0"]
        nu = n + 1; s = mp.mpf(nu) / q
        Gam, b, full = kkt_gamma(p, s, S, A, b0)
        E, W = block_mp(p, n, S, b, q)
        mg = margin_mp(E, W, p, n)
        Pq = Pq_mp(E, W, p, n, q)
        gam = mg / Pq ** (mp.mpf(nu) / q)
        B = 2 * (2 * mp.mpf(p) * (n + 1)) ** (-mp.mpf(nu) / 2)
        rhs = mp.mpf(n + 1) ** (-(mp.mpf(nu) - q) / q) * B * Gam
        ratio = gam / rhs
        print(f"== p={p} n={n} q={q:g}  nu={nu}  s={mp.nstr(s,6)}  support {S}")
        print(f"   Gamma_s (KKT, {DPS} dps)        {mp.nstr(Gam, 36)}")
        print(f"   inner min over ALL D re-check   {mp.nstr(full, 36)}   (diff {mp.nstr(abs(full-Gam),6)})")
        print(f"   shares b                        {[mp.nstr(x, 28) for x in b]}")
        print(f"   margin (brute, all p^n x (p-1)) {mp.nstr(mg, 36)}")
        print(f"   P_q                             {mp.nstr(Pq, 36)}")
        print(f"   gamma_q                         {mp.nstr(gam, 36)}")
        print(f"   RHS of (T)                      {mp.nstr(rhs, 36)}")
        print(f"   RATIO gamma_q / RHS             {mp.nstr(ratio, 40)}")
        print(f"   |RATIO - 1|                     {mp.nstr(abs(ratio-1), 6)}", flush=True)
        rows.append(dict(p=p, n=n, q=q, S=S, Gamma=mp.nstr(Gam, 36), b=[mp.nstr(x, 28) for x in b],
                         margin=mp.nstr(mg, 36), Pq=mp.nstr(Pq, 36), gamma=mp.nstr(gam, 36),
                         rhs=mp.nstr(rhs, 36), ratio=mp.nstr(ratio, 40),
                         abs_ratio_minus_1=mp.nstr(abs(ratio - 1), 6)))
    (OUT / "eq_mp.json").write_text(json.dumps(rows, indent=1))
    print("\nwrote", OUT / "eq_mp.json")
