"""P2B step 7: evaluation of the chain-frame route (ELEGANT.md) for the JUNK problem.

ELEGANT's Lemma 1: with the chain vector m_l = (1, v_l1, v_l1 v_l2, ..., prod_j v_lj) in C^D
(D = n+1, ||m_l||^2 = D) and M := sum_l c_l m_l m_l^*,

    M_{k,k}  = A                                   (the clean equation)
    M_{k,k'} = C_{ {k'+1..k} }   for k > k'        (the INTERVAL junk totals)
    M_{k,k'} = mirror moments    for k < k'        (never controlled by (KL))

so with junk,  M/A = I + L' + U',  L' strictly lower (= interval junk / A),
U' strictly upper (mirror moments / A).

This script checks the two structural facts the coordinator's relay proposes, and the one
that decides whether the route gives a QUADRATIC bound:

 (E1)  E_1(M/A) = D exactly (both L' and U' have zero diagonal)  -- no junk loss at all;
 (E2)  E_2(M/A) = binom(D,2) - tr(L' U')  EXACTLY -- the correction is BILINEAR, not linear;
 (Ek)  for every k, E_k(I + L' + U') - binom(D,k) is a sum of terms each carrying at least
       one L' factor AND at least one U' factor (a permutation that uses a strictly-lower
       entry must also use a strictly-upper one to return);
 (U)   is ||U'|| small when the junk is small?  ELEGANT's own n = 1 witness (v = (1, i),
       exactly junk-free) has ||U'|| = sqrt2 -- so NO: ||U'|| is an O(1) function of the
       phases even at eta = 0, and the bilinear correction ||L'|| ||U'|| is then LINEAR in
       the junk.  The route therefore needs its own coupling lemma
       "||L'|| ||U'|| <= (Gram-determinant credit) + C eta^2",
       structurally the same statement P2B proves for the kill-point route at n = 2.
"""
import itertools, json, math, os
import numpy as np

OUT = "FINAL_RUNS/P2B"


def chain_M(c, v):
    N, n = v.shape
    D = n + 1
    m = np.ones((N, D), dtype=complex)
    for k in range(1, D):
        m[:, k] = m[:, k - 1] * v[:, k - 1]
    return sum(c[l] * np.outer(m[l], np.conj(m[l])) for l in range(N)), m


def Ek(X, k):
    """k-th elementary symmetric function of the eigenvalues = sum of principal k-minors."""
    D = X.shape[0]
    return sum(np.linalg.det(X[np.ix_(L, L)]) for L in itertools.combinations(range(D), k))


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(99)
    res = {}

    print("== (E1)/(E2): exact structure of the junk corrections ==", flush=True)
    res["struct"] = {}
    for n in (1, 2, 3, 4):
        D = n + 1
        w1 = w2 = 0.0
        for _ in range(400):
            N = D + rng.integers(0, 3)
            v = np.exp(2j * np.pi * rng.random((N, n)))
            c = rng.normal(size=N) + 1j * rng.normal(size=N)
            c = c / c.sum()                 # A = C_empty = 1
            M, _ = chain_M(c, v)
            Lp = np.tril(M, -1)
            Up = np.triu(M, 1)
            w1 = max(w1, abs(Ek(M, 1) - D))
            if D >= 2:
                w2 = max(w2, abs(Ek(M, 2) - (D * (D - 1) / 2 - np.trace(Lp @ Up))))
        res["struct"][n] = {"E1_err": w1, "E2_err": w2}
        print(f"   n={n}: max|E_1(M) - D| = {w1:.3e};  "
              f"max|E_2(M) - [binom(D,2) - tr(L'U')]| = {w2:.3e}", flush=True)

    print("\n== (Ek): every correction is at least bilinear (scale L' -> t L') ==", flush=True)
    res["bilinear"] = {}
    for n in (2, 3):
        D = n + 1
        worst = 0.0
        for _ in range(200):
            N = D
            v = np.exp(2j * np.pi * rng.random((N, n)))
            c = rng.normal(size=N) + 1j * rng.normal(size=N)
            c = c / c.sum()
            M, _ = chain_M(c, v)
            Lp, Up = np.tril(M, -1), np.triu(M, 1)
            for k in range(1, D + 1):
                base = Ek(np.eye(D) + Up, k)
                ref = float(math.comb(D, k))
                worst = max(worst, abs(base - ref))
        res["bilinear"][n] = worst
        print(f"   n={n}: max |E_k(I + U') - binom(D,k)| over k = {worst:.3e}  "
              f"(so every correction carries an L' factor)", flush=True)

    print("\n== (U): is ||U'|| small when the junk is small? ==", flush=True)
    # ELEGANT's exact witness at n = 1
    v = np.array([[1.0 + 0j], [1j]])
    c = np.array([(1 - 1j) / 2, (1 + 1j) / 2])
    M, _ = chain_M(c, v)
    CT = np.array([c.sum(), np.sum(c * v[:, 0])])
    print(f"   n=1 witness v=(1,i): C_empty={CT[0]:.6f}, junk C_{{1}}={CT[1]:.3e}, "
          f"||U'||={np.linalg.norm(np.triu(M,1),2):.6f}", flush=True)
    res["witness_n1"] = {"junk": float(abs(CT[1])), "Unorm": float(np.linalg.norm(np.triu(M, 1), 2))}

    # n = 2, exactly junk-free three-atom populations (common phase family): ||U'|| vs phases
    rows = []
    for th in np.linspace(0.05, np.pi - 0.05, 60):
        x = np.array([np.exp(1j * th), np.exp(-1j * th), 1.0 + 0j])
        v = np.stack([x, x], axis=1)
        Mm = np.stack([np.ones(3, dtype=complex), x, x * x], axis=0)  # rows: C_empty, C_1, C_2
        full = np.stack([np.ones(3, dtype=complex), x, x, x * x], axis=0)
        # junk-free atoms: solve C = (1,0,0,0) if possible; the common-phase family admits it
        try:
            c = np.linalg.solve(Mm, np.array([1.0 + 0j, 0, 0]))
        except np.linalg.LinAlgError:
            continue
        junk = float(np.linalg.norm((full @ c)[1:]))
        M, _ = chain_M(c, v)
        rows.append((float(th), junk, float(np.linalg.norm(np.triu(M, 1), 2)),
                     float(np.sum(np.abs(c)))))
    bad = max(rows, key=lambda r: r[2])
    print(f"   n=2 junk-free common-phase family: max ||U'|| = {bad[2]:.6f} at theta={bad[0]:.4f} "
          f"(junk = {bad[1]:.2e}, sum|c_l| = {bad[3]:.4f})", flush=True)
    res["n2_junkfree_maxU"] = {"theta": bad[0], "Unorm": bad[2], "junk": bad[1], "sumc": bad[3]}
    print("   => ||U'|| is NOT O(eta): it is an O(1) function of the phases even at eta = 0,",
          flush=True)
    print("      so the bilinear correction ||L'||*||U'|| is LINEAR in eta unless it is traded",
          flush=True)
    print("      against the Gram-determinant credit sum_L (1 - G_L) -- the same coupling",
          flush=True)
    print("      lemma P2B proves for the kill-point route at n = 2.", flush=True)

    json.dump(res, open(f"{OUT}/elegant.json", "w"), indent=1, default=float)
    print(f"\nwrote {OUT}/elegant.json", flush=True)


if __name__ == "__main__":
    main()
