# Sanity checks for the lemmas of REPORT.md (exact small cases, double precision).
#  (A)  class C: E_x (Phi-A)^k = 0 for 1<=k<=p-1, and (generically) != 0 at k=p;
#       single-frequency with DC: = 0 for k<=(p-1)/2, (generically) != 0 at k=(p+1)/2.
#  (A') cone moment bound E|Phi-c|^k <= Re((A-c)^k)/cos(k pi/p), k<=(p-1)/2, on positive-margin networks.
#  (B)  slice subordination: |sum_{T subset S,|T|=k} beta_T zeta^T| cos(pi/p)^k <= (4/p)|A-c| for positive-margin class-C nets.
#  (W)  even-sublattice alternation (width theorem for all sign patterns) on positive-margin slot-symmetric nets.
import itertools, math, numpy as np
rng = np.random.default_rng(2026)

def field_classC(beta, xi, p):
    """beta (m,), xi (m,n) unimodular; returns dict x->Phi over Z_p^n with z_t = rho^{2 x_t}"""
    m, n = xi.shape; om = np.exp(4j * np.pi / p)
    X = np.array(list(itertools.product(range(p), repeat=n)))
    Z = om ** X                                                   # (p^n, n)
    vals = np.prod(1 + xi[None, :, :] * Z[:, None, :], axis=2) @ beta
    return X, vals

def field_dc(gam, alp, dl, W, p):
    """single-frequency with DC: slot polynomial g_it(z)=gam+alp z+dl z^2, z=rho^{x_t}"""
    m, n = gam.shape; rho = np.exp(2j * np.pi / p)
    X = np.array(list(itertools.product(range(p), repeat=n)))
    Z = rho ** X
    G = gam[None] + alp[None] * Z[:, None, :] + dl[None] * Z[:, None, :] ** 2
    return X, np.prod(G, axis=2) @ W

def cm(z, p): return z.real - np.abs(z.imag) / math.tan(math.pi / p)

print("(A) higher-order circularity")
for p, n, m in [(5, 3, 3), (7, 3, 4), (11, 2, 3)]:
    beta = rng.standard_normal(m) + 1j * rng.standard_normal(m); xi = np.exp(2j * np.pi * rng.random((m, n)))
    X, P = field_classC(beta, xi, p); A = P.mean()
    mom = [abs(np.mean((P - A) ** k)) / np.mean(abs(P - A) ** k) for k in range(1, p + 1)]
    print(f"  class C p={p} n={n}: |E(Phi-A)^k|/E|Phi-A|^k for k=1..p:", " ".join(f"{v:.1e}" for v in mom))
    gam, alp, dl = [rng.standard_normal((m, n)) + 1j * rng.standard_normal((m, n)) for _ in range(3)]
    W = rng.standard_normal(m) + 1j * rng.standard_normal(m)
    X, P = field_dc(gam, alp, dl, W, p); A = P.mean()
    mom = [abs(np.mean((P - A) ** k)) / np.mean(abs(P - A) ** k) for k in range(1, (p + 1) // 2 + 1)]
    print(f"  with DC  p={p} n={n}: k=1..(p+1)/2:", " ".join(f"{v:.1e}" for v in mom))

print("(A',B) cone moment bounds and slice subordination on positive-margin class-C networks")
def if_block(n):
    nu = n + 1
    beta = np.ones(nu) / nu
    xi = np.array([[np.exp(2j * np.pi * i / nu)] * n for i in range(nu)])
    return beta.astype(complex), xi
worst_B = 0; worst_A = 0; tested = 0
for p, n in [(5, 2), (5, 3), (7, 2), (7, 3), (11, 2), (11, 3)]:
    for trial in range(60):
        beta, xi = if_block(n)
        eps = rng.uniform(0.01, 0.25)
        beta = beta * np.exp(eps * (rng.standard_normal(len(beta)) + 1j * rng.standard_normal(len(beta))))
        xi = xi * np.exp(1j * eps * rng.standard_normal(xi.shape))
        X, P = field_classC(beta, xi, p); A = P.mean()
        c = cm(P, p).min()
        if c <= 0: continue
        tested += 1
        R = math.cos(math.pi / p)
        # (A') cone moments
        for k in range(1, (p - 1) // 2 + 1):
            lhs = np.mean(abs(P - c) ** k); rhs = ((A - c) ** k).real / math.cos(k * math.pi / p)
            worst_A = max(worst_A, lhs / rhs)
        # (B) slices: random S, zeta, all k
        for t in range(40):
            S = [tt for tt in range(n) if rng.random() < 0.7] or [0]
            zeta = np.exp(2j * np.pi * rng.random(len(S)))
            for k in range(1, len(S) + 1):
                tot = 0
                for T in itertools.combinations(range(len(S)), k):
                    cols = [S[j] for j in T]
                    tot += np.sum(beta * np.prod(xi[:, cols], axis=1)) * np.prod(zeta[list(T)])
                worst_B = max(worst_B, abs(tot) * R ** k / ((4 / p) * abs(A - c)))
print(f"  tested {tested} positive-margin networks; max LHS/RHS (A') = {worst_A:.4f} (must be <=1); max LHS/RHS (B) = {worst_B:.4f} (must be <=1)")

print("(W) even-sublattice alternation, slot-symmetric single features with positive margin (mixed sign patterns)")
def ss_field(U, W, n, p):
    rho = np.exp(2j * np.pi / p)
    out = {}
    def rec(prefix, left, a):
        if a == p - 1:
            N = prefix + [left]; out[tuple(N)] = np.sum(W * np.prod((U * rho ** np.arange(p)[None, :]) ** np.array(N)[None, :], axis=1)); return
        for v in range(left + 1): rec(prefix + [v], left - v, a + 1)
    rec([], n, 0); return out
ok = 0; viol = 0; checked = 0
for p, n in [(5, 6), (5, 7), (7, 6), (5, 9)]:
    nu = n + 1; th = math.pi / p
    for trial in range(400):
        # IF block with perturbation in all frequencies (mixed sign patterns appear)
        a = np.arange(p); t0 = np.pi * np.arange(nu) / nu
        U = np.array([np.cos(2 * np.pi * a / p + ti) for ti in t0]) + rng.uniform(0.0, 0.08) * rng.standard_normal((nu, p))
        W = np.exp(1j * n * t0) * (1 + 0.05 * rng.standard_normal(nu))
        F = ss_field(U, W, n, p)
        vals = np.array(list(F.values()))
        if (vals.real - np.abs(vals.imag) / math.tan(th)).min() <= 0: continue
        mixed = len(set(tuple(np.sign(u)) for u in U)) > 1
        checked += 1
        d = (p - 1) // 2; dp = (d * pow(2, -1, p)) % p          # 2 d' = d mod p
        for b in range(p):
            ap = (b + dp) % p
            K = n // 2
            # direct check of the proof's key consequence: m-1 >= number of sign changes of h_psi along determined points
            Fk = np.array([F[tuple([n - 2 * k if a_ == b else (2 * k if a_ == ap else 0) for a_ in range(p)])] for k in range(K + 1)])
            # arcs: arg F(N(k)) in (-th,th); write F = e^{i(pi-th)k} C S(k) -> arg S(k) in I_k
            C0 = Fk[0]
            Sk = Fk / np.exp(1j * (math.pi - th) * np.arange(K + 1))
            for k0 in range(min(p, K + 1)):
                psi = -(math.pi - th) * k0      # line through the centre omega_{k0} of the arc I_{k0}
                h = np.imag(np.exp(-1j * psi) * Sk)
                det = [k for k in range(K + 1) if (k - k0) % p != 0]
                sg = np.sign(h[det]); ch = int(np.sum(sg[1:] != sg[:-1]))
                if ch >= len(det) - 1: ok += 1
                else: viol += 1
print(f"  checked {checked} positive-margin nets: alternation satisfied {ok} times, violated {viol} times "
      f"(proof predicts |D|-1 sign changes on determined points)")
