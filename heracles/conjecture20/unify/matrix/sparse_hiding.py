"""Theorem 1 (obstruction) sanity check: at a sparse (p,n,b) the own-mode relaxation with FREE non-own modes beats IF,
arbitrarily close to the IF block, both with the chain-matrix cost (free mirror) and with the exact ATOMIC cost.

Construction.  y* = programme maximiser, B = its binding set (symmetric), assumed != Z_p^x.  Feature l in spt y*:
  * chain version: own interference eps*L on the intervals, mirror U = ((2-r)/r) eps L^*  (second-order cost decrease);
  * atomic version: nu atoms of the IF block of feature l, perturbed by eps (random), rescaled to clean amplitude y*_l;
Hiding: for every non-empty T, the coefficients d_k (k != -l) of g at chi = 2l 1_T are set so that
  G_T(delta) = sum_k d_k (1 - rho^{k delta}) vanishes on B u {0}  (G_T = c * conj(rho^{l delta}) on the non-binding set).
Everything else IF.  We compute the exact margin over all x, delta and the ratio h / (nu^{1-s} Gamma Q^s).
"""
import sys, itertools
import numpy as np
sys.path.insert(0, '/home/user/grok/heracles/conjecture20/mechanism_scripts')
from prog import solve as prog_solve, kappa
from mrel import subsets, interval

rng = np.random.default_rng(7)


def build_fields(p, n, ystar, l, own_l, B):
    """fields as coefficient dicts: phi[j][chi] ; own_l: dict T -> field coefficient of feature l at 2l1_T (T=() is clean)"""
    F = (p - 1) // 2
    rho = np.exp(2j * np.pi / p)
    phi = {j: {} for j in range(1, F + 1)}
    zero = tuple([0] * n)
    for j in range(1, F + 1):
        phi[j][zero] = ystar[j - 1] + 0j
    for T, v in own_l.items():
        if T == ():
            phi[l][zero] = v
            continue
        chi = tuple((2 * l % p) if t in T else 0 for t in range(n))
        mchi = tuple((-2 * l % p) if t in T else 0 for t in range(n))
        # non-binding set N (nonzero, not in B)
        N = [d for d in range(1, p) if d not in B]
        # G(delta) = c conj(rho^{l delta}) on N, 0 elsewhere; d_k = -Ghat(k) = -(1/p) sum_delta G rho^{-k delta}
        # need d_{-l} = v  ->  -(1/p) c |N| = v  -> c = -p v / |N|   (since conj(rho^{l d}) rho^{l d} = 1)
        c = -p * v / len(N)
        G = np.zeros(p, dtype=complex)
        for d in N:
            G[d] = c * np.conj(rho ** (l * d))
        dk = {k: -np.sum(G * rho ** (-k * np.arange(p))) / p for k in range(1, p)}
        assert abs(dk[(-l) % p] - v) < 1e-12
        # d_{-j} = phi_j(chi), d_j = conj(phi_j(-chi))
        for j in range(1, F + 1):
            if j != l:
                phi[j][chi] = phi[j].get(chi, 0) + dk[(-j) % p]
            else:
                phi[j][chi] = phi[j].get(chi, 0) + v
            phi[j][mchi] = phi[j].get(mchi, 0) + np.conj(dk[j % p])
    return phi


def margin(p, n, phi):
    F = (p - 1) // 2
    rho = np.exp(2j * np.pi / p)
    X = np.array(list(itertools.product(range(p), repeat=n)))
    h = np.inf; hB = {}
    Phis = {}
    for j in range(1, F + 1):
        Phis[j] = sum(c * np.exp(2j * np.pi * (X @ np.array(chi)) / p) for chi, c in phi[j].items())
    for d in range(1, p):
        g = sum((Phis[j] * (1 - rho ** (-j * d))).real for j in range(1, F + 1))
        hB[d] = g.min()
        h = min(h, g.min())
    return h, hB


def chain_cost(own, n, U, r):
    nu = n + 1
    H = np.zeros((nu, nu), dtype=complex)
    for t in range(nu):
        H[t, t] = own[()]
        for tp in range(t):
            H[t, tp] = own[interval(tp, t)]
    H = H + U
    return np.sum(np.linalg.svd(H, compute_uv=False) ** r)


def run(p, n, b):
    nu = n + 1; s = nu / b; r = 1 / s
    G, ystar, cov = prog_solve(p, s, restarts=8)
    K = kappa(p)
    covfull = K @ ystar
    Bhalf = [d for d in range(1, (p - 1) // 2 + 1) if covfull[d - 1] < G * (1 + 1e-7)]
    B = set(Bhalf) | set((-d) % p for d in Bhalf)
    S = [j for j in range(1, (p - 1) // 2 + 1) if ystar[j - 1] > 1e-9]
    print(f"p={p} n={n} b={b} s={s:.3f}: Gamma={G:.6f} support={S} binding(+-)={sorted(B)} (|Z_p^x|={p-1})")
    if len(B) == p - 1:
        print("   dense binding set: exact hiding impossible; skip")
        return
    IFcost = sum(nu ** (1 - r) * y ** r for y in ystar if y > 0)
    for l in S:
        y = ystar[l - 1]
        for eps in [0.2, 0.1, 0.05, 0.02]:
            # ---- chain version
            L = {T: (eps * y * (rng.normal() + 1j * rng.normal()) if T != () else 0) for T in subsets(n)}
            own = {T: (y if T == () else L[T]) for T in subsets(n)}
            Lmat = np.zeros((nu, nu), dtype=complex)
            for t in range(nu):
                for tp in range(t):
                    Lmat[t, tp] = own[interval(tp, t)]
            U = ((2 - r) / r) * Lmat.conj().T
            q_l = nu ** (-r) * chain_cost(own, n, U, r)
            phi = build_fields(p, n, ystar, l, own, B)
            h, hB = margin(p, n, phi)
            Q = IFcost - nu ** (1 - r) * y ** r + q_l
            ratio_chain = h / (nu ** (1 - s) * G * Q ** s)
            # ---- atomic version: perturbed IF block (paper Prop 9 phases), clean rescaled to y
            om = np.exp(2j * np.pi / nu)
            beta = np.full(nu, y / nu, dtype=complex)
            xi = np.array([[om ** k] * n for k in range(nu)], dtype=complex)
            beta = beta * (1 + eps * (rng.normal(size=nu) + 1j * rng.normal(size=nu)))
            xi = xi * np.exp(1j * eps * rng.normal(size=(nu, n)))
            A = beta.sum(); rot = np.conj(A) / abs(A); beta = beta * rot * y / abs(A)
            ownA = {}
            for T in subsets(n):
                bT = sum(bb * np.prod([x[t] for t in T]) for bb, x in zip(beta, xi))
                ownA[T] = np.conj(bT)
            QA = np.sum(np.abs(beta) ** r)
            phiA = build_fields(p, n, ystar, l, ownA, B)
            hA, _ = margin(p, n, phiA)
            QtotA = IFcost - nu ** (1 - r) * y ** r + QA
            ratio_atoms = hA / (nu ** (1 - s) * G * QtotA ** s)
            print(f"   l={l} eps={eps:.2f}: chain ratio {ratio_chain:.6f}  (h={h:.6f}, Gamma={G:.6f}) | atomic ratio {ratio_atoms:.6f}"
                  f"  bonus_atoms={abs(A)*0+y - nu**(1-s)*QA**s:+.2e}")


if __name__ == '__main__':
    for (p, n, b) in [(11, 2, 2.0), (13, 2, 2.0), (11, 3, 2.0), (17, 2, 2.5), (7, 2, 2.0), (5, 2, 2.0)]:
        run(p, n, b)
