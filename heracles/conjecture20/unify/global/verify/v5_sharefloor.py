# R13 (share floor) independent checks.
#  (1) piece bound: for class-C neurons (all factors in the frequency-j plane, no DC),
#      max_{x,d} |g_piece(x,d)| <= 2^{n+1} B_p sum_i ||omega_i||^nu <= 2^{n+1} B_p P_A^s.
#  (2) h(theta) <= h(theta \ A) + g_A(x_R, d_R) at every minimiser (x_R,d_R) of theta \ A's gaps (pure algebra).
#  (3) psi = nu^{-1} (s Gamma_s / 2^{n+1})^{1/(s-1)} at p=11, n=2, b=2 (REPORT: ~3e-3).
#  (4) consistency on IF optima (putative maximisers): removing any subset A must satisfy
#      g_A(x_R,d_R) >= gamma_IF (P^s - (P-P_A)^s)  (otherwise IF would not be globally optimal).
import itertools, math, numpy as np
from vnet import *

rng = np.random.default_rng(5)
worst = -np.inf
for trial in range(400):
    p = int(rng.choice([5, 7, 11])); n = int(rng.choice([2, 3])); m = int(rng.integers(1, 5)); j = int(rng.integers(1, (p - 1) // 2 + 1))
    b = rng.uniform(2, n + 1); s = (n + 1) / b
    Bj = freq_basis(p, j)
    U = rng.normal(size=(m, n, 2)) @ Bj * rng.uniform(0.1, 3, size=(m, 1, 1)); W = rng.normal(size=(m, 2)) @ Bj
    G = gaps(table(U, W))
    S = neuron_sq(U, W)
    bound1 = 2 ** (n + 1) * B_p(p, n) * np.sum(S ** ((n + 1) / 2)); bound2 = 2 ** (n + 1) * B_p(p, n) * cost(U, W, b) ** s
    worst = max(worst, np.abs(G).max() / bound1)
    assert bound1 <= bound2 * (1 + 1e-12)
print(f"(1) class-C piece bound: max |g|/(2^(n+1) B_p sum||w||^nu) = {worst:.10f} (<=1 required; sum<=P^s also checked)")

# (2)
bad = 0
for trial in range(200):
    p = 5; n = 2; m = 5
    U = rng.normal(size=(m, n, p)); W = rng.normal(size=(m, p)); A = rng.random(m) < 0.4
    if A.all() or (~A).all(): continue
    GR = gaps(table(U[~A], W[~A])); GA = gaps(table(U[A], W[A])); Gt = gaps(table(U, W))
    mins = np.argwhere(GR == GR.min())
    for idx in mins:
        if Gt.min() > GR.min() + GA[tuple(idx)] + 1e-9: bad += 1
print(f"(2) h(theta) <= h(theta\\A) + g_A(x_R,d_R): violations {bad}")

# (3) psi at p=11, n=2, b=2 (Gamma_1.5(11) = 0.53572593 from v2_programme.out)
G11 = 0.53572593; s = 1.5; n = 2; nu = 3
psi = (s * G11 / 2 ** (n + 1)) ** (1 / (s - 1)) / nu
print(f"(3) psi(p=11,n=2,b=2) = {psi:.4e}, share floor psi/(1+psi) = {psi/(1+psi):.4e}")

# (4) IF optimum at (5,2), b=2: y = 2^{-s}(1,1) (uniform vertex), blocks of 3 neurons
p, n, b = 5, 2, 2.0; s = 1.5; nu = 3
Gam = 2.5 * 2 ** -s
gIF = nu ** (1 - s) * B_p(p, n) * Gam
Us, Ws = [], []
for j in (1, 2):
    lam = (2 ** n * 2 ** -s / nu) ** (1 / nu)
    U, W = if_block(p, n, j, lam); Us.append(U); Ws.append(W)
U = np.concatenate(Us); W = np.concatenate(Ws)
print(f"(4) IF network gamma/gamma_IF = {gamma(U, W, b) / gIF:.10f}")
P = cost(U, W, b); worst = np.inf
for k in range(1, 6):
    for A in itertools.combinations(range(6), k):
        A = np.array([i in A for i in range(6)])
        PA = cost(U[A], W[A], b)
        GR = gaps(table(U[~A], W[~A])); GA = gaps(table(U[A], W[A]))
        mins = np.argwhere(GR <= GR.min() + 1e-12)
        lhs = min(GA[tuple(i)] for i in mins)
        worst = min(worst, lhs / (gIF * (P ** s - (P - PA) ** s)))
print(f"(4) min over subsets A of g_A(x_R,d_R) / (gamma_IF (P^s-(P-P_A)^s)) = {worst:.6f} (>=1 consistent with IF optimality)")
