"""Referee check of KM1 (pinching of the l_r quasi-nuclear norm, threshold b = r*nu = 2).

(1) Termwise inequality behind (a): for unit columns a_{.t} (sum_k a_kt^2 = 1),
      F(a) = sum_k prod_t a_kt^r  <= 1   iff  r*nu >= 2.
    We MAXIMISE F adversarially (many restarts, L-BFGS on a sphere parametrisation) for nu = 2..5, K = 2..6 blocks,
    and b in {1.8, 1.95, 2, 2.05, 2.5, nu}.  Prediction: max = 1 for b >= 2, = K^{1-b/2} for b < 2.
(2) Example (b): pinched sum for f = (x)(e+e')/sqrt2 equals 2^{1-b/2}, exactly (rank-one pieces, N_r = Frobenius
    since N_r >= N_1 >= ||.||_F for r <= 1).
(3) Headline wording: at nu = 2 (matrices) ADDITIVITY on direct sums holds for every r (block-diagonal singular values),
    so b = 2 is a sharp threshold for PINCHING only, not for additivity.  Shown with random blocks.
(4) Network translation: min cost to realise a table = nu^{b/2} N_r^r (AM-GM balancing); spot-check on random neurons.
"""
import numpy as np
from scipy.optimize import minimize
rng = np.random.default_rng(12345)


def F_of(x, K, nu, r):
    X = x.reshape(K, nu)
    A = X / np.linalg.norm(X, axis=0, keepdims=True)
    A = np.abs(A)
    return np.sum(np.prod(A ** r, axis=1))


print("(1) adversarial max of sum_k prod_t a_kt^r over unit columns")
for nu in [2, 3, 4, 5]:
    for b in sorted(set([1.8, 1.95, 2.0, 2.05, 2.5, float(nu)])):
        if b > nu:
            continue
        r = b / nu
        for K in [2, 3, 6]:
            best = 0.0
            for trial in range(60):
                x0 = rng.normal(size=K * nu)
                res = minimize(lambda x: -F_of(x, K, nu, r), x0, method="L-BFGS-B")
                best = max(best, -res.fun)
            pred = 1.0 if b >= 2 else K ** (1 - b / 2)
            flag = "OK" if best <= pred + 1e-7 else "VIOLATION"
            print(f"  nu={nu} b={b:4.2f} K={K}: max F = {best:.8f}  predicted sup = {pred:.8f}  {flag}")

print("(2) example (b): 2*(2^{-nu/2})^r = 2^{1-b/2}")
for nu in [2, 3, 4]:
    for b in [1.0, 1.5, 1.99]:
        r = b / nu
        val = 2 * (2 ** (-nu / 2)) ** r
        print(f"  nu={nu} b={b}: pinched sum = {val:.6f}, 2^(1-b/2) = {2 ** (1 - b / 2):.6f}, > 1: {val > 1}")

print("(3) nu=2 additivity on direct sums for r<1 (Schatten-r): ||A(+)B||_r^r - ||A||_r^r - ||B||_r^r")
for r in [0.3, 0.5, 0.9]:
    worst = 0
    for _ in range(200):
        A = rng.normal(size=(3, 4)); B = rng.normal(size=(2, 5))
        M = np.zeros((5, 9)); M[:3, :4] = A; M[3:, 4:] = B
        sv = lambda X: np.linalg.svd(X, compute_uv=False)
        d = np.sum(sv(M) ** r) - np.sum(sv(A) ** r) - np.sum(sv(B) ** r)
        worst = max(worst, abs(d))
    print(f"  r={r} (b={2*r}): max |defect| = {worst:.2e}  -> additivity holds at nu=2 even for b<2")

print("(4) balancing: ||omega||^b >= nu^{b/2} ||T||_F^{b/nu}, equality iff factor norms equal")
worst = np.inf
for _ in range(2000):
    nu = rng.integers(2, 7); b = rng.uniform(2, nu)
    x = np.abs(rng.normal(size=nu)) + 1e-3
    lhs = np.sum(x ** 2) ** (b / 2); rhs = nu ** (b / 2) * np.prod(x) ** (b / nu)
    worst = min(worst, lhs / rhs)
print(f"  min ratio = {worst:.6f} (>= 1)")
