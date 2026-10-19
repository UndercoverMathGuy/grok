"""Referee: independent pointwise re-derivation of the (5,2) Region-II bound with a DIFFERENT certificate family.

At a data point (alpha, m1, m2, m12, phi) (A = e^{i alpha}, B = m1, C = m2, D = m12 e^{i phi}), the certified statement
needs  min(trivial, certificate) <= 0.995.  Here the certificate is optimised over ALL probabilities q on the grid
Z_5^2 x {1..4} (an SOCP, my own formulation; the HS degree-2 family of hs2/cert_hs is a subset), with my own float
payment F (pay estimate: max(P_lb, three-chain circle bound on a dense grid), which upper-estimates the rigorous F).
Because the LP family is larger, my value is <= theirs; a point where even my value exceeds 0.995 would contradict
the certificate logs.  We sample random points and then maximise min(trivial, LP) by Nelder-Mead (adversarial).
"""
import sys, os, time, warnings
warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import cvxpy as cp
from scipy.optimize import minimize
from cc2_payment import plb, chain_grid

rng = np.random.default_rng(5)
p = 5; rho = np.exp(2j * np.pi / p)
KA = 1 - np.cos(2 * np.pi / 5); GAM = 2.5 * 2 ** -1.5
X = [(a, b) for a in range(p) for b in range(p)]
c1 = np.array([1 - rho ** (-d) for d in range(1, 5)]); c2 = np.array([1 - rho ** (-2 * d) for d in range(1, 5)])
# linear maps q (25*4 vector, index x*4 + d) -> Cq_j(chi) = sum q c_j(d) rho^{<chi,x>}
chis = [(a, b) for a in range(p) for b in range(p)]
M1 = np.zeros((25, 100), complex); M2 = np.zeros((25, 100), complex)
for ci, ch in enumerate(chis):
    for xi, x in enumerate(X):
        ph = rho ** (ch[0] * x[0] + ch[1] * x[1])
        for d in range(4):
            M1[ci, xi * 4 + d] = c1[d] * ph; M2[ci, xi * 4 + d] = c2[d] * ph
own = [chis.index(c) for c in [(0, 0), (2, 0), (0, 2), (2, 2)]]
non1 = [i for i in range(25) if i not in own]; non2 = [i for i in range(25) if chis[i] != (0, 0)]
Lk = np.vstack([M1[non1], M2[non2]])
Lk_ri = np.vstack([Lk.real, Lk.imag])
qv = cp.Variable(100, nonneg=True)
coef = cp.Parameter(100); invF = cp.Parameter(nonneg=True)
u = cp.Variable(nonneg=True)
prob = cp.Problem(cp.Minimize(cp.norm(cp.hstack([u * invF, np.sqrt(12) * cp.norm(Lk_ri @ qv)]))),
                  [cp.sum(qv) == 1, u >= coef @ qv])


def lp_value(alpha, m1, m2, m12, phi, F):
    A = np.exp(1j * alpha); own_c = np.conj(np.array([A, m1, m2, m12 * np.exp(1j * phi)]))
    # a_own = Re sum_k own_c[k] * Cq_1(chi_k)
    coef.value = np.real(own_c[:, None] * M1[own]).sum(0)
    invF.value = 1 / F
    prob.solve(solver=cp.CLARABEL)
    return prob.value / GAM


def F_est(m, php):
    return max(plb(*m), chain_grid(np.array(m), php, nL=40, nps=72))


def bound_at(z):
    alpha, m1, m2, m12, php = z
    alpha = min(max(alpha, 0), np.pi / 2)
    m1, m2, m12 = [min(max(v, 0), 1.4) for v in (m1, m2, m12)]
    if m1 < m2: m1, m2 = m2, m1
    F = F_est((m1, m2, m12), php)
    triv = KA * np.cos(alpha) / (GAM * F)
    if triv <= 0.6:
        return triv, triv, None
    v = lp_value(alpha, m1, m2, m12, php - alpha, F)
    return min(triv, v), triv, v


if __name__ == '__main__':
    t0 = time.time()
    # sanity: IF block data must give 0.7818 (= kappa_a / Gamma), single atom data
    print("IF data (0,0,0,0):", bound_at([0, 0, 0, 0, 0]))
    print("single atom (0,1,1,1,0):", bound_at([0, 1, 1, 1, 0]))
    vals = []
    for k in range(150):
        z = [rng.uniform(0, 0.9), rng.uniform(0, 1.4), rng.uniform(0, 1.4), rng.uniform(0, 1.4), rng.uniform(-np.pi, np.pi)]
        v, tr, lp = bound_at(z)
        vals.append((v, z))
    vals.sort(key=lambda t: -t[0])
    print("random sample: top values", [round(v, 4) for v, _ in vals[:8]], flush=True)
    best = 0; bestz = None
    for v0, z0 in vals[:3]:
        res = minimize(lambda z: -bound_at(z)[0], z0, method='Nelder-Mead', options=dict(maxiter=150, xatol=1e-4, fatol=1e-6))
        print(f"  polished from {v0:.4f}: {-res.fun:.5f} at {np.round(res.x, 4)}", flush=True)
        if -res.fun > best:
            best, bestz = -res.fun, res.x
    print(f"adversarial pointwise sup of min(trivial, LP-certificate) found = {best:.5f} (claim needs <= 0.995) "
          f"at {np.round(bestz, 4)}; time {time.time()-t0:.0f}s")
