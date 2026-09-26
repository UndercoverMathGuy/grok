# Share-floored discounted programme (NUMERICAL, multistart SLSQP = lower bounds on the max):
# every frequency of T must carry budget share >= eps; T gets amplitude-rate advantage delta (sqrt3 = full discount),
# others block rate. value = max_beta min_D sum_k r_k beta_k^{3/2} cov_k(D), r = 1/sqrt3 off T, delta/sqrt3 on T.
# Ratio > 1 means junk on T could pay (in this accounting) despite being forced to use every frequency of T.
import sys, numpy as np
from scipy.optimize import minimize
sys.path.insert(0, '..')
from common import cover_matrix
rng = np.random.default_rng(7)
def value(p, T, delta, eps, Gam, starts=40):
    A = cover_matrix(p); h = A.shape[0]; r = np.full(h, 3 ** -0.5); Ti = np.array(T) - 1; r[Ti] = delta / np.sqrt(3)
    cons = [{'type': 'ineq', 'fun': lambda x: A @ (r * np.clip(x[:h], 0, None) ** 1.5) - x[h]},
            {'type': 'eq', 'fun': lambda x: x[:h].sum() - 1}]
    bnds = [(0, 1)] * h + [(None, None)]
    for i in Ti: bnds[i] = (eps, 1)
    best = -9
    for s in range(starts):
        b0 = rng.dirichlet(np.ones(h) * rng.choice([0.2, 1.0])); b0[Ti] = np.maximum(b0[Ti], eps); b0 /= b0.sum()
        x0 = np.r_[b0, (A @ (r * b0 ** 1.5)).min()]
        res = minimize(lambda x: -x[h], x0, constraints=cons, bounds=bnds, method='SLSQP', options=dict(maxiter=500, ftol=1e-12))
        if res.success or res.status == 9:
            x = res.x; b = np.clip(x[:h], 0, None)
            if abs(b.sum() - 1) < 1e-6 and (b[Ti] >= eps - 1e-7).all():
                best = max(best, (A @ (r * b ** 1.5)).min())
    return best / (Gam / np.sqrt(3))
if __name__ == '__main__':
    for p, Gam in ((23, 0.40293339), (59, 0.31304888)):
        print(f"p={p}")
        for L in (3, 5, 8):
            T = list(range(1, L + 1))
            row = [value(p, T, np.sqrt(3), e, Gam) for e in (0.0, 0.01, 0.03, 0.06)]
            print(f"   AP {{1..{L}}} full discount, share floor eps = 0 / .01 / .03 / .06 : " + " / ".join(f"{v:.3f}" for v in row), flush=True)
