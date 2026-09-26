# NUMERICAL: non-separable transfer value  min over K shapes and weights of max over orders of sum_k r_k beta_sigma(m_k)/|m_k|.
import sys, numpy as np
from scipy.optimize import minimize
from transfer_cert import ORDERS, beta
rng = np.random.default_rng(0)
As, Ao, K = float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
def val(v):
    m = np.abs(v[:3 * K]).reshape(K, 3) + 1e-12; r = np.exp(v[3 * K:]); r = r / r.sum()
    m = m / np.linalg.norm(m, axis=1, keepdims=True)
    return max(sum(r[k] * beta(m[k], o, As, Ao) for k in range(K)) for o in ORDERS)
best = min(minimize(val, rng.normal(size=4 * K), method='Nelder-Mead', options=dict(maxiter=20000, xatol=1e-9, fatol=1e-11)).fun for _ in range(300))
print(f"A_s={As} A*={Ao} K={K}: non-separable value ~ {best:.4f}")
