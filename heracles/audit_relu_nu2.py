"""audit_relu_nu2.py -- how much does the nu = 2 upper bound actually discriminate sparse from dense?
The nu-2 chain gives margin <= (||theta||^2/(4p)) * 2 V_S for a network whose clean spectrum lives on S, with
V_S = max{ min_D sum_{k in S} x_k g_k(D) : sum x = 1 } (LP, FREE amplitudes -- the doc uses equal amplitudes M_K/K).
Dense is a theorem only if (upper bound on every sparse S) < (what a dense ReLU net provably attains) = eta_dense * V_full.
With eta_dense = 0.196 (single-frequency neurons, doc) and the only rigorous per-neuron constant eta <= 1 (Lipschitz),
the test is V_S / V_full < 0.196; with the NUMERICAL sup 0.212 it is V_S / V_full < 0.925.
uv run python heracles/audit_relu_nu2.py"""
import itertools, numpy as np
from scipy.optimize import linprog
from audit_common import Gmat

for p in (23,):
    g = Gmat(p); nf = g.shape[0]; full = p / (p - 1)
    print(f"p={p} nf={nf}  V_full={full:.4f}")
    for K in range(1, nf + 1):
        best = 0
        for S in itertools.combinations(range(1, nf), K - 1):
            S = [0] + list(S)
            # max t s.t. sum_k x_k g_k(D) >= t, sum x = 1
            c = np.zeros(K + 1); c[-1] = -1
            A = np.hstack([-g[S].T, np.ones((p - 1, 1))]); b = np.zeros(p - 1)
            r = linprog(c, A_ub=A, b_ub=b, A_eq=[[1] * K + [0]], b_eq=[1], bounds=[(0, None)] * K + [(None, None)])
            best = max(best, -r.fun)
        eq = max(g[[0] + list(S_)].sum(0).min() / K for S_ in itertools.islice(itertools.combinations(range(1, nf), K - 1), 3000))
        print(f"  K={K:2d}  V_S/V_full (free LP) {best / full:.3f}   equal-amplitude (M_K/K)/V_full >= {eq / full:.3f}")
