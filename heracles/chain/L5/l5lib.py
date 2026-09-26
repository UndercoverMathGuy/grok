"""L5 helpers. Units: budget shares beta_k (sum beta = 1). Own-share bound: gamma <= B V_s(beta),
V_s(beta) = min_D sum_k beta_k^{3/2} cov_k(D). Block value gamma_sym = B Gamma*/sqrt3.
Weighted programme: Gamma_c = max min_D sum y cov  s.t.  sum_k c_k y_k^{2/3} <= 1  (exact, by vertex enumeration).
c_k = 3^{1/3}: block rate (junk-free, nu-neuron block).  c_k = 1: full discount (one atom, junk perfectly hidden)."""
import itertools, sys
import numpy as np
sys.path.insert(0, '..')
from common import cover_matrix

def vertices(p):
    """all vertices of {v >= 0, A v >= 1} (astra ARGUMENT sec 2.1), deduplicated."""
    A = cover_matrix(p); h = A.shape[0]; out = {}
    for K in range(1, h + 1):
        for S in itertools.combinations(range(h), K):
            for T in itertools.combinations(range(h), K):
                M = A[np.ix_(T, S)]
                if abs(np.linalg.det(M)) < 1e-12:
                    continue
                vS = np.linalg.solve(M, np.ones(K))
                if (vS < -1e-12).any():
                    continue
                v = np.zeros(h); v[list(S)] = np.clip(vS, 0, None)
                if (A @ v < 1 - 1e-9).any():
                    continue
                out[tuple(np.round(v, 10))] = v
    return np.array(list(out.values()))

def gamma_c(V, c):
    """Gamma_c = (min over vertices of sum c_k v_k^{2/3})^{-3/2}; returns (value, argmin vertex)."""
    costs = (c[None, :] * V ** (2 / 3)).sum(1)
    i = costs.argmin()
    return costs[i] ** -1.5, V[i]
