# Referee check of H3-num tie-breaking: c2b assigns a vertex whose RD slack is ZERO on an interval to the LAST zero,
# so a vertex optimal on [1, s_c) is never counted "at s = 1".  Count vertices with S_v(1) = min_s S_v (ties included).
import numpy as np, math, sys, itertools
sys.path.insert(0, '/home/user/grok/heracles/conjecture20/unify/E_reduce')
import elib as ER
for p in [11, 13]:
    ss = np.round(np.linspace(1.0, 1.5, 101), 4)
    _, _, best = ER.programme_vertices(p, 1.2)
    V = [v for _, v in best]
    # dedupe
    U = []
    for v in V:
        if not any(np.allclose(v, w, atol=1e-9) for w in U): U.append(v)
    Phi = np.array([[-s * math.log((v[v > 0] ** (1 / s)).sum()) for s in ss] for v in U])
    env = Phi.max(0)
    S = env[None, :] - Phi
    at1_tie = sum(1 for k in range(len(U)) if S[k, 0] <= S[k].min() + 1e-12)
    at1_strict = sum(1 for k in range(len(U)) if S[k, 0] < S[k, 1:].min() - 1e-12)
    print('p=%d: raw vertex list %d, distinct %d; vertices with s=1 among their minimisers: %d (strict unique minimiser at s=1: %d)'
          % (p, len(V), len(U), at1_tie, at1_strict))
