"""R5 check at local maxima with UNEQUAL cost shares: the KKT identity E_q[g_i] = h w_i (q a probability on the
argmin set) holds at every local maximiser w.r.t. neuron rescalings, not only at symmetric IF optima."""
import sys, math
import numpy as np
from scipy.optimize import linprog
sys.dont_write_bytecode = True
sys.path.insert(0, "/home/user/grok/heracles/conjecture20/unify/split")
import netopt as no
import elib as E
rng = np.random.default_rng(21)
out = []
for (p, n, b, m) in [(5, 2, 2.2, 7), (7, 2, 2.4, 8), (5, 3, 3.0, 7)]:
    nu = n + 1; s = nu / b; model = no.Full(p, n)
    for rep in range(6):
        U, W, h, P = no.optimise(model, rng.standard_normal((m, n, p)), rng.standard_normal((m, p)), b,
                                  stages=(0.3, 0.1, 0.03, 0.01, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5))
        if h <= 0: continue
        st = E.stats(U, W, s)
        if st['w'].max() / st['w'].min() < 1.5: continue
        G = E.gaps(U, W)[:, 1:]; hh = G.min(); mask = G <= hh + 1e-4 * abs(hh)
        Gi = np.array([E.gaps(U[i:i+1], W[i:i+1])[:, 1:][mask] for i in range(m)])
        na = Gi.shape[1]
        c = np.zeros(na + 1); c[-1] = 1
        A1 = np.hstack([Gi / hh, -np.ones((m, 1))]); A2 = np.hstack([-Gi / hh, -np.ones((m, 1))])
        res = linprog(c, A_ub=np.vstack([A1, A2]), b_ub=np.concatenate([st['w'], -st['w']]),
                      A_eq=np.concatenate([np.ones(na), [0]])[None], b_eq=[1], bounds=[(0, None)] * (na + 1), method='highs')
        line = (f"(p,n,b)=({p},{n},{b}) rep {rep}: shares {np.round(np.sort(st['w'])[::-1], 3)}; |argmin| = {na}; "
                f"min_q max_i |E_q g_i/h - w_i| = {res.fun:.2e}")
        print(line, flush=True); out.append(line)
open("check_kkt_unequal.out", "w").write("\n".join(out) + "\n")
