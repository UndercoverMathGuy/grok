# Forced-spread probe of TDP*: allocations whose N support frequencies all carry weight >= lam * (max weight),
# found by reweighted LP over a support grown greedily from the programme optimum (best-coverage additions).
# Reports e(a), min kappa-mass over the binding set, E0/m at the best d, and Phi*/Gamma (TDP* closes iff <= 1),
# together with e_thr(p) = 1/(1 + tau'(c - sqrt(2/p))_+) (allocations with e <= e_thr are closed by Parseval alone).
import numpy as np, sys
from scipy.optimize import linprog
from lib import *
from tdp_map import phi_star

def forced_prog(K, S, s, lam, iters=60):
    r = 1/s; F = K.shape[1]; S = np.array(sorted(S)); KS = K[:, S]; n = len(S)
    # variables y (weights on S) and scale; constraint y_l >= lam * z where z is a free 'level' var: y_l - lam z >= 0, z <= y_l for top?
    # simpler: fix the floor relative to the current solution's max, iterate.
    w = np.ones(n); y = np.ones(n); best = (np.inf, None)
    for it in range(iters):
        lb = lam*y.max()/max(1e-12, (np.sum(y**r))**(1/r))   # floor relative to current scale (normalised below)
        res = linprog(w, A_ub=-KS, b_ub=-np.ones(K.shape[0]), bounds=[(0, None)]*n, method='highs')
        if res.status != 0: break
        y0 = res.x
        # enforce floor: raise small weights to lam*max (keeps coverage >= 1), then renormalise cost later
        yf = np.maximum(y0, lam*y0.max())
        Q = np.sum(yf**r)
        if Q < best[0]: best = (Q, yf.copy())
        wn = (yf + 1e-9*yf.max())**(r-1)
        if np.allclose(wn, w, rtol=1e-6): break
        w = wn; y = yf
    a = np.zeros(F); a[S] = best[1]; return a

if __name__ == '__main__':
    s = 1.5; r, c, tp = consts(s)
    p = int(sys.argv[1]); Ns = [int(x) for x in sys.argv[2].split(',')]; lams = [float(x) for x in sys.argv[3].split(',')]
    K = kmat(p); F = K.shape[1]; rng = np.random.default_rng(1)
    G, y0 = load_support(p); G = max(G, gamma_lower(p, s, y0)); S0 = list(np.nonzero(y0)[0])
    eth = 1/(1 + tp*max(0, c - np.sqrt(2/p)))
    print(f"p={p}: Gamma_ref={G:.5f}  K*={len(S0)}  e_thr(p)={eth:.4f}", flush=True)
    for N in Ns:
        # grow support greedily: add the frequency maximising the min coverage of (current uniform allocation + it)
        S = list(S0)
        while len(S) < N:
            cand = rng.choice(np.setdiff1d(np.arange(F), S), min(200, F-len(S)), replace=False)
            base = K[:, S].sum(1)
            j = cand[np.argmax([(base + K[:, j]).min() for j in cand])]
            S.append(int(j))
        for lam in lams:
            a = forced_prog(K, S, s, lam)
            cov = K @ a; m = cov.min(); B = np.nonzero(cov <= m*(1+1e-7))[0]; spt = a > 0
            km = min(K[b, spt].sum() for b in B)
            ph, d = phi_star(a, K, s, c, tp)
            E0 = waterfill_excess(a, K[d], c)
            e = m/(G*norm_r(a, s))
            print(f"  N={N} lam={lam}: |spt|={spt.sum()} e={e:.4f} {'(closed by e_thr)' if e <= eth else ''} "
                  f"min kappa-mass={km:.2f} E0/m={E0/((a*K[d]).sum()):.4f} Phi*/Gamma={ph/G:.4f}", flush=True)
