# Phase-free L2 version of the single-set core:
#   L2-RT: min_delta [ sum_l (alpha_l+beta_l) kappa_l(delta) - sqrt(sum_l beta_l^2 kappa_l(delta)) ]
#          <= Gamma^{(S)}_s (sum alpha^r + lambda sum beta^r)^s
# (true depth >= sqrt(sum beta^2 kappa) since max_w |f| >= RMS_w f). L2-RT implies RT_n.
import sys, numpy as np
sys.path.insert(0,'/home/user/grok/heracles/conjecture20/unify/E_reduce'); sys.path.insert(0,'/home/user/grok/heracles/conjecture20/unify/global')
import lib as G
from scipy.optimize import minimize
rng=np.random.default_rng(1)
def ratio(x,K,s,lam,Gam):
    F=K.shape[1]; al=x[:F]**2; be=x[F:]**2; r=1/s
    lhs=np.min(K@(al+be)-np.sqrt(K@(be**2)))
    return lhs/(Gam*((al**r).sum()+lam*(be**r).sum())**s)
for p in [7,11,13,17,23,29,31]:
    F=(p-1)//2; K=G.kappa_mat(p)[:F]
    for (n,b) in [(2,2.0),(2,2.5),(3,2.0)]:
        s=(n+1)/b; lam=(n/(n+1))**(1-1/s)
        Gam=G.programme(p,s,restarts=30)[0]   # reweighted-LP lower bound on Gamma (conservative for the check)
        best=0
        for t in range(60):
            x0=np.concatenate([rng.uniform(0,1,F)*(rng.uniform()<0.5), rng.uniform(0,1,F)])
            res=minimize(lambda x:-ratio(x,K,s,lam,Gam),x0,method='L-BFGS-B',options={'maxiter':3000})
            best=max(best,-res.fun)
        print(f"p={p:3d} n={n} b={b} s={s:.2f} Gamma~{Gam:.4f}  max L2-RT ratio found {best:.5f}")
