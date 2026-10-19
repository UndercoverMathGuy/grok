# Theorem (real phases): if U(w0+w) = U(w0-w) for some w0 (all phases real at w0), then RT_n holds:
# y = alpha + beta (1 + eps) or alpha + beta (1 - eps) (the cheaper), eps_l = cos theta_l(w0) in {+-1}, has
#   coverage(delta) >= sum (alpha+beta) kappa(delta) - D_delta >= LHS   and   sum y^r <= sum alpha^r + lambda sum beta^r.
import sys, numpy as np
sys.path.insert(0,'/home/user/grok/heracles/conjecture20/unify/global')
import lib as G
rng=np.random.default_rng(9); worst=np.inf
for trial in range(3000):
    p=int(rng.choice([7,11,13,17,19,23,29,31,37,41,43,47,53])); F=(p-1)//2; K=G.kappa_mat(p)[:F]
    n=int(rng.integers(2,9)); b=rng.uniform(2,n+1); s=(n+1)/b; r=1/s; lam=(n/(n+1))**(1-r)
    S=rng.random(F)<rng.uniform(0.2,1)
    al=rng.exponential(1,F)*S*(rng.random(F)<0.5); be=rng.exponential(1,F)*S
    eps=rng.choice([-1.0,1.0],F); w0=int(rng.integers(p)); l=np.arange(1,F+1); W=np.arange(p)
    ph=np.where(eps>0,0,np.pi)-2*np.pi*l*w0/p                 # real phases at w0
    U=(be[None,:]*np.cos(2*np.pi*np.outer(W,l)/p+ph[None,:])).sum(1)
    dep=np.array([np.max(np.abs(U-np.roll(U,d))) for d in range(1,F+1)])
    L=np.min(K@(al+be)-dep)
    if L<=0: continue
    B=(al**r).sum()+lam*(be**r).sum()
    best=0
    for sg in (1,-1):
        y=al+be*(1+sg*eps); cov=(K@y).min(); cost=(y**r).sum()
        best=max(best,cov*B**s/cost**s/L)
        assert cov>=L-1e-9, (cov,L)
    worst=min(worst,best)
print("min over 3000 random real-phase configurations of [value(cheaper half) / (L/B^s)] =",round(worst,5),"(>=1 required)")
