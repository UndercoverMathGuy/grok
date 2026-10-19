# Relaxation of the relations case (n=2, all interference on T={1,2}):
#   eta <= V(y,t,th) = min_{delta,w} sum_l y_l kappa_l(delta) [1 + t_l (c + cos(2 pi l w/p + th_l))],
#   with sum y^r <= 1, 0 <= t_l <= tau.  Theorem A: c = 1/(2 sqrt 3), tau = 3^{s-1}.
# Question: is sup V/(sum y^r)^s equal to Gamma_s(p)?  (ratio printed = sup / Gamma_s)
import numpy as np, sys, cma, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0,'..')
from prog2 import kappa
from progx import exhaustive
def setup(p):
    F=(p-1)//2; l=np.arange(1,F+1); K=kappa(p)            # K[delta,l]
    W=2*np.pi*np.outer(np.arange(p),l)/p                  # W[w,l]
    return F,l,K,W
def V(y,t,th,K,W,c):
    G=1+t[None,:]*(c+np.cos(W+th[None,:]))               # p x F
    # rows (delta,w): sum_l y_l K[d,l] G[w,l]
    return np.min((K*y[None,:])@G.T)
def search(p,s,c,tau,G0,restarts=20,seed=0,maxfe=20000):
    F,l,K,W=setup(p); r=1/s; rng=np.random.default_rng(seed); best=(-1,None)
    def unpack(x):
        y=np.exp(np.clip(x[:F]-np.max(x[:F]),-30,0)); t=tau/(1+np.exp(-x[F:2*F])); th=x[2*F:]; return y,t,th
    def f(x):
        y,t,th=unpack(x); return -V(y,t,th,K,W,c)/np.sum(y**r)**s/G0
    for R in range(restarts):
        x0=np.concatenate([rng.normal(0,1.5,F)-1,rng.normal(2,2,F),rng.uniform(0,2*np.pi,F)])
        es=cma.CMAEvolutionStrategy(x0,0.8,{'verbose':-9,'maxfevals':maxfe,'seed':int(rng.integers(1e9))})
        es.optimize(f)
        v=-es.result.fbest
        if v>best[0]: best=(v,unpack(es.result.xbest))
    return best
if __name__=="__main__":
    s=float(sys.argv[1]); mode=sys.argv[2]; ps=[int(a) for a in sys.argv[3:]]
    for p in ps:
        F=(p-1)//2
        G0,S0,y0=exhaustive(p,s,min(F,5))
        if mode=='A': c,tau=1/(2*np.sqrt(3)),3**(s-1)
        elif mode=='fam': c,tau=1-(2/3)**(s-1),(3/2)**(s-1)  # aligned family: |beta|=|D|=(3/2)^{s-1}a -> written as y(1+t(c+cos)) with t=tau? see note
        v,(y,t,th)=search(p,s,c,tau,G0)
        yn=y/np.sum(y**(1/s))**s
        print(f"p={p} s={s} mode={mode} Gamma={G0:.5f} supp*={S0}  sup ratio={v:.5f}  y={np.round(yn,3)} t={np.round(t,2)}",flush=True)
