# Fast version of the theta-free Parseval criterion (C_L2):  Htil(a)=min_d [(Ka)_d + tau*g_d(a)] <= Gamma ||a||_r ?
import numpy as np, sys, cma, warnings, json
warnings.filterwarnings('ignore'); sys.path.insert(0,'..')
from prog2 import kappa
from prog3 import restricted
CS=1/(2*np.sqrt(3))
def g_all(a,K):
    # water-filling for every direction at once; K: (D,F)
    o=np.argsort(-a); A=a[o]; Ks=K[:,o]
    top=np.cumsum(Ks,axis=1)                       # sum of kappa over top j (j=1..F)
    rest=np.cumsum((Ks*A[None,:]**2)[:,::-1],axis=1)[:,::-1]  # sum over j..F-1 of kappa a^2
    D,F=K.shape; g=np.zeros(D)
    tot=top[:,-1]; act=tot>12
    for d in np.nonzero(act)[0]:
        for j in range(F):   # j capped features (indices 0..j-1)
            capk=top[d,j-1] if j>0 else 0.0
            if capk>=12: break
            R=rest[d,j]
            if R<=0: continue
            lam=np.sqrt(R/(12-capk))
            hi=A[j-1] if j>0 else np.inf
            if lam<=hi+1e-15 and lam>=A[j]-1e-15:
                m=np.minimum(A/lam,1); g[d]=CS*lam*(np.sum(Ks[d]*m)-12); break
    return g
def ratio(a,K,tau,r,s,G0):
    return np.min(K@a+tau*g_all(a,K))/np.sum(a**r)**s/G0
def search(p,s,G0,restarts=10,seed=0,maxfe=4000):
    K=kappa(p); F=K.shape[1]; r=1/s; tau=3**(s-1); rng=np.random.default_rng(seed); best=(-1,None)
    f=lambda x: -ratio(np.exp(np.clip(x-np.max(x),-30,0)),K,tau,r,s,G0)
    for R in range(restarts):
        x0=np.zeros(F) if R==0 else rng.normal(0,1.0 if R%2 else 2.5,F)
        es=cma.CMAEvolutionStrategy(x0,0.7,{'verbose':-9,'maxfevals':maxfe,'seed':int(rng.integers(1e9))}); es.optimize(f)
        if -es.result.fbest>best[0]: best=(-es.result.fbest,np.exp(es.result.xbest-np.max(es.result.xbest)))
    return best
if __name__=="__main__":
    s=1.5; sup={}
    for fn in ['supports_small.json','supports_s1.5.json']: sup.update(json.load(open(fn)))
    for p in [int(x) for x in sys.argv[1:]]:
        S=sup[str(p)]['S']; K=kappa(p)
        Q,y=restricted(K,1/s,[l-1 for l in S],np.random.default_rng(0)); G0=Q**(-s)
        F=K.shape[1]; tau=3**(s-1)
        u=ratio(np.ones(F),K,tau,1/s,s,G0)
        v,a=search(p,s,G0)
        an=a/np.sum(a**(1/s))**s; gpos=np.sum(g_all(an,K)>0)
        print(f"p={p} Gamma(S)={G0:.5f} uniform-F ratio={u:.3f}  sup ratio={v:.4f}  #dirs with gain>0 at argmax={gpos}  support size={np.sum(an>1e-3*an.max())}",flush=True)
