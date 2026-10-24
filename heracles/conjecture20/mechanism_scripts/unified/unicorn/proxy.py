# Large-K* proxy: start from a big cover R (n_R random frequencies, weights optimised), then add a block of 32 frequencies
# bought at the maximal discount with FREE interference, re-optimise all weights, and see how much the worst case improves.
# Compares a flat-capable block (antipodal progression at R's bottleneck) with a random block of the same size.
import numpy as np, sys
from scipy.optimize import minimize
from scipy.special import logsumexp
s=1.5; c=1-(2/3)**(s-1); p=int(sys.argv[1]); nR=int(sys.argv[2]); m=32
rng=np.random.default_rng(int(sys.argv[3]) if len(sys.argv)>3 else 5); d=np.arange(1,p); F=np.arange(1,(p-1)//2+1)
K=lambda S: 1-np.cos(2*np.pi*np.outer(S,d)/p)
def optimise(KS,boost,b0):
    def obj(z,T):
        b=np.exp(z); b=b/b.sum(); a=b**s*boost; cv=a@KS
        lse=logsumexp(-T*cv); wts=np.exp(-T*cv-lse); g=s*b**(s-1)*boost*(KS@wts)
        return lse/T, b*(g-(b@g))
    z=np.log(b0/b0.sum())
    for T in [50,200,1000,5000,20000]:
        cv0=((np.exp(z)/np.exp(z).sum())**s*boost)@KS
        z=minimize(obj,z,args=(T/cv0.min(),),jac=True,method='L-BFGS-B',options=dict(maxiter=400)).x
    b=np.exp(z); b/=b.sum(); return ((b**s*boost)@KS).min(), b
R=rng.choice(F,nR,replace=False); KR=K(R)
vR,bR=optimise(KR,np.ones(nR),np.ones(nR))
d0=int(d[np.argmin((bR**s)@KR)]); inv=pow(d0,-1,p); fold=lambda l: min(l%p,p-l%p)
blocks={'flat block (antipodal progression)':np.array(sorted({fold(((p-1)//2-j)*inv) for j in range(m)}-set(R.tolist()))),
        'random block':np.array(sorted(set(rng.choice(F,m,replace=False).tolist())-set(R.tolist())))}
print(f"p={p}, cover of {nR} frequencies: optimised worst-case coverage {vR:.5f}")
for name,B in blocks.items():
    KS=np.vstack([KR,K(B)]); boost=np.concatenate([np.ones(nR),np.full(len(B),1/(1-c))])
    best=-1
    for share in [0.05,0.2,0.4]:
        b0=np.concatenate([bR*(1-share),np.full(len(B),share/len(B))])
        v,b=optimise(KS,boost,b0)
        if v>best: best,bb=v,b
    print(f"  + {name:36s}: worst case {best/vR:.4f} x before, block gets {bb[nR:].sum():.3f} of the budget",flush=True)
