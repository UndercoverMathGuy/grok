# Strongest cheater so far: flat block B (bought at max discount) + clean frequencies chosen to PATCH B's blind spots
# + the programme support + random extras; all weights re-optimised.  Control: same pool without the discount on B.
import numpy as np, json, sys
from scipy.optimize import minimize
from scipy.special import logsumexp
s=1.5; c=1-(2/3)**(s-1); p=int(sys.argv[1]); m=int(sys.argv[2]); boostB=float(sys.argv[3]) if len(sys.argv)>3 else 1/(1-c)
runs=json.load(open(f'/home/user/grok/heracles/conjecture20/mechanism_scripts/relations3/large_{p}.json'))['runs']
r0=max(runs,key=lambda r:r['Gamma']); SR=np.array(r0['S']); yR=np.array(r0['y']); G=r0['Gamma']
d=np.arange(1,p); F=np.arange(1,(p-1)//2+1); rng=np.random.default_rng(11)
K=lambda S: 1-np.cos(2*np.pi*np.outer(S,d)/p)
d0=int(d[np.argmin(yR@K(SR))]); inv=pow(d0,-1,p); fold=lambda l: min(l%p,p-l%p)
B=np.array(sorted({fold(((p-1)//2-j)*inv) for j in range(m)}-set(SR.tolist())))
HB=K(B).mean(0); Z=np.where(HB<0.3)[0]                       # B's weak set
cand=np.setdiff1d(F,np.concatenate([SR,B])); kz=K(cand)[:,Z]
patch=cand[np.argsort(-kz.min(1))[:40]]                         # best patchers of the weak set (max-min strength on Z)
extra=rng.choice(np.setdiff1d(cand,patch),60,replace=False)
pool=np.concatenate([SR,patch,extra,B]); nB=len(B); KS=K(pool)
print(f"p={p}: block m={nB}, weak set |Z|={len(Z)} wrong answers; best patcher min strength on Z {kz.min(1).max():.2f}")
def optimise(boost,b0):
    def obj(z,T):
        b=np.exp(z); b=b/b.sum(); a=b**s*boost; cv=a@KS
        lse=logsumexp(-T*cv); wts=np.exp(-T*cv-lse); g=s*b**(s-1)*boost*(KS@wts)
        return lse/T, b*(g-(b@g))
    z=np.log(b0/b0.sum())
    for T in [200,1000,5000,20000,80000]:
        z=minimize(obj,z,args=(T/G,),jac=True,method='L-BFGS-B',options=dict(maxiter=500)).x
    b=np.exp(z); b/=b.sum(); return ((b**s*boost)@KS).min(), b
n=len(pool)
for label,boost in [("control (no discount)",np.ones(n)),(f"cheater (block x{boostB:.3f})",np.concatenate([np.ones(n-nB),np.full(nB,boostB)]))]:
    best=(-1,None)
    for share in [0.0,0.1,0.3]:
        b0=np.concatenate([yR**(1/s),np.full(len(patch)+len(extra),0.02/(len(patch)+len(extra))),np.full(nB,share/nB+1e-6)])
        v,b=optimise(boost,b0)
        if v>best[0]: best=(v,b)
    v,b=best
    print(f"  {label:26s}: worst case {v/G:.4f} x clean optimum | budget: block {b[-nB:].sum():.3f}, patchers {b[len(SR):len(SR)+len(patch)].sum():.3f}, programme support {b[:len(SR)].sum():.3f}",flush=True)
