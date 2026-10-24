# Forced block share f: optimise all weights inside R and inside B (shares fixed), discount on B, interference ignored
# (most generous to the cheater).  Looks for a second valley at large f that local search from f=0 would miss.
import numpy as np, json, sys
from scipy.optimize import minimize
from scipy.special import logsumexp
s=1.5; c=1-(2/3)**(s-1); p=int(sys.argv[1]); m=int(sys.argv[2]); kind=sys.argv[3]
runs=json.load(open(f'/home/user/grok/heracles/conjecture20/mechanism_scripts/relations3/large_{p}.json'))['runs']
r0=max(runs,key=lambda r:r['Gamma']); SR=np.array(r0['S']); yR=np.array(r0['y']); G=r0['Gamma']
d=np.arange(1,p); rng=np.random.default_rng(3)
K=lambda S: 1-np.cos(2*np.pi*np.outer(S,d)/p)
d0=int(d[np.argmin(yR@K(SR))]); inv=pow(d0,-1,p); fold=lambda l: min(l%p,p-l%p)
if kind=='antipodal': B=sorted({fold(((p-1)//2-j)*inv) for j in range(m)}-set(SR.tolist()))
elif kind=='random': B=sorted(set(rng.choice(np.arange(1,(p-1)//2+1),m,replace=False).tolist())-set(SR.tolist()))
B=np.array(B); KR=K(SR); KB=K(B); nR=len(SR)
for f in [0.0,0.05,0.1,0.2,0.35,0.5,0.7]:
    def obj(z,T):
        bR=np.exp(z[:nR]); bR=(1-f)*bR/bR.sum(); bB=np.exp(z[nR:]); bB=f*bB/bB.sum()
        cv=bR**s@KR + (bB**s/(1-c))@KB
        lse=logsumexp(-T*cv); wts=np.exp(-T*cv-lse)
        gR=s*bR**(s-1)*(KR@wts); gB=s*bB**(s-1)/(1-c)*(KB@wts)
        return lse/T, np.concatenate([bR*(gR-(bR@gR)/(1-f)) if f<1 else 0*bR, bB*(gB-(bB@gB)/max(f,1e-12))])
    z=np.concatenate([np.log(yR**(1/s)), np.zeros(len(B))])
    for T in [200,1000,5000,20000]:
        z=minimize(obj,z,args=(T/G,),jac=True,method='L-BFGS-B',options=dict(maxiter=500)).x
    bR=np.exp(z[:nR]); bR=(1-f)*bR/bR.sum(); bB=np.exp(z[nR:]); bB=f*bB/bB.sum()
    cv=bR**s@KR+(bB**s/(1-c))@KB
    print(f"p={p} block={kind} m={len(B)} share f={f:.2f}: best value with free discount = {cv.min()/G:.4f} x clean optimum",flush=True)
