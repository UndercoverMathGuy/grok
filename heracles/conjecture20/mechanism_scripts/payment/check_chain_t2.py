# Numerical sanity check of each step of the TARGET 2 proof on random unconstrained configurations.
#  A: theta Q^s >= L ((1/3) tr Hh^r)^s                      (cost lemma for the twisted PSD matrix)
#  B: ((1/3) tr Hh^r)^s >= theta^{lambda(Hh)}               (Jensen)
#  C: lambda(Hh) <= lt(b,y),  b=sqrt((|B|^2+|C|^2)/2)/L, y=|D'|/L
#  D: |A|-c1 M <= L [sqrt(1+C^2-2Cy) - c1 lam' b]   for C=c1 kap, kap<=1-lam'
#  final: |A| - (1-theta) max(|B|,|C|,|D|) <= theta Q^s
import numpy as np
rng=np.random.default_rng(3)
L3=np.log(3)
def lt(b,y):
    P=2*b*b+y*y; q=max(-2*b*b*y,P-1)
    nu=np.maximum(1+np.roots([1,0,-P,-q]).real,0)
    return sum(n*np.log(n) if n>0 else 0 for n in nu)/(3*L3)
worst=[np.inf]*5
for t in range(100000):
    m=rng.integers(1,7); s=rng.uniform(1.01,1.5); r=1/s; th=3**(1-s); c1=1-th
    beta=(rng.normal(size=m)+1j*rng.normal(size=m)*rng.choice([0,0.2,1]))
    if rng.random()<0.4:
        k=rng.integers(0,3,size=m); xi=np.exp(2j*np.pi*k/3+1j*rng.normal(scale=0.2,size=m)); eta=xi*np.exp(1j*rng.normal(scale=0.2,size=m))
    else:
        xi=np.exp(1j*rng.uniform(0,2*np.pi,m)); eta=np.exp(1j*rng.uniform(0,2*np.pi,m))
    rho=np.abs(beta); p=beta/rho; L=rho.sum(); Q=(rho**r).sum()
    A=beta.sum(); B=(beta*xi).sum(); C=(beta*eta).sum(); D=(beta*xi*eta).sum()
    x=p*xi; y_=p*p*xi*eta
    U=np.stack([np.ones(m),x,y_],1); H=(U.T*rho)@U.conj()
    Hh=H/L; nu=np.clip(np.linalg.eigvalsh(Hh),0,None)
    F=np.mean(nu**r)
    lamH=sum(n*np.log(n) for n in nu if n>1e-14)/(3*L3)
    b=np.sqrt((abs(B)**2+abs(C)**2)/2)/L; y=abs(y_@rho)/L
    sA=th*Q**s-L*F**s
    sB=F**s-th**lamH
    sC=lt(min(b,1),min(y,1))-lamH
    # D with optimal (lam',kap) on a grid
    M=max(abs(B),abs(C),abs(D))
    best=np.inf
    for lp in np.linspace(0,1,51):
        kap=min(y/c1 if c1>0 else 0,1-lp); Cc=c1*kap
        best=min(best,L*(np.sqrt(1+Cc*Cc-2*Cc*y)-c1*lp*b))
    sD=best-(abs(A)-c1*M)
    sF=th*Q**s-(abs(A)-c1*M)
    for k,v in enumerate([sA/L,sB,sC,sD/L,sF/L]): worst[k]=min(worst[k],v)
print("min slacks: A",worst[0]," B",worst[1]," C",worst[2]," D",worst[3]," final",worst[4])
