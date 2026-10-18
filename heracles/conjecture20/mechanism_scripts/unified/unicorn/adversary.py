# Smarter cheater: choose the flat block's placement (anchor offset) so that its weak family avoids the cover's worst wrong
# answers; then re-optimise weights with the discount and FREE interference; then charge the exact interference.
import numpy as np, sys
from scipy.optimize import minimize
from scipy.special import logsumexp
s=1.5; c=1-(2/3)**(s-1); p=int(sys.argv[1]); nR=int(sys.argv[2]); m=int(sys.argv[3])
rng=np.random.default_rng(int(sys.argv[4]) if len(sys.argv)>4 else 5); d=np.arange(1,p); F=np.arange(1,(p-1)//2+1); w=np.arange(p)
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
vR,bR=optimise(KR,np.ones(nR),np.ones(nR)); covR=(bR**s)@KR
worst=np.argsort(covR)[:4*nR]                       # the cover's near-tied worst wrong answers
fold=lambda l: min(l%p,p-l%p)
best=None
for anchor in rng.choice(d,400,replace=False):        # candidate placements
    inv=pow(int(anchor),-1,p)
    B=np.array(sorted({fold(((p-1)//2-j)*inv) for j in range(m)}-set(R.tolist())))
    strength=(1-np.cos(2*np.pi*np.outer(B,d[worst])/p)).mean(0)
    if best is None or strength.min()>best[0]: best=(strength.min(),B,int(anchor))
smin,B,anchor=best
print(f"p={p} cover={nR} block m={len(B)}: best placement gives the block min strength {smin:.2f} (avg 1, max 2) on the cover's {len(worst)} worst wrong answers")
KS=np.vstack([KR,K(B)]); boost=np.concatenate([np.ones(nR),np.full(len(B),1/(1-c))]); bestv=-1
for share in [0.05,0.2,0.4]:
    v,b=optimise(KS,boost,np.concatenate([bR*(1-share),np.full(len(B),share/len(B))]))
    if v>bestv: bestv,bb=v,b
aB=(bb**s/(1-c))[nR:]
print(f"  free interference: worst case {bestv/vR:.4f} x cover alone, block gets {bb[nR:].sum():.3f} of budget")
E=2*np.pi*np.outer(w,B)/p
def osc_s(ps,T): ph=np.cos(E+ps)@aB; return (logsumexp(T*ph)+logsumexp(-T*ph))/T
bo=None
for t in range(3):
    ps=rng.uniform(0,2*np.pi,len(B))
    for T in [0.5,2,8,32]: ps=minimize(osc_s,ps,args=(T/np.sqrt(len(B))/aB.max(),),method='L-BFGS-B',options=dict(maxiter=300)).x
    phi=np.cos(E+ps)@aB; o=phi.max()-phi.min()
    if bo is None or o<bo[0]: bo=(o,phi)
o,phi=bo; cheat=(bb**s*boost)@KS; h=np.inf
for idx in np.argsort(cheat):
    if cheat[idx]-o>=h: break
    h=min(h,cheat[idx]+(phi-np.roll(phi,d[idx])).min())
print(f"  exact, interference charged: worst case {h/vR:.4f} x cover alone   (block wave half peak-to-trough / total size {o/2/aB.sum():.3f})")
