# Backward elimination path for the 1-D family (reusing repo LP), best x saved as json in scratch.
import os, sys, math, json, time, numpy as np
os.environ['PYTHONDONTWRITEBYTECODE']='1'; sys.dont_write_bytecode=True
sys.path.insert(0,".")
from oned_local import lp, true_eff, lamstar, polish
p,n=int(sys.argv[1]),int(sys.argv[2]); b=2.0; J=int(sys.argv[3]) if len(sys.argv)>3 else 120
nu=n+1; s=nu/b; r=1/s
tg=np.linspace(0.003,p-1-0.003,J); lam=np.array([lamstar(t,p) for t in tg])
active=np.ones(J,bool); w=np.ones(J); log=[]
while active.sum()>=3:
    idx=np.where(active)[0]; out=lp(lam[idx],p,n,w[idx])
    if out is None or out[0]<=0: break
    c,Vt,lz=out; a=np.abs(Vt); keep=a>1e-9*a.max()
    e,x=true_eff(lam[idx],Vt,lz,p,n,b,keep); log.append((int(keep.sum()),e,x))
    newactive=np.zeros(J,bool); newactive[idx[keep]]=True
    if newactive.sum()==active.sum():
        j=idx[keep][np.argmin(a[keep])]; newactive[j]=False
    active=newactive; w=np.ones(J); w[idx[keep]]=(a[keep]/a[keep].max()+1e-3)**(r-1)
best=None
for m,e,x in sorted(log,key=lambda t:-t[1])[:4]:
    rr,xx=polish(x,m,p,n,b)
    print('m',m,'elim eff',e,'polished',rr,flush=True)
    if best is None or rr>best[0]: best=(rr,xx,m)
rr,xx,m=best
json.dump({'p':p,'n':n,'width':m,'numerical_ratio':rr,'lam':list(xx[:m]),'logV':list(xx[m:2*m]),'argV':list(xx[2*m:])},open('oned_p%d_n%d.json'%(p,n),'w'),indent=1)
print('saved eff',rr,'m',m)
