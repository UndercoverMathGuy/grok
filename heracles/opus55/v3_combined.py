# Combined reduction of (V3), r=2/3: cost-ratio lower bound = max( ELEGANT-complex bound, Hermitian bound ) at each (delta,e1).
# Complex (chain matrix with complex weights, mirrors |U_T| <= delta + 2 sqrt(e1^2-1) [Jensen], capped at e1):
#   e2 >= (3 - sum|J||U|)/9 ,  e3 >= |det M|/27 >= (1 - 3 delta mu - delta mu^2 - delta^2 mu)/27 ,  e1 >= 1
#   bound_C = C_r int log(1+e1 t+e2 t^2+e3 t^3) t^{-1-r} dt   (compared with 3^{1-r}(1-delta/2)^r on sum x^r)
# Hermitian: sum (3x)^r >= min sum (e1+mu)^r, sum mu=0, |mu|^2 <= 6 kappa^2, kappa = delta + sqrt(2 e1(e1-1))
import numpy as np
from scipy.integrate import quad
r=2/3; Cr=r*np.sin(np.pi*r)/np.pi
def boundC(d,e1):
    mu=min(e1,d+2*np.sqrt(max(e1*e1-1,0)))
    e2=max(0,(3-3*d*mu)/9); e3=max(0,(1-3*d*mu-d*mu**2-d*d*mu)/27)
    f=lambda u: np.log1p(e1*np.exp(u)+e2*np.exp(2*u)+e3*np.exp(3*u))*np.exp(-r*u)
    return Cr*quad(f,-60,60,limit=400)[0]                    # lower bound on sum x^r
a=np.array([1,-1,0])/np.sqrt(2); b=np.array([1,1,-2])/np.sqrt(6); th=np.linspace(0,2*np.pi,721)
def boundH(d,e1):
    k=d+np.sqrt(2*e1*max(e1-1,0)); rho=np.sqrt(6)*k; best=np.inf
    for t in np.linspace(0,rho,41):
        lam=e1+t*(np.outer(np.cos(th),a)+np.outer(np.sin(th),b)); ok=(lam>=-1e-12).all(1)
        if ok.any(): best=min(best,(np.clip(lam[ok],0,None)**r).sum(1).min())
    return best/3**r                                         # lower bound on sum x^r
worst=np.inf; rows=[]
for d in np.r_[np.linspace(0,0.5,51),np.linspace(0.52,1.98,74)]:
    target=3**(1-r)*(1-d/2)**r
    m=np.inf; arg=None
    for e1 in np.r_[1+np.logspace(-9,-1,40),np.linspace(1.1,4,59),np.linspace(4,40,19)]:
        v=max(boundC(d,e1),boundH(d,e1))
        if v<m: m=v; arg=e1
    rows.append((d,m/target,arg)); worst=min(worst,m/target)
    if m/target<1.0005 or d in (0,0.1,0.2,0.25,0.3,0.35,0.5,1.0,1.5): print(f"delta={d:.3f}: min_e1 max(C,H)/target = {m/target:.5f} (at e1={arg:.4f})",flush=True)
print("WORST:",worst)
