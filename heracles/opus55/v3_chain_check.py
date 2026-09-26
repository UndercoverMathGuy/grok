# Sanity: for actual populations, true cost sum|c|^r >= boundC(delta,e1) and >= boundH(delta,e1) (normalised G=1).
import numpy as np
from scipy.integrate import quad
rng=np.random.default_rng(21); r=2/3; Cr=r*np.sin(np.pi*r)/np.pi
def boundC(d,e1):
    mu=min(e1,d+2*np.sqrt(max(e1*e1-1,0)))
    e2=max(0,(3-3*d*mu)/9); e3=max(0,(1-3*d*mu-d*mu**2-d*d*mu)/27)
    f=lambda u: np.log1p(e1*np.exp(u)+e2*np.exp(2*u)+e3*np.exp(3*u))*np.exp(-r*u)
    return Cr*quad(f,-60,60,limit=400)[0]
a=np.array([1,-1,0])/np.sqrt(2); b=np.array([1,1,-2])/np.sqrt(6); th=np.linspace(0,2*np.pi,721)
def boundH(d,e1):
    sp=np.sqrt(2*e1*max(e1-1,0)); rho=np.sqrt(2*(2*d*d+(d+sp)**2)); best=np.inf
    for t in np.linspace(0,rho,41):
        lam=e1+t*(np.outer(np.cos(th),a)+np.outer(np.sin(th),b)); ok=(lam>=-1e-12).all(1)
        if ok.any(): best=min(best,(np.clip(lam[ok],0,None)**r).sum(1).min())
    return best/3**r
viol=0; worstC=np.inf; worstH=np.inf
for trial in range(3000):
    N=int(rng.integers(1,9)); kind=trial%3
    if kind==0: c=np.r_[np.ones(3)/3,0.1*rng.random(max(N-3,0))][:max(N,3)]; N=len(c); c=c*(1+rng.normal(0,.3,N))*np.exp(1j*rng.normal(0,.3,N)); v=np.exp(1j*(2*np.pi*np.arange(N)/3+rng.normal(0,.4,N))); w=np.exp(1j*(2*np.pi*np.arange(N)/3+rng.normal(0,.4,N)))
    else: c=(rng.random(N)*(3 if kind==2 else 1)+.05)*np.exp(2j*np.pi*rng.random(N)*(1 if kind==2 else .15)); v=np.exp(2j*np.pi*rng.random(N)); w=np.exp(2j*np.pi*rng.random(N))
    G=c.sum()
    if abs(G)<1e-6: continue
    c=c/G                                          # normalise G = 1 (rotate + scale)
    d=max(abs((c*v).sum()),abs((c*w).sum()),abs((c*v*w).sum())); e1=np.abs(c).sum(); cost=(np.abs(c)**r).sum()
    if d>=2: continue
    bc,bh=boundC(d,e1),boundH(d,e1)
    worstC=min(worstC,cost/bc); worstH=min(worstH,cost/bh)
    if cost<bc*(1-1e-9) or cost<bh*(1-1e-9): viol+=1; print("VIOLATION",trial,cost,bc,bh,d,e1)
print(f"populations tested; violations {viol};  min cost/boundC {worstC:.5f};  min cost/boundH {worstH:.5f}")
