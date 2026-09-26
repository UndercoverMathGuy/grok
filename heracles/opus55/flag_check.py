import numpy as np
from scipy.optimize import minimize
r=2/3; d=0.8147669648482744; e1=1.0013300797096196; cost=1.1769238058153584
sp=np.sqrt(2*e1*(e1-1)); rho=np.sqrt(2*(2*d*d+(d+sp)**2)); k=rho/np.sqrt(6)
cands=[]
if e1-2*k>0: cands.append(('A',2*(e1+k)**r+(e1-2*k)**r))
if e1-k>0: cands.append(('B',2*(e1-k)**r+(e1+2*k)**r))
c=e1; d2=(rho**2-1.5*c*c)/2
if d2>0:
    dd=np.sqrt(d2); x=1.5*c+dd; y=1.5*c-dd; cands.append(('face',x**r+max(y,0)**r))
if rho**2>=6*c*c: cands.append(('vertex',(3*c)**r))
print("rho",rho,"cands",cands)
a=np.array([1,-1,0])/np.sqrt(2); b=np.array([1,1,-2])/np.sqrt(6); best=np.inf
for s in range(300):
    z=np.random.default_rng(s).normal(size=2)
    def f(z):
        mu=z[0]*a+z[1]*b; n=np.linalg.norm(mu)
        if n>rho: mu=mu*rho/n
        lam=e1+mu
        return (np.clip(lam,0,None)**r).sum()+1e3*max(0,-lam.min())
    best=min(best,minimize(f,z,method='Nelder-Mead',options={'xatol':1e-12,'fatol':1e-14}).fun)
print("true min sum lam^r",best," -> bound",best/3**r,"  cost",cost,"  candidate-min/3^r",min(v for _,v in cands)/3**r)
