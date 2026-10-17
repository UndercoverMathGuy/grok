import numpy as np
from scipy.optimize import minimize, brentq
sig=[(1,),(2,),(2,2),(2,2,2),(2,2,2,3),(2,2,3),(2,3),(2,3,3),(3,),(3,3),(3,3,3),(3,3,3,3),(4,)]
def ratio(x,s):
    th,ps,c=x; d=1-2*c*np.cos(ps)
    if c<=0: return -9
    beta=lambda k:d+2*c*np.cos(k*th+ps)
    Q=abs(d)**(1/s)+2*c**(1/s); bonus=1-5**(1-s)*Q**s
    return bonus/max(sum(abs(beta(k)) for k in g) for g in sig)
rng=np.random.default_rng(0)
def rmax(s):
    best=-9
    for t in range(150):
        x0=np.array([rng.uniform(0,np.pi),rng.uniform(-np.pi,np.pi),rng.uniform(0.05,0.8)])
        r=minimize(lambda x:-ratio(x,s),x0,method='Nelder-Mead',options={'xatol':1e-11,'fatol':1e-13,'maxiter':4000})
        best=max(best,-r.fun)
    return best
for s in (1.25,1.5,1.75,2.0,2.25,2.5): print(s,round(rmax(s),5),flush=True)
for name,t in (("1/2",0.5),("cos(pi/5)",np.cos(np.pi/5)),("cos(pi/7)",np.cos(np.pi/7)),("1",1.0)):
    print("n=4 sym3 crosses",name,"at s =",round(brentq(lambda s:rmax(s)-t,1.1,2.5),4),flush=True)
