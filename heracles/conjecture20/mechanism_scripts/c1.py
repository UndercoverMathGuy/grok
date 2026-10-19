# c1(n=2,s) = sup (|A| - 3^{1-s} Q^s) / max_T |B_T| over atom systems (beta_i, xi_i in T^2).
import numpy as np, sys
from scipy.optimize import minimize
def run(s,N,trials,rng):
    r=1/s; best=-1
    def unpack(v):
        b=v[:N]*np.exp(1j*v[N:2*N]); t1=v[2*N:3*N]; t2=v[3*N:4*N]; return b,t1,t2
    def obj(v):
        b,t1,t2=unpack(v); A=abs(b.sum()); Q=np.sum(np.abs(b)**r+1e-300)
        return -(A-3**(1-s)*Q**s)
    def cons(v):
        b,t1,t2=unpack(v)
        B=[b@np.exp(1j*t1),b@np.exp(1j*t2),b@np.exp(1j*(t1+t2))]
        return np.array([1-abs(x)**2 for x in B])
    for _ in range(trials):
        v0=np.concatenate([rng.uniform(0.05,1,N),rng.uniform(0,2*np.pi,3*N)])
        res=minimize(obj,v0,constraints=[{'type':'ineq','fun':cons}],method='SLSQP',options={'maxiter':500,'ftol':1e-12})
        if res.success or True:
            v=res.x
            if np.all(cons(v)>-1e-7): best=max(best,-res.fun)
    return best
rng=np.random.default_rng(0)
for s in [1.1,1.2,1.3,1.4,1.5]:
    vals=[run(s,N,40,rng) for N in (1,2,3,4,5)]
    print(f"s={s}: c1 by width 1..5 = {np.round(vals,4)}   single atom 1-3^(1-s) = {1-3**(1-s):.4f}",flush=True)
