# Sharp single-T payment constant at n=2:  c12(s) = sup (|A| - 3^{1-s} (sum|beta_i|^r)^s) / |D|
# over atoms beta_i in C, unit xi_i, eta_i with B = sum beta xi = 0, C = sum beta eta = 0, D = sum beta xi eta.
import numpy as np, sys
from scipy.optimize import minimize
def unpack(x,m):
    rho=np.abs(x[:m]); ph=x[m:2*m]; a=x[2*m:3*m]; b=x[3*m:4*m]
    return rho*np.exp(1j*ph), np.exp(1j*a), np.exp(1j*b)
def run(s,m,trials,seed):
    r=1/s; rng=np.random.default_rng(seed); best=-9
    for t in range(trials):
        x0=np.concatenate([rng.uniform(0.2,1,m),rng.uniform(0,2*np.pi,3*m)])
        obj=lambda x: -(abs(np.sum(unpack(x,m)[0]))-3**(1-s)*np.sum(np.abs(x[:m])**r)**s)
        def cons(x):
            be,xi,et=unpack(x,m); B=np.sum(be*xi); C=np.sum(be*et); return np.array([B.real,B.imag,C.real,C.imag])
        def dnorm(x):
            be,xi,et=unpack(x,m); return abs(np.sum(be*xi*et))-1.0
        res=minimize(obj,x0,method='SLSQP',constraints=[{'type':'eq','fun':cons},{'type':'eq','fun':dnorm}],options={'maxiter':500,'ftol':1e-12})
        if res.success and np.max(np.abs(cons(res.x)))<1e-8 and abs(dnorm(res.x))<1e-8:
            best=max(best,-res.fun)
    return best
s=float(sys.argv[1])
for m in [2,3,4,5,6]:
    print(f"s={s} m={m} atoms: sup (|A|-a)/|D| ~ {run(s,m,150,m):.4f}   aligned family 1-(2/3)^(s-1)={1-(2/3)**(s-1):.4f}   Theorem A 1/(2sqrt3)=0.2887",flush=True)
