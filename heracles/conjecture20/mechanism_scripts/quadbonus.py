import numpy as np, sys
from scipy.optimize import minimize
s=float(sys.argv[1]); r=1/s; th=3**(1-s)
rng=np.random.default_rng(int(sys.argv[2]) if len(sys.argv)>2 else 0)
def stats(x,m):
    rho=np.abs(x[:m]); ph=x[m:2*m]; a=x[2*m:3*m]; b=x[3*m:4*m]
    beta=rho*np.exp(1j*ph)
    A=beta.sum(); BT=np.array([(beta*np.exp(1j*a)).sum(),(beta*np.exp(1j*b)).sum(),(beta*np.exp(1j*(a+b))).sum()])
    Q=(rho**r).sum(); return A,BT,Q
def ratio(x,m):
    A,BT,Q=stats(x,m); aA=abs(A)
    return (aA-th*Q**s)*aA/(np.sum(np.abs(BT)**2)+1e-300)
best=-np.inf
# IF block: 3 atoms 1/3, phases a_i=2pi i/3, b_i=2pi i/3 (chain orthogonal)
for m in [3,4,5,6]:
  for k in range(40):
    if k%2==0:
        x0=np.concatenate([np.r_[np.ones(3)/3,np.zeros(m-3)]+0.05*rng.random(m),0.05*rng.normal(size=m),
            np.r_[2*np.pi*np.arange(3)/3,rng.uniform(0,6.3,m-3)]+0.1*rng.normal(size=m),np.r_[2*np.pi*np.arange(3)/3,rng.uniform(0,6.3,m-3)]+0.1*rng.normal(size=m)])
    else:
        x0=np.concatenate([rng.random(m),rng.uniform(0,6.3,3*m)])
    res=minimize(lambda x:-ratio(x,m),x0,method='Nelder-Mead',options={'maxiter':20000,'xatol':1e-12,'fatol':1e-14})
    res=minimize(lambda x:-ratio(x,m),res.x,method='BFGS')
    v=-res.fun
    if v>best:
        best=v; A,BT,Q=stats(res.x,m)
        print(f"m={m} ratio={v:.5f} |B|/|A|={np.abs(BT)/abs(A)} bonus/|A|={(abs(A)-th*Q**s)/abs(A):.5f}",flush=True)
print("sup bonus*|A|/sum|B_T|^2 approx",best, " single atom c1/3=",(1-th)/3)
