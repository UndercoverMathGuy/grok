# Optimise the allocation y against the joint-cost Riesz game at a fixed critical profile (rs,rt) with gain G*(rs,rt):
#   f(y) = sup{t: V_joint(y; gain t G*) >= 0} (all directions, coverage term).  Network wants f(y) < 1.
import json,numpy as np,sys,time
from scipy.optimize import minimize
from ded import run
s=1.5; p=int(sys.argv[1]); l=int(sys.argv[2]); rs=float(sys.argv[3]); rt=float(sys.argv[4]); G=float(sys.argv[5]); budget=int(sys.argv[6])
D={json.loads(x)['p']:json.loads(x) for x in open('prog15.out')}
d=D[p]; S=d['S']; y0=np.array(d['y']); r=1/s; F=(p-1)//2
def proj(v): y=y0*np.exp(v); return y/np.sum(y**r)**s
def f(v):
    yv=proj(v); y=dict(zip(S,yv)); a={j:3**(1-s)*y[j] for j in S}
    cov={dd:3**(1-s)*(d['Gamma']-sum(y[j]*(1-np.cos(2*np.pi*j*dd/p)) for j in S)) for dd in range(1,F+1)}
    g=lambda t: run(p,s,S,a,l,list(range(1,F+1)),d['Gamma'],rs*a[l],rt*a[l],t*G*a[l],starts=(0.0,),joint=True,cov=cov)[0]
    lo,hi=0.3,4.0
    for it in range(11):
        mid=0.5*(lo+hi)
        if g(mid)>=0: lo=mid
        else: hi=mid
    return lo
t0=time.time(); f0=f(np.zeros(len(S))); print(f"p={p} l={l} ({rs},{rt}): joint sigma at y* (all dirs) = {f0:.4f}",flush=True)
rng=np.random.default_rng(1); best=(f0,np.zeros(len(S)))
for eps in (0.03,0.1):
    for t in range(3):
        v=eps*rng.standard_normal(len(S)); fv=f(v)
        if fv<best[0]: best=(fv,v)
res=minimize(f,best[1]+0.02*rng.standard_normal(len(S)),method='Nelder-Mead',options={'maxfev':budget,'xatol':1e-3,'fatol':1e-4})
fb=min(res.fun,best[0])
print(f"  min over allocations: {fb:.4f} at y={np.round(proj(res.x if res.fun<=best[0] else best[1]),4).tolist()} (y*={np.round(y0,4).tolist()}) t={time.time()-t0:.0f}",flush=True)
