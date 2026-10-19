# O2 test model: IF carriers with OUTPUT-side riding (output components at k+-2l, proportional to the
# block's clean output weights) hiding the single interference character chi=(2l,2l) of a two-neuron
# ("aligned") cheap feature l.  n=2.  Exact field model (large-p min over x replaced by -|Z|, conservative).
import numpy as np, sys
sys.path.insert(0,'..')
from prog3 import solve
from scipy.optimize import minimize
def run(p,s,l,S,starts=30,seed=0):
    rng=np.random.default_rng(seed); nu=3
    d=np.arange(1,p); rho=lambda m: np.exp(-2j*np.pi*np.outer(d,m)/p)
    S=list(S); K=len(S); others=[k for k in S if k!=l]; m=len(others)
    kap=1-np.cos(2*np.pi*np.outer(d,S)/p)              # (p-1) x K
    cl=1-np.exp(-2j*np.pi*d*l/p)
    cp=np.array([1-np.exp(-2j*np.pi*d*(k+2*l)/p) for k in others]).T   # (p-1) x m
    cm=np.array([1-np.exp(-2j*np.pi*d*(k-2*l)/p) for k in others]).T
    il=S.index(l); io=[S.index(k) for k in others]
    def unpack(z):
        q=np.abs(z[:K]); q=q/q.sum(); phi=z[K]
        mu=z[K+1:K+1+m]+1j*z[K+1+m:K+1+2*m]; nv=z[K+1+2*m:K+1+3*m]+1j*z[K+1+3*m:K+1+4*m]
        return q,phi,mu,nv
    def margins(z):
        q,phi,mu,nv=unpack(z)
        r=nu**(1-s)*q**s; r[il]=2**(1-s)*q[il]**s
        clean=r.copy(); load=np.abs(mu)**2+np.abs(nv)**2
        clean[io]=np.sqrt(np.maximum(r[io]**2-load,0))
        Z=r[il]*np.exp(1j*phi)*cl+cp@mu+np.conj(cm)@np.conj(nv)
        return kap@clean-np.sqrt(np.abs(Z)**2+1e-16), r, load
    best=-np.inf; bz=None
    for st in range(starts):
        z0=np.concatenate([rng.dirichlet(np.ones(K)),[rng.uniform(0,2*np.pi)],0.05*rng.standard_normal(4*m)])
        x0=np.concatenate([z0,[0.0]])
        cons=[{'type':'ineq','fun':lambda x: margins(x[:-1])[0]-x[-1]},
              {'type':'ineq','fun':lambda x: (lambda mr: mr[1][io]**2-mr[2])(margins(x[:-1]))}]
        res=minimize(lambda x:-x[-1],x0,constraints=cons,method='SLSQP',options={'maxiter':3000,'ftol':1e-12})
        x=res.x; t=margins(x[:-1])[0].min()
        if t>best: best=t; bz=x[:-1]
    return best,bz,unpack(bz)
if __name__=="__main__":
    s=float(sys.argv[1])
    for p in [int(a) for a in sys.argv[2:]]:
        G,y,_=solve(p,s); S=list(np.nonzero(y)[0]+1); IF=3**(1-s)*G
        for l in S:
            t,z,(q,phi,mu,nv)=run(p,s,l,S)
            print(f"p={p} S={S} cheap l={l}: best margin/IF = {t/IF:.4f}  q={np.round(q,3)} |mu|={np.round(np.abs(mu),4)} |nu|={np.round(np.abs(nv),4)}",flush=True)
