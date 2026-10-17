# Sharp single-feature bonus at n=2 with continuation in rho.
#   Gstar(rho) = sup { Re A - 1 : 3^{1-s} Q^s = 1, max_T |beta_T| <= rho },  atoms beta_i in C, xi_i in T^2.
import numpy as np, sys, json, time
from scipy.optimize import minimize
def make(s,N):
    r=1/s
    def unpack(v): return v[:N]*np.exp(1j*v[N:2*N]), v[2*N:3*N], v[3*N:4*N]
    def obj(v): return -np.real(np.sum(v[:N]*np.exp(1j*v[N:2*N])))
    def eq(v): return np.array([3**(1-s)*np.sum((v[:N]**2+1e-30)**(r/2))**s-1])
    def BT(v):
        b,t1,t2=unpack(v); return np.array([b@np.exp(1j*t1),b@np.exp(1j*t2),b@np.exp(1j*(t1+t2))])
    return obj,eq,BT
def opt(s,N,rho,v0):
    obj,eq,BT=make(s,N)
    ineq=lambda v: rho**2-np.abs(BT(v))**2
    try:
        res=minimize(obj,v0,constraints=[{'type':'eq','fun':eq},{'type':'ineq','fun':ineq}],method='SLSQP',options={'maxiter':600,'ftol':1e-14})
    except Exception: return -np.inf,v0
    v=res.x
    if np.all(ineq(v)>-1e-9) and abs(eq(v)[0])<1e-9: return -obj(v)-1,v
    return -np.inf,v
def ifblock(s,N,rng,eps):
    # equal atoms on the first 3 slots (IF block), extra atoms tiny
    amp=np.full(N,1e-3); amp[:min(3,N)]=1.0
    v=np.concatenate([amp,np.zeros(N),2*np.pi*np.arange(N)/3,2*np.pi*np.arange(N)/3])
    v=v+eps*rng.standard_normal(4*N)
    obj,eq,BT=make(s,N); sc=(1/(eq(v)[0]+1))**(1/s); v[:N]*=sc  # rescale amplitudes (Q^s homogeneous deg 1 in amp)
    return v
if __name__=="__main__":
    s=float(sys.argv[1]); trials=int(sys.argv[2]); Ns=[1,2,3,4,5]
    rng=np.random.default_rng(int(sys.argv[3]) if len(sys.argv)>3 else 0)
    rhomax=3**(s-1)
    rhos=np.concatenate([[0.02,0.05,0.1],np.arange(0.15,rhomax,0.05),[rhomax]])
    warm={N:[] for N in Ns}; out={}; t0=time.time()
    for rho in rhos:
        vals={}
        for N in Ns:
            best=(-np.inf,None)
            starts=[w for w in warm[N]]
            starts+=[ifblock(s,N,rng,0.05+0.3*rng.random()) for _ in range(trials)]
            starts+=[np.concatenate([rng.uniform(0.05,1,N),rng.normal(0,0.3,N),rng.uniform(0,2*np.pi,2*N)]) for _ in range(trials)]
            for v0 in starts:
                g,v=opt(s,N,rho,v0)
                if g>best[0]: best=(g,v)
            vals[N]=best[0]
            if best[1] is not None: warm[N]=[best[1]]
        G=max(vals.values()); out[float(rho)]=G
        print(f"s={s} rho={rho:.4f} Gstar={G:.6f} G/rho={G/rho:.4f} G/rho^2={G/rho**2:.4f} byN={[round(vals[N],5) for N in Ns]} t={time.time()-t0:.0f}",flush=True)
    json.dump(out,open(f'bonus_s{s}_seed{sys.argv[3] if len(sys.argv)>3 else 0}.json','w'))
