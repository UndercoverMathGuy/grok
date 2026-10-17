# Sharp single-feature bonus at n=2:  Gstar(rho) = sup { Re A - a : a = 3^{1-s} Q^s = 1, max_T |beta_T| <= rho }
# over atom systems beta_i in C, xi_i in T^2 (paper Lemma 6), T in {1},{2},{1,2}.  SLSQP, many restarts, widths N=1..Nmax.
import numpy as np, sys, json
from scipy.optimize import minimize
def run(s,rho,N,trials,rng,warm=None):
    r=1/s; best=(-np.inf,None)
    def unpack(v):
        b=v[:N]*np.exp(1j*v[N:2*N]); t1=v[2*N:3*N]; t2=v[3*N:4*N]; return b,t1,t2
    def obj(v):
        b,t1,t2=unpack(v); return -(np.real(b.sum()))
    def eq(v):
        b,t1,t2=unpack(v); Q=np.sum((v[:N]**2+1e-30)**(r/2)); return np.array([3**(1-s)*Q**s-1])
    def ineq(v):
        b,t1,t2=unpack(v)
        B=[b@np.exp(1j*t1),b@np.exp(1j*t2),b@np.exp(1j*(t1+t2))]
        return np.array([rho**2-abs(x)**2 for x in B])
    starts=[]
    if warm is not None: starts.append(warm)
    for _ in range(trials):
        starts.append(np.concatenate([rng.uniform(0.05,1,N),rng.normal(0,0.3,N),rng.uniform(0,2*np.pi,2*N)]))
    for v0 in starts:
        try:
            res=minimize(obj,v0,constraints=[{'type':'eq','fun':eq},{'type':'ineq','fun':ineq}],method='SLSQP',options={'maxiter':1000,'ftol':1e-13})
        except Exception: continue
        v=res.x
        if np.all(ineq(v)>-1e-8) and abs(eq(v)[0])<1e-8:
            val=-obj(v)-1
            if val>best[0]: best=(val,v.copy())
    return best
if __name__=="__main__":
    s=float(sys.argv[1]); Nmax=int(sys.argv[2]); trials=int(sys.argv[3])
    rng=np.random.default_rng(0)
    rhomax=3**(s-1)
    rhos=list(np.round(np.concatenate([[0.01,0.02,0.05],np.linspace(0.1,rhomax,25)]),5))
    out={}
    for rho in rhos:
        vals=[]
        for N in range(1,Nmax+1):
            v,_=run(s,rho,N,trials,rng); vals.append(v)
        G=max(vals)
        out[float(rho)]=G
        print(f"s={s} rho={rho:.4f}  Gstar={G:.6f}  Gstar/rho={G/rho:.4f}  Gstar/rho^2={G/rho**2:.4f}  byN={np.round(vals,5)}",flush=True)
    json.dump(out,open(f'bonus_s{s}.json','w'))
