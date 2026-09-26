"""arch_mplus.py -- test the ABSOLUTE chain matrix M_+ = sum_l |c_l| m_l m_l^*  (PSD, no mirror problem).
Conjectures: (L+) lambda_min(M_+) >= m ; (D+) det M_+ >= m^D ; (E+) E_k(M_+) >= C(D,k) m^k ; m = min_{T^n} Re h.
Any of them, with Cauchy-Binet+Hadamard (E_k(M_+) <= D^k e_k(|c|)) or Rotfel'd, gives C1 (Re form) for every N.
Adversarial minimisation, n=1,2, N=D..D+3. fp64."""
import numpy as np, sys, json, itertools
from scipy.optimize import minimize
from math import comb
rng=np.random.default_rng(int(sys.argv[1]) if len(sys.argv)>1 else 0)
def mRe(c,v,n):
    # min over torus of Re h, h = sum c_l prod (1+u_j v_lj)
    if n==1:
        A=c.sum();C=(c*v[:,0]).sum(); return A.real-abs(C)
    G=256; th=np.exp(2j*np.pi*np.arange(G)/G)
    # n=2: min over u1 grid of Re a - |b|
    a=c.sum()+np.outer(th,(c*v[:,0])).sum(1); b=(c*v[:,1]).sum()+np.outer(th,(c*v[:,0]*v[:,1])).sum(1)
    f=a.real-np.abs(b); i=f.argmin()
    # refine
    from scipy.optimize import minimize_scalar
    def ff(t):
        u=np.exp(1j*t); aa=c.sum()+u*(c*v[:,0]).sum(); bb=(c*v[:,1]).sum()+u*(c*v[:,0]*v[:,1]).sum(); return aa.real-abs(bb)
    t0=2*np.pi*i/G; rs=minimize_scalar(ff,bounds=(t0-2*np.pi/G,t0+2*np.pi/G),method='bounded',options=dict(xatol=1e-12))
    return min(f[i],rs.fun)
def stats(x,n,N):
    D=n+1
    c=x[:N]+1j*x[N:2*N]; ph=x[2*N:].reshape(N,n); v=np.exp(1j*ph)
    m=mRe(c,v,n)
    if m<=1e-3: return None
    ch=np.concatenate([np.ones((N,1)),np.cumprod(v,axis=1)],axis=1)   # N x D
    Mp=(ch.T*np.abs(c))@ch.conj()
    ev=np.linalg.eigvalsh(Mp)
    cf=np.poly(-ev)  # prod (z+ev): coefficients E_k
    Ek=[cf[k]/(comb(D,k)*m**k) for k in range(1,D+1)]
    return ev[0]/m, Ek, m
def obj(x,n,N,which):
    s=stats(x,n,N)
    if s is None: return 1e3
    return s[0] if which=='lam' else s[1][which]
out={}
for n in [1,2]:
    D=n+1
    for N in range(D,D+3):
        for which in ['lam']+list(range(1,D)):
            best=1e9
            for t in range(25):
                x0=np.concatenate([rng.normal(size=2*N),rng.uniform(0,2*np.pi,N*n)])
                if t%2==0: # near block
                    l=np.arange(N); cc=np.where(l<D,1.0/D,0.0)+0.05*rng.normal(size=N); x0=np.concatenate([cc,0.05*rng.normal(size=N),np.tile((2*np.pi*l/D)[:,None],(1,n)).ravel()+0.3*rng.normal(size=N*n)])
                res=minimize(obj,x0,args=(n,N,which),method='Nelder-Mead',options=dict(maxiter=6000,maxfev=6000,xatol=1e-9,fatol=1e-12))
                best=min(best,res.fun)
            name='lambda_min(M+)/m' if which=='lam' else f'E_{which+1}(M+)/(C(D,k)m^k)'
            print(f"n={n} N={N} min {name} = {best:.6f}",flush=True)
            out[f"n{n}_N{N}_{which}"]=best
json.dump(out,open('FINAL_RUNS/ARCH/mplus.json','w'),indent=1)
