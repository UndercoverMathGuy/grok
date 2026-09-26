"""arch_eplus.py -- adversarial test of CONJECTURE (E+):
   det(I + t M_+) >= (1 + t m)^D  for all t>=0,   M_+ := sum_l |c_l| m_l m_l^*  (PSD 'absolute chain matrix'),
   equivalently E_k(M_+) >= C(D,k) m^k, k=1..D;  m = min_{T^n} Re h   (variant 'hmin': m -> min |h|).
usage: arch_eplus.py n N which(version: m|hmin) nstart seed"""
import numpy as np, sys, json
from scipy.optimize import minimize
from math import comb
n=int(sys.argv[1]); N=int(sys.argv[2]); ver=sys.argv[3]; nstart=int(sys.argv[4]); seed=int(sys.argv[5])
D=n+1; rng=np.random.default_rng(seed); NS=float(sys.argv[6]) if len(sys.argv)>6 else 1.0
G={1:1,2:360,3:40}[n]
th=np.exp(2j*np.pi*np.arange(G)/G)
def hparts(c,v,U):
    # U: array (...,n-1) of first n-1 torus coords; returns a,b with h = a + b u_n
    g=np.ones(U.shape[:-1]+(len(c),),dtype=complex)
    for j in range(n-1): g=g*(1+U[...,j,None]*v[:,j])
    return (g*c).sum(-1),(g*c*v[:,n-1]).sum(-1)
def val(a,b): return a.real-np.abs(b) if ver=='m' else np.abs(np.abs(a)-np.abs(b))
def aimed(c,v):
    if n==1:
        a,b=hparts(c,v,np.zeros((1,0))); return float(val(a,b)[0])
    grids=np.stack(np.meshgrid(*([th]*(n-1)),indexing='ij'),-1).reshape(-1,n-1)
    a,b=hparts(c,v,grids); f=val(a,b); i=f.argmin(); t0=np.angle(grids[i])
    def ff(t):
        a,b=hparts(c,v,np.exp(1j*t)[None,:]); return float(val(a,b)[0])
    rs=minimize(ff,t0,method='Nelder-Mead',options=dict(xatol=1e-10,fatol=1e-14,maxiter=400))
    return min(f[i],rs.fun)
def ratios(x):
    c=x[:N]+1j*x[N:2*N]; v=np.exp(1j*x[2*N:].reshape(N,n))
    m=aimed(c,v)
    if m<=1e-3*np.abs(c).sum(): return None
    ch=np.concatenate([np.ones((N,1)),np.cumprod(v,axis=1)],axis=1)
    ev=np.clip(np.linalg.eigvalsh((ch.T*np.abs(c))@ch.conj()),0,None)
    cf=np.poly(-ev)
    return np.array([cf[k]/(comb(D,k)*m**k) for k in range(1,D+1)]),m,ev
def obj(x,k):
    s=ratios(x); return 1e3 if s is None else s[0][k]
best=np.full(D,1e9); rec=[None]*D
for t in range(nstart):
    for k in range(1,D):   # k index -> E_{k+1}
        l=np.arange(N)
        if t%3==0 or NS<1:
            cc=np.where(l<D,1.0/D,0.0)+0.1*NS*rng.normal(size=N)
            x0=np.concatenate([cc,0.1*NS*rng.normal(size=N),np.tile((2*np.pi*l/D)[:,None],(1,n)).ravel()+0.5*NS*rng.normal(size=N*n)])
        elif t%3==1:
            x0=np.concatenate([rng.normal(size=2*N),rng.uniform(0,2*np.pi,N*n)])
        else: # aligned-ish positive amplitudes, random phases
            x0=np.concatenate([np.abs(rng.normal(size=N)),0.2*rng.normal(size=N),rng.uniform(0,2*np.pi,N*n)])
        res=minimize(obj,x0,args=(k,),method='Nelder-Mead',options=dict(maxiter=5000,maxfev=5000,xatol=1e-9,fatol=1e-12))
        if res.fun<best[k]: best[k]=res.fun; rec[k]=res.x.tolist()
print(f"n={n} N={N} ver={ver}: min E_k(M+)/(C(D,k) m^k), k=2..D: {np.round(best[1:],6)}",flush=True)
json.dump(dict(n=n,N=N,ver=ver,best=best[1:].tolist(),rec=rec[1:]),open(f'FINAL_RUNS/ARCH/eplus_n{n}_N{N}_{ver}_s{seed}_ns{NS}.json','w'))
