# Fast sharp-bonus solver (analytic gradients).  n=2 atoms beta_i = amp_i e^{i ph_i}, xi_i = (e^{i t1_i}, e^{i t2_i}).
# Gstar(r1,r2,r12) = sup { Re A - 1 : 3^{1-s} Q^s = 1, |beta_1|<=r1, |beta_2|<=r2, |beta_12|<=r12 }   (a = 3^{1-s}Q^s = 1)
import numpy as np
from scipy.optimize import minimize
def funcs(s,N,rb):
    r=1/s; rb=np.asarray(rb,float)
    def sp(v): return v[:N],v[N:2*N],v[2*N:3*N],v[3*N:4*N]
    def obj(v):
        am,ph,t1,t2=sp(v); return -np.sum(am*np.cos(ph))
    def jobj(v):
        am,ph,t1,t2=sp(v); return np.concatenate([-np.cos(ph),am*np.sin(ph),np.zeros(2*N)])
    def eq(v):
        am=np.abs(v[:N])+1e-300; Q=np.sum(am**r); return np.array([3**(1-s)*Q**s-1])
    def jeq(v):
        am=np.abs(v[:N])+1e-300; Q=np.sum(am**r)
        return np.concatenate([3**(1-s)*Q**(s-1)*am**(r-1)*np.sign(v[:N]+1e-300),np.zeros(3*N)])[None,:]
    def ineq(v):
        am,ph,t1,t2=sp(v)
        out=[]
        for th in (t1,t2,t1+t2):
            B=np.sum(am*np.exp(1j*(ph+th))); out.append(abs(B)**2)
        return rb**2-np.array(out)
    def jineq(v):
        am,ph,t1,t2=sp(v); J=np.zeros((3,4*N))
        for k,(th,m1,m2) in enumerate(((t1,1,0),(t2,0,1),(t1+t2,1,1))):
            e=np.exp(1j*(ph+th)); B=np.sum(am*e)
            dam=2*np.real(np.conj(B)*e); dph=2*np.real(np.conj(B)*1j*am*e)
            J[k,:N]=-dam; J[k,N:2*N]=-dph; J[k,2*N:3*N]=-dph*m1; J[k,3*N:]=-dph*m2
        return J
    return obj,jobj,eq,jeq,ineq,jineq
def opt(s,N,rb,v0,maxiter=500):
    obj,jobj,eq,jeq,ineq,jineq=funcs(s,N,rb)
    try:
        res=minimize(obj,v0,jac=jobj,constraints=[{'type':'eq','fun':eq,'jac':jeq},{'type':'ineq','fun':ineq,'jac':jineq}],
                     method='SLSQP',options={'maxiter':maxiter,'ftol':1e-15})
    except Exception: return -np.inf,v0
    v=res.x
    if np.all(ineq(v)>-1e-10) and abs(eq(v)[0])<1e-10: return -obj(v)-1,v
    return -np.inf,v
def norm(s,N,v):
    r=1/s; v=v.copy(); v[:N]=np.abs(v[:N]); Q=np.sum(v[:N]**r); v[:N]=v[:N]/(3**(1-s)*Q**s); return v
def ifblock(s,N,rng,eps):
    amp=np.full(N,1e-3); amp[:min(3,N)]=1.0
    v=np.concatenate([amp,np.zeros(N),2*np.pi*np.arange(N)/3,2*np.pi*np.arange(N)/3])+eps*rng.standard_normal(4*N)
    return norm(s,N,v)
def rand(s,N,rng):
    return norm(s,N,np.concatenate([rng.uniform(0.05,1,N),rng.normal(0,0.3,N),rng.uniform(0,2*np.pi,2*N)]))
def gstar(s,rb,rng,warm=None,Ns=(1,2,3,4),trials=6):
    best=(-np.inf,None,None)
    for N in Ns:
        starts=[w for (n_,w) in (warm or []) if n_==N]
        starts+=[ifblock(s,N,rng,0.02+0.4*rng.random()) for _ in range(trials)]+[rand(s,N,rng) for _ in range(trials)]
        for v0 in starts:
            g,v=opt(s,N,rb,v0)
            if g>best[0]: best=(g,N,v)
    return best
