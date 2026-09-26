"""End-to-end soundness test of the certificate as a LOWER BOUND on the true cost (n=2):
for a population with clean modulus a0, S=sum|c|, m_A:  d=1-m_A/a0, S'=S/a0,
   cost/a0^r >= max(chain(d,S'), peel(d,S'), S'^r).   Also each tool separately. Includes a Nelder-Mead adversary
that MINIMISES cost/bound for each tool."""
import numpy as np, sys, json
from scipy.optimize import minimize
A=0.5; r=2/3; rng=np.random.default_rng(int(sys.argv[1]) if len(sys.argv)>1 else 0)
def LB(S,e2,e3):
    e2=max(e2,0.0); e3=max(e3,0.0); co=[e3,e2,S,1.0]
    while co[0]==0: co=co[1:]
    y=-1/np.roots(co); return float(np.sum((y.astype(complex)**r).real))
def tools(d,S):
    mu=1-d; w=np.sqrt(max(S*S-1,0)); D1=d/A; D12=d/A**2; u1=min(S,D1+2*w); u12=min(S,D12+2*w)
    E2=3-2*D1*u1-D12*u12; E3=1-2*D1*u1-D12*u12-D1*D1*u12-u1*u1*D12
    b1=(mu-S*(1-A)**2)/((1+A)**2-(1-A)**2); b2=(mu-S*(1-A*A))/((1+A)**2-(1-A*A))
    b1=min(max(b1,0),S); b2=min(max(b2,0),b1)
    return LB(S,E2/9,E3/27),(S-b1)**r+(b1-b2)**r+b2**r,S**r
ph=np.exp(1j*np.linspace(0,2*np.pi,4001))*A
def stats(z,N):
    c=z[:N]+1j*z[N:2*N]; v=np.exp(1j*z[2*N:].reshape(N,2))
    C0=c.sum(); C1=(c*v[:,0]).sum(); C2=(c*v[:,1]).sum(); C12=(c*v[:,0]*v[:,1]).sum()
    m=((C0+C2*ph).real-A*np.abs(C1+C12*ph)).min()
    return abs(C0),np.abs(c).sum(),m,(np.abs(c)**r).sum()
def ratios(z,N):
    a0,S,m,cost=stats(z,N)
    if m<=0.05*a0: return None
    d=max(1-m/a0,0); t=tools(d,S/a0); return [cost/a0**r/x for x in t]
worst=[9,9,9]
def blockz(N,eps):
    w3=2*np.pi*np.arange(3)/3
    c=np.concatenate([np.ones(3)/3,np.zeros(N-3)]); th=np.concatenate([np.stack([w3,w3],1),2*np.pi*rng.random((N-3,2))])
    z=np.concatenate([c,np.zeros(N),th.ravel()]); return z+eps*rng.normal(size=z.size)
for trial in range(300):
    N=int(rng.integers(3,7)); z0=blockz(N,10**rng.uniform(-2.5,-0.2))
    for tool in range(3):
        f=lambda z:(lambda q: 9 if q is None else q[tool])(ratios(z,N))
        res=minimize(f,z0,method='Nelder-Mead',options=dict(maxiter=1500,xatol=1e-9,fatol=1e-12))
        worst[tool]=min(worst[tool],res.fun)
print(dict(chain=worst[0],peel=worst[1],Sr=worst[2]))
json.dump(dict(chain=worst[0],peel=worst[1],Sr=worst[2]),open('FINAL_RUNS/DEPTHA/e2e_seed%s.json'%(sys.argv[1] if len(sys.argv)>1 else 0),'w'))
