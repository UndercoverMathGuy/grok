# Free-hiding criterion for a single cheap feature at n=2:
# V = min_{b,z,z'} max_{delta-pairs in B} [ (|e_d|+|e_-d|)/2 - lam*kappa_l(d) ],  e_d = (1/2)(c_l + b conj c_l + sum_j z_j c_j + z'_j conj c_j)
# V>=0  => no riding/partner compensation (cost ignored!) can rescue one cheap feature with gain lam*R*K_l(mu).
import sys, numpy as np, cvxpy as cp
sys.path.insert(0,__import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from progx import exhaustive
from prog2 import kappa
s=float(sys.argv[1]) if len(sys.argv)>1 else 1.5
c1=1-3**(1-s)
primes=[int(a) for a in sys.argv[2].split(',')] if len(sys.argv)>2 else [5,7,11,13,17,19,23,29,31]
for p in primes:
    G,S,y=exhaustive(p,s,4 if p<=31 else 3)
    K=kappa(p); F=(p-1)//2
    cov=K[:, [j-1 for j in S]]@y
    B=[d+1 for d in range(F) if cov[d]<=G*(1+1e-7)]
    rho=np.exp(2j*np.pi/p)
    c=lambda j,d: 1-rho**(-j*d)
    out=[]
    for l in S:
        others=[j for j in S if j!=l]
        res={}
        for lam,name in [(0.5,'1/2'),(c1,'c1')]:
            b=cp.Variable(complex=True); z=cp.Variable(len(others),complex=True); zp=cp.Variable(len(others),complex=True); t=cp.Variable()
            cons=[]
            for d in B:
                e=[]
                for dd in (d,-d):
                    expr=c(l,dd)+b*np.conj(c(l,dd))
                    if others:
                        expr=expr+cp.sum(cp.multiply(np.array([c(j,dd) for j in others]),z))+cp.sum(cp.multiply(np.array([np.conj(c(j,dd)) for j in others]),zp))
                    e.append(0.5*expr)
                kl=1-np.cos(2*np.pi*l*d/p)
                cons.append(0.5*(cp.abs(e[0])+cp.abs(e[1]))-lam*kl<=t)
            pr=cp.Problem(cp.Minimize(t),cons); pr.solve(solver='CLARABEL')
            res[name]=t.value
        out.append(f"l={l}: V(1/2)={res['1/2']:+.4f} V(c1)={res['c1']:+.4f}")
    print(f"p={p} S={S} |B|pairs={len(B)} B={B} :: "+"; ".join(out),flush=True)
