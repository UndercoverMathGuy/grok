# Is (L) at n=2 true COEFFICIENTWISE?  With x_l=|c_l|, m=min_{T^2} Re P:  e2(x) >= m^2/3 ,  e3(x) >= m^3/27 ?
# (sufficient for (L) via the Bernstein integral; ELEGANT's junk-free proof is coefficientwise)
import numpy as np
from scipy.optimize import minimize
rng=np.random.default_rng(1)
G=48; th=np.linspace(0,2*np.pi,G,endpoint=False); S,T=np.meshgrid(np.exp(1j*th),np.exp(1j*th),indexing='ij'); S=S.ravel(); T=T.ravel()
def parts(z,N):
    c=z[:N]*np.exp(1j*z[N:2*N]); v=np.exp(1j*z[2*N:3*N]); w=np.exp(1j*z[3*N:4*N])
    P=((1+np.outer(S,v))*(1+np.outer(T,w))*c).sum(1).real
    x=np.abs(c); e1=x.sum(); e2=(e1**2-(x**2).sum())/2
    e3=(e1**3-3*e1*(x**2).sum()+2*(x**3).sum())/6
    return P,e1,e2,e3
def ratio(z,N,k,soft):
    P,e1,e2,e3=parts(z,N)
    m=-soft*np.log(np.mean(np.exp(-(P-P.min())/soft)))+P.min() if soft else P.min()
    if m<=0: return 0.0
    return {2:3*e2/m**2, 3:27*e3/m**3}[k]     # coefficientwise claim: ratio >= 1
for k in (2,3):
    for N in (3,4,5,6):
        worst=np.inf
        for t in range(25):
            # feasible start: the 3-atom block (cube-root phases) plus N-3 small atoms, randomly perturbed
            base_c=np.r_[np.ones(3)/3,0.05*rng.random(N-3)]; om=2*np.pi*np.arange(N)/3
            z=np.concatenate([base_c*(1+0.3*rng.standard_normal(N)).clip(0.05),0.3*rng.standard_normal(N),
                              om+0.4*rng.standard_normal(N),om+0.4*rng.standard_normal(N)])
            for s in (0.05,0.01,0.002):
                z=minimize(lambda z:ratio(z,N,k,s) if ratio(z,N,k,s)>0 else 1e3,z,method='Nelder-Mead',options={'maxiter':4000,'xatol':1e-9,'fatol':1e-12}).x
            r=ratio(z,N,k,None)
            if r>0: worst=min(worst,r)
        print(f"e{k}: N={N}: min over search of C(3,{k})-normalised e{k}/m^{k} = {worst:.5f}  ({'VIOLATED' if worst<1-1e-6 else 'holds'})",flush=True)
