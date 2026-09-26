"""arch_schatten.py -- test (S): for n=2, M = I + L + U (L = junk C1,C2,C12 below the diagonal,
U = arbitrary completion above), is  sum_j sigma_j(M)^r >= 3 * m(L)^r ,  m = min_{T^2} Re h ?
Also n=1. Adversarial: minimise the ratio over (L,U). fp64, local CPU."""
import numpy as np, sys, json
from scipy.optimize import minimize
rng=np.random.default_rng(0)
TH=np.exp(2j*np.pi*np.arange(4096)/4096)
def m_n2(C1,C2,C12):
    f=1+np.real(C1*TH)-np.abs(C2+C12*TH)
    return f.min()
def ratio_n2(x,r):
    C1,C2,C12,U1,U2,U12=x[0::2]+1j*x[1::2]
    m=m_n2(C1,C2,C12)
    if m<=1e-3: return 1e3
    M=np.array([[1,U1,U12],[C1,1,U2],[C12,C2,1]])
    s=np.linalg.svd(M,compute_uv=False)
    return (s**r).sum()/(3*m**r)
def ratio_n1(x,r):
    C=x[0]+1j*x[1];U=x[2]+1j*x[3]
    m=1-abs(C)
    if m<=1e-3: return 1e3
    s=np.linalg.svd(np.array([[1,U],[C,1]]),compute_uv=False)
    return (s**r).sum()/(2*m**r)
out={}
for r in [0.4,0.5,0.6,2/3,0.8,0.9]:
    best1=1e9;best2=1e9;arg2=None
    for t in range(60):
        x0=rng.normal(size=4)*0.5
        res=minimize(ratio_n1,x0,args=(r,),method='Nelder-Mead',options=dict(xatol=1e-10,fatol=1e-13,maxiter=4000))
        best1=min(best1,res.fun)
        x0=rng.normal(size=12)*rng.choice([0.05,0.3,1.0])
        res=minimize(ratio_n2,x0,args=(r,),method='Nelder-Mead',options=dict(xatol=1e-10,fatol=1e-13,maxiter=20000,maxfev=20000))
        if res.fun<best2: best2=res.fun;arg2=res.x.tolist()
    print(f"r={r:.3f}  n=1 min ratio {best1:.6f}   n=2 min ratio {best2:.6f}",flush=True)
    out[str(r)]=dict(n1=best1,n2=best2,arg2=arg2)
json.dump(out,open('/Users/ruhaanrajadhyaksha/projects/grok/heracles/FINAL_RUNS/ARCH/schatten_free_U.json','w'),indent=1)
