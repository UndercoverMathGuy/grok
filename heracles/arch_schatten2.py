"""arch_schatten2.py -- stronger adversary for (S) with FREE mirror completion U.
n=1: closed-form rank-one family; n=2: multi-scale starts incl. rank-one/rank-two completions, Powell+NM."""
import numpy as np, json
from scipy.optimize import minimize
rng=np.random.default_rng(1)
TH=np.exp(2j*np.pi*np.arange(2048)/2048)
def m_n2(C1,C2,C12): return (1+np.real(C1*TH)-np.abs(C2+C12*TH)).min()
def ratio_n2(x,r):
    C1,C2,C12,U1,U2,U12=x[0::2]+1j*x[1::2]
    m=m_n2(C1,C2,C12)
    if m<=1e-3: return 1e3+abs(m)
    M=np.array([[1,U1,U12],[C1,1,U2],[C12,C2,1]])
    s=np.linalg.svd(M,compute_uv=False)
    return (s**r).sum()/(3*m**r)
res_all={}
# n=1 closed form: M=[[1,U],[C,1]]
for r in [0.3,0.4,0.44,0.5,2/3]:
    best=9
    for x in np.linspace(0.01,0.95,95):
        for U in np.concatenate([np.linspace(-6,6,241)]):
            s=np.linalg.svd(np.array([[1,U],[x,1]]),compute_uv=False)
            best=min(best,(s**r).sum()/(2*(1-x)**r))
    print(f"n=1 r={r:.3f} grid min ratio {best:.5f}")
    res_all[f"n1_r{r:.3f}"]=best
for r in [0.4,0.5,0.6,2/3,0.8]:
    best=9;arg=None
    for t in range(150):
        kind=t%3
        L=(rng.normal(size=3)+1j*rng.normal(size=3))*rng.choice([0.1,0.3,0.6])
        if kind==0: U=(rng.normal(size=3)+1j*rng.normal(size=3))*rng.choice([0.3,1,3])
        elif kind==1: # rank-one completion a_k/a_k'
            a=np.array([1,L[0],L[2]]); L[1]=L[2]/L[0]; U=np.array([a[0]/a[1],a[1]/a[2],a[0]/a[2]])
        else: U=np.conj(L)*rng.choice([-1,1,2])
        x0=np.empty(12); x0[0::2]=np.r_[L,U].real; x0[1::2]=np.r_[L,U].imag
        r1=minimize(ratio_n2,x0,args=(r,),method='Powell',options=dict(xtol=1e-9,ftol=1e-12,maxiter=6000))
        r2=minimize(ratio_n2,r1.x,args=(r,),method='Nelder-Mead',options=dict(xatol=1e-10,fatol=1e-13,maxiter=8000,maxfev=8000))
        if r2.fun<best: best=r2.fun;arg=r2.x.tolist()
    x=np.array(arg);C=x[0::2]+1j*x[1::2]
    M=np.array([[1,C[3],C[5]],[C[0],1,C[4]],[C[2],C[1],1]])
    print(f"n=2 r={r:.3f} min ratio {best:.6f}  |L|={np.abs(C[:3]).round(3)} |U|={np.abs(C[3:]).round(3)} sv={np.linalg.svd(M,compute_uv=False).round(3)} m={m_n2(*C[:3]):.4f}",flush=True)
    res_all[f"n2_r{r:.3f}"]=dict(min=best,arg=arg)
json.dump(res_all,open('FINAL_RUNS/ARCH/schatten_free_U_v2.json','w'),indent=1)
