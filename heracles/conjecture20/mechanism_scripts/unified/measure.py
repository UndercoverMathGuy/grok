# Measurement of the global ratio of unified.tex (Conjecture "no unicorns", class C, n=2, b=2, e=1).
#   rho = min over interference families of  Det / Pay,
#   Det = (1/2)(-min_x F_{d0}(x) - min_x F_{-d0}(x)),
#   F_d(x) = sum_{l,T} v_{l,T} |c_l(d)| cos(2pi*2l*s_T(x)/p + psi_{l,T} + arg c_l(d)),
#   Pay = sum_l kappa_l(d0) * Pi(v_{l,.}).
# line geometry (one slot-set T): F depends on w = s_T(x) in Z_p.   Pi = v_l; report rho_line, then R = rho_line / c.
# plane geometry (n=2, T in {1},{2},{1,2}): x in Z_p^2.   Pi = min(c1 max_T v, ||v||_2/(2 sqrt3)) (both admissible).
# The prover's offset d0 is fixed per run (helps the network), so each value is a lower bound for the restricted sigma*.
import numpy as np, json, sys, math
from scipy.optimize import minimize
from scipy.special import logsumexp
s=1.5; c1=1-3**(1-s); c12=1-(2/3)**(s-1); cA=1/(2*math.sqrt(3))
rng=np.random.default_rng(int(sys.argv[3]) if len(sys.argv)>3 else 0)
def kap(l,d,p): return 1-np.cos(2*np.pi*np.outer(l,d)/p) if np.ndim(d) else 1-np.cos(2*np.pi*np.asarray(l)*d/p)
def cvals(S,d,p):
    c=1-np.exp(-2j*np.pi*np.asarray(S)*d/p); return np.abs(c), np.angle(c)
def binding(p,S,y,tol=1e-7):
    d=np.arange(1,p); cov=(np.asarray(y)[:,None]*(1-np.cos(2*np.pi*np.outer(S,d)/p))).sum(0)
    return d[cov<=cov.min()+tol*abs(cov.min())]
def softmin(F,tau):  # smooth min and its weights
    m=-logsumexp(-tau*F)/tau; wts=np.exp(-tau*(F-F.min())); wts/=wts.sum(); return m,wts
class Problem:
    def __init__(s_,p,S,d0,geom):
        s_.p,s_.S,s_.d0,s_.geom=p,np.asarray(S),d0,geom
        K=len(S)
        if geom=='line':
            w=np.arange(p); s_.phase=[2*np.pi*np.outer(w,2*s_.S)/p]           # (p,K)
            s_.T=1
        else:
            x1,x2=np.meshgrid(np.arange(p),np.arange(p),indexing='ij'); x1=x1.ravel(); x2=x2.ravel()
            s_.phase=[2*np.pi*np.outer(x1,2*s_.S)/p, 2*np.pi*np.outer(x2,2*s_.S)/p, 2*np.pi*np.outer((x1+x2)%p,2*s_.S)/p]
            s_.T=3
        s_.kap=1-np.cos(2*np.pi*s_.S*d0/p)
        s_.cabs,s_.carg=cvals(S,d0,p)
    def det_and_grad(s_,v,psi,tau):
        # v,psi: (T,K).  returns Det (smoothed or exact if tau None) and grads
        tot=0; gv=np.zeros_like(v); gp=np.zeros_like(psi)
        for sign in (+1,-1):
            arg=[s_.phase[t]+psi[t]+sign*s_.carg for t in range(s_.T)]
            F=sum((np.cos(arg[t])*(v[t]*s_.cabs)).sum(1) for t in range(s_.T))
            if tau is None: tot+=-0.5*F.min(); continue
            m,wts=softmin(F,tau); tot+=-0.5*m
            for t in range(s_.T):
                gv[t]+=-0.5*(wts@np.cos(arg[t]))*s_.cabs
                gp[t]+=-0.5*(wts@(-np.sin(arg[t])))*v[t]*s_.cabs
        return tot,gv,gp
    def pay_and_grad(s_,v,mode):
        if s_.geom=='line':
            return (s_.kap*v[0]).sum(), np.array([s_.kap])
        # plane: Pi = min(c1*max_T v, ||v||/(2sqrt3)), smoothed max
        P=0; g=np.zeros_like(v)
        for k in range(v.shape[1]):
            vk=v[:,k]; mx=vk.max(); nrm=np.sqrt((vk**2).sum())+1e-300
            if c1*mx<=cA*nrm:
                j=vk.argmax(); P+=s_.kap[k]*c1*mx; g[j,k]+=s_.kap[k]*c1
            else:
                P+=s_.kap[k]*cA*nrm; g[:,k]+=s_.kap[k]*cA*vk/nrm
        return P,g
    def ratio(s_,z,tau):
        T,K=s_.T,len(s_.S); v=np.exp(z[:T*K]).reshape(T,K); psi=z[T*K:].reshape(T,K)
        D,gv,gp=s_.det_and_grad(v,psi,tau); P,gP=s_.pay_and_grad(v,None)
        R=D/P; gRv=(gv*P-D*gP)/P**2
        return R, np.concatenate([(gRv*v).ravel(), (gp/P).ravel()])
    def exact(s_,z):
        T,K=s_.T,len(s_.S); v=np.exp(z[:T*K]).reshape(T,K); psi=z[T*K:].reshape(T,K)
        D,_,_=s_.det_and_grad(v,psi,None); P,_=s_.pay_and_grad(v,None); return D/P,v,psi
def run(pr,starts):
    T,K=pr.T,len(pr.S); best=(np.inf,None)
    for t in range(starts):
        kind=t%3
        lv=np.zeros((T,K)) if kind==0 else rng.normal(0,1.0,(T,K))
        if kind==2:  # random subset only
            mask=rng.random((T,K))<0.5; lv[~mask]=-8
        z=np.concatenate([lv.ravel(), rng.uniform(0,2*np.pi,T*K)])
        for tau in [3,10,30,100,300]:
            scale=tau/ max(1e-9,np.exp(z[:T*K]).sum()) if False else tau
            r=minimize(lambda zz: pr.ratio(zz,tau/np.exp(zz[:T*K]).mean()),z,jac=True,method='L-BFGS-B',
                       bounds=[(-12,3)]*(T*K)+[(None,None)]*(T*K),options=dict(maxiter=300))
            z=r.x
        R,v,psi=pr.exact(z)
        if R<best[0]: best=(R,(v,psi))
    return best
if __name__=="__main__":
    geom=sys.argv[1]; which=sys.argv[2]
    sup=json.load(open('../relations3/supports.json'))
    if which=='small': ps=[101,151,211,307,401,503,601,701,809,1009] if geom=='line' else [53,101,151,211]
    else: ps=[int(which)]
    for p in ps:
        if str(p) in sup: S,y=sup[str(p)]['S'],sup[str(p)]['y']
        else:
            runs=json.load(open(f'../relations3/large_{p}.json'))
            r0=max(runs['runs'],key=lambda r:r['Gamma']) if 'runs' in runs else list(runs.values())[0]
            S,y=r0['S'],r0['y']
        B=binding(p,S,y); out=[]
        for d0 in B[:2]:
            pr=Problem(p,S,int(d0),geom)
            R,(v,psi)=run(pr, 24 if geom=='line' else 9)
            out.append((int(d0),R))
        rs=[r for _,r in out]
        if geom=='line':
            print(f"p={p} K={len(S)} line: rho_line(min over d0)={min(rs):.3f} (max over d0 {max(rs):.3f})  "
                  f"-> R with c12: {min(rs)/c12:.2f}, with Theorem-A 0.289: {min(rs)/cA:.2f}, with c1: {min(rs)/c1:.2f}",flush=True)
        else:
            print(f"p={p} K={len(S)} plane: R=Det/Pay with Pi=min(c1 max,||.||/2sqrt3): min over d0 {min(rs):.3f} (max {max(rs):.3f})",flush=True)
