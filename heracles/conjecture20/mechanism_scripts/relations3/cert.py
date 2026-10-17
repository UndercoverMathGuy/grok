# Certificates for the relations case at an allocation a (support S, weights y), s=1.5, tau=sqrt3.
#  e      = efficiency min(Ka)/(Gamma_ref ||a||)     (Gamma_ref = best known programme value: conservative)
#  CL2    = min_delta[(Ka)_delta + tau g_delta(a)]/(Gamma_ref||a||)   (water-filling Parseval criterion, Theorem A or P12)
#  sigP   = 1/sqrt(sum_S kappa(delta0))  (Parseval lower bound for the weighted Sidon constant)
#  sigT   = inf_b max(Pars(b),TiltMean(b))/sum b s   (certified weighted Sidon constant, numerically minimised over b)
#  sigV   = same at the vertex direction b ~ a s
#  req(c) = c - (1/e-1)/tau  (Theorem 1 needs sig >= req)
import numpy as np, sys, cma, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0,'.')
from core import cosmat, A, i0e, CA, C12, TAU
r=2/3; s=1.5
def kapS(p,S):
    d=np.arange(1,(p-1)//2+1); return 1-np.cos(2*np.pi*np.outer(d,np.asarray(S))/p)
def norm(y): return np.sum(np.asarray(y)**r)**s
def g_waterfill(a,K,c):
    """g_delta(a)=max_{0<=z<=a} c<z,kap>-sqrt(<z^2,kap>) for all rows of K (F x |S|)."""
    a=np.asarray(a,float); F=K.shape[0]; out=np.zeros(F); ic2=1/c**2
    tot=K.sum(1); act=tot>ic2
    if not act.any(): return out
    Ka=K[act]; lo=np.full(Ka.shape[0],1e-12*a.max()); hi=np.full(Ka.shape[0],a.max())
    for it in range(100):
        lam=np.sqrt(lo*hi); m=np.minimum(a[None,:]/lam[:,None],1); f=(Ka*m*m).sum(1)-ic2
        lo=np.where(f>0,lam,lo); hi=np.where(f>0,hi,lam)
    lam=np.sqrt(lo*hi); m=np.minimum(a[None,:]/lam[:,None],1)
    out[act]=np.maximum(c*lam*((Ka*m).sum(1)-ic2),0); return out
class Tilt:
    def __init__(s_,p,S,nl=40,binned=None):
        s_.C=cosmat(p,S); s_.p=p; s_.nl=nl; s_.binned=binned
    def T(s_,b):
        b=np.asarray(b,float); bm=b.max(); lams=np.exp(np.linspace(np.log(0.02),np.log(60),s_.nl))/bm
        x=lams[:,None]*b[None,:]
        logPi=np.sum(np.log(i0e(x))+x,1); dl=np.sum(b[None,:]*A(x),1)
        u=s_.C@b; U=u.sum() if False else b.sum()
        if s_.binned:
            h,ed=np.histogram(u,bins=s_.binned,range=(-U,U)); c_=(ed[1:]+ed[:-1])/2; nz=h>0; h=h[nz]; c_=c_[nz]
            E=lams[:,None]*(c_[None,:]-U); w=np.exp(E)*h[None,:]; Zs=w.sum(1)/s_.p; Zps=(w*c_[None,:]).sum(1)/s_.p
        else:
            E=lams[:,None]*(u[None,:]-U); w=np.exp(E); Zs=w.mean(1); Zps=(w*u[None,:]).mean(1)
        lZ=np.log(Zs)+lams*U
        ZoP=np.exp(lZ-logPi); ZpoP=Zps/Zs*ZoP
        val=(2*dl-ZpoP)/ZoP
        return max(val.max(),0)
def sig_inf(tl,sv,b0,evals=1500,seed=0):
    K=len(sv)
    def obj(z):
        b=np.exp(np.clip(z-z.max(),-25,0)); pay=(b*sv).sum()
        return max(np.sqrt((b**2).sum()/2),tl.T(b))/pay
    best=(obj(np.log(b0)),np.log(b0))
    rng=np.random.default_rng(seed)
    for x0 in [np.log(b0),np.zeros(K),rng.normal(0,1,K),np.log(sv+1e-3)*2]:
        es=cma.CMAEvolutionStrategy(x0,1.0,{'verbose':-9,'maxfevals':evals//4,'seed':int(rng.integers(1e9))}); es.optimize(obj)
        if es.result.fbest<best[0]: best=(es.result.fbest,es.result.xbest)
    return best
def certify(p,S,y,Gref,do_inf=True,binned=None,evals=1500):
    S=np.asarray(S); y=np.asarray(y,float); K=kapS(p,S); cov=K@y; n=norm(y)
    e=cov.min()/(Gref*n); B=np.nonzero(cov<=cov.min()*(1+1e-6))[0]
    res={'p':p,'K':len(S),'e':e}
    for nm,c in [('A',CA),('12',C12)]:
        res['CL2_'+nm]=np.min(cov+TAU*g_waterfill(y,K,c))/(Gref*n)
    # best binding delta for the Sidon criterion = smallest sum kappa
    d0=B[np.argmin(K[B].sum(1))]; sv=np.sqrt(K[d0]/2); res['sumk']=K[d0].sum(); res['sigP']=1/np.sqrt(K[d0].sum())
    tl=Tilt(p,S,binned=binned); b=y*sv; res['sigV']=max(np.sqrt((b**2).sum()/2),tl.T(b))/(b*sv).sum(); res['tiltV']=tl.T(b)/(b*sv).sum()
    if do_inf:
        f,z=sig_inf(tl,sv,b,evals=evals); 
        bb=np.exp(np.clip(z-z.max(),-25,0))
        if binned: tl2=Tilt(p,S); f=max(np.sqrt((bb**2).sum()/2),tl2.T(bb))/(bb*sv).sum()
        res['sigT']=f
    for nm,c in [('A',CA),('12',C12)]: res['req_'+nm]=c-(1/e-1)/TAU
    return res
if __name__=="__main__":
    from core import load
    D=load()
    print("  p   K    e     CL2_A  CL2_12  sumk  sigP  tiltV  sigV  sigT | reqA  req12")
    for p in sorted(D):
        R=certify(p,D[p]['S'],D[p]['y'],D[p]['Gamma'])
        print(f"{p:5d} {R['K']:3d} {R['e']:6.4f} {R['CL2_A']:6.4f} {R['CL2_12']:6.4f} {R['sumk']:5.2f} {R['sigP']:5.3f} {R['tiltV']:5.3f} {R['sigV']:5.3f} {R['sigT']:5.3f} | {R['req_A']:5.3f} {R['req_12']:5.3f}",flush=True)
