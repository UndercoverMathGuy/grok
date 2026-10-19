# Core quantities for the relations case (n=2, T={1,2}); kappa_l(delta)=1-cos(2 pi l delta/p), s_l=|sin(pi l delta/p)|.
import numpy as np, json
from scipy.special import i0e, i1e
S15=1.5; TAU=3**0.5; CA=1/(2*np.sqrt(3)); C12=1-(2/3)**0.5
def load(fn='supports.json'):
    return {int(k):v for k,v in json.load(open(fn)).items()}
def kapS(p,S):
    d=np.arange(1,(p-1)//2+1); return 1-np.cos(2*np.pi*np.outer(d,np.asarray(S))/p)
def binding(p,S,y,tol=1e-6):
    cov=kapS(p,S)@np.asarray(y); m=cov.min(); return np.nonzero(cov<=m*(1+tol))[0]+1, cov
def cosmat(p,S):
    # rows delta' = 0..p-1
    return np.cos(2*np.pi*np.outer(np.arange(p),np.asarray(S))/p)
A=lambda x: i1e(x)/i0e(x)
def tilt_terms(b,C,lam):
    """returns (Pi'/Pi, Z/Pi, Z'/Pi) at lam (vector lam allowed), using scaled exponentials."""
    b=np.asarray(b,float); lam=np.atleast_1d(lam)[:,None]
    x=lam*b[None,:]                                  # nl x K
    logPi=np.sum(np.log(i0e(x))+x,1)                 # log Pi
    dlogPi=np.sum(b[None,:]*A(x),1)                  # Pi'/Pi
    u=C@b                                            # p
    E=lam*u[None,:]                                  # nl x p
    mx=E.max(1,keepdims=True)
    w=np.exp(E-mx); Zs=w.mean(1); Zps=(w*u[None,:]).mean(1)
    lZ=np.log(Zs)+mx[:,0]
    ZoverPi=np.exp(lZ-logPi); ZpoverPi=Zps/Zs*ZoverPi
    return dlogPi, ZoverPi, ZpoverPi, logPi, lZ
def tilted_mean_bound(b,C,lams=None):
    """sup_lam (2Pi'-Z')/Z  (lower bound for min(max P,-min P), any phases)."""
    b=np.asarray(b,float)
    if lams is None: lams=np.exp(np.linspace(np.log(0.05),np.log(40),120))/max(b.max(),1e-300)
    d,ZoP,ZpoP,_,_=tilt_terms(b,C,lams)
    val=(2*d - ZpoP)/ZoP
    i=np.argmax(val); return max(val[i],0.0), lams[i]
def log_tilt_bound(b,C,lams=None):
    b=np.asarray(b,float)
    if lams is None: lams=np.exp(np.linspace(np.log(0.05),np.log(40),120))/max(b.max(),1e-300)
    _,ZoP,_,logPi,_=tilt_terms(b,C,lams)
    inner=2-ZoP
    val=np.where(inner>0,(logPi+np.log(np.clip(inner,1e-300,None)))/lams,-np.inf)
    i=np.argmax(val); return max(val[i],0.0), lams[i]
def relation_split(x,C,p):
    """rho0 = e^{E}/p (j=0 term), eps = e^{E}(1/p)sum_{j!=0} e^{-(Kx)_j} - 1 (excess over independent phases); rho=rho0+eps."""
    x=np.asarray(x,float); logPi=np.sum(np.log(i0e(x))+x); X=x.sum(); E=X-logPi
    u=C[1:]@x; R=np.exp(u-X).sum()/C.shape[0]          # R e^{-X}
    rho0=np.exp(E)/p; eps=R*np.exp(X-logPi)-1          # rho = Z/Pi-1 = rho0 + eps exactly
    return rho0, eps, E
