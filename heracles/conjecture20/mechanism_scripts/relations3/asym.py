# Asymptotic (p->oo, K = kap*ln p) model of the tilt certificate at a binding direction, random-like supports.
# kappa_l(delta0) iid with law  e^{-t k} * arcsine(k), k=1-cos U, t from the LD of the min over ~p directions:
#   I(gamma)=t(1-gamma)-log I0(t) = 1/kap, A(t)=1-gamma.  s_l=sqrt(k_l/2).  Uniform weights: b_l ~ s_l.
# Certified fraction (tilted mean, relation mass -> 0 when E <= ln p - C):  sup_x E[s A(x s)]/E[s^2]  s.t. E[g(x s)] <= 1/kap.
# Also: LD truth for max D (random phases heuristic) and the positive-definite product-test cap.
import numpy as np
from scipy.special import i0e,i1e
from scipy.optimize import brentq
A=lambda x: i1e(x)/i0e(x); lI0=lambda x: np.log(i0e(x))+x; g=lambda x: x-lI0(x)
U=(np.arange(200000)+0.5)/200000*np.pi; kU=1-np.cos(U)          # arcsine law of kappa
def tilted(t):
    w=np.exp(-t*kU); return w/w.sum()
def rate(gam):
    t=brentq(lambda t: A(t)-(1-gam),1e-12,2000); return t*(1-gam)-lI0(t), t
def law(kap):
    gam=brentq(lambda gm: rate(gm)[0]-1/kap,0.01,0.9999); _,t=rate(gam); return gam,tilted(t)
def cert(kap,wfun=lambda s: s):
    gam,w=law(kap); s=np.sqrt(kU/2); b=wfun(s)
    def Eg(x): return (w*g(x*b)).sum()
    xm=brentq(lambda x: Eg(x)-1/kap,1e-6,1e4)
    return gam, (w*b*A(xm*b)).sum()/(w*b*s).sum(), xm, (w*s).sum(), (w*s*s).sum()/(w*s).sum()
def ldtruth(kap,wfun=lambda s: s):
    # heuristic max of D for 'random' phases: y with Lambda*(y)=1/kap per unit, tilted family x: E[b x A(xb)] - E[log I0(xb)] = 1/kap
    gam,w=law(kap); s=np.sqrt(kU/2); b=wfun(s)
    f=lambda x: (w*(x*b*A(x*b)-lI0(x*b))).sum()-1/kap
    x=brentq(f,1e-6,1e4); return (w*b*A(x*b)).sum()/(w*b*s).sum()
if __name__=="__main__":
    print(" kap   gamma  sbar   | TiltM(b~s)  TiltM(b~1)  TiltM(b~s^3) | LDtruth | needed: cA*1=0.289, c12=0.1835")
    for kap in [1.0,1.25,1.5,1.75,2.0,2.5,3.0,3.5,4.0,5.0]:
        gam,c1,x,sbar,_=cert(kap); _,c2,_,_,_=cert(kap,lambda s: np.ones_like(s)); _,c3,_,_,_=cert(kap,lambda s: s**3)
        print(f"{kap:4.2f}  {gam:5.3f}  {sbar:5.3f}  |  {c1:6.3f}     {c2:6.3f}     {c3:6.3f}     | {ldtruth(kap):6.3f}")
