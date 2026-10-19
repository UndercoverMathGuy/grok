# Legendre family at p = 3 mod 4: need rho(s,p) := Gamma_s/U_s >= T(s)(1-2/sqrt p) for all s,
# T(s) = (1+1/nmin)^{s-1}, nmin = max(2, ceil(2s-1)).
# Lower-bound rho by explicit allocations y_l = 1 + theta*eta_l on F (theta optimised in [0,1]):
#   (i)  Legendre-sine pattern  eta_l = -chi(l) sin(2 pi l w0/p)  (exactly flat cosine transform)
#   (ii) Rudin-Shapiro halving  eta_l = rs_l (+-1), both global signs tried
# rho(y) = [min_delta sum_l y_l kappa_l(delta) / (p/2)] * F^s / (sum_l y_l^r)^s   (U_s = (p/2) F^{-s}).
import numpy as np, math, sys
def chi_table(p):
    t=np.full(p,-1.0); t[0]=0
    for a in range(1,p): t[(a*a)%p]=1
    return t
def rs(N):
    a=np.ones(1); b=np.ones(1)
    while len(a)<N: a,b=np.concatenate([a,b]),np.concatenate([a,-b])
    return a[:N]
def mincov(y,p):
    # sum_{l=1}^F y_l (1-cos(2 pi l d/p)) for d=1..p-1 via FFT
    F=len(y); v=np.zeros(p); v[1:F+1]=y
    c=np.real(np.fft.fft(v))      # sum_l y_l cos(2 pi l d / p)
    return (y.sum()-c[1:]).min()
def T(s):
    nmin=max(2,math.ceil(2*s-1-1e-12)); return (1+1/nmin)**(s-1)
def rho(y,p,s):
    F=len(y); r=1/s
    return mincov(y,p)/(p/2)*F**s/np.sum(y**r)**s
def best_theta(eta,p,s):
    best=1.0
    for th in np.linspace(0,1,41)[1:]:
        best=max(best,rho(1+th*eta,p,s))
    return best
ps=[int(x) for x in sys.argv[1:]] or [1019,2027,4079,10007]
for p in ps:
    F=(p-1)//2; chi=chi_table(p); l=np.arange(1,F+1)
    pats={}
    pats['legsin']=-chi[l]*np.sin(2*np.pi*l*1/p)
    r_=rs(F); pats['rs+']=r_; pats['rs-']=-r_
    worst=(9,None)
    for s in np.concatenate([1+np.linspace(0.5,40,80)/np.sqrt(p),[1.2,1.3,1.5,1.75,2,2.5,3,4,5]]):
        need=T(s)*(1-2/np.sqrt(p))
        if need<=1: continue
        vals={k:best_theta(e,p,s) for k,e in pats.items()}
        k=max(vals,key=vals.get); ratio=vals[k]/need
        if ratio<worst[0]: worst=(ratio,(round(s,5),round((s-1)*np.sqrt(p),2),k,{kk:round(vv,5) for kk,vv in vals.items()},round(need,5)))
    print(p, 'p%4=',p%4,' worst ratio (best construction / need) =',round(worst[0],5),' at',worst[1])
