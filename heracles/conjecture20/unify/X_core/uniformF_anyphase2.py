# RT_n for beta uniform on F with ANY phases: need rho(s,p) >= T(s)(1 - sqrt2/sqrt p) (depth >= L2 = sqrt2 in Legendre units).
# Lower bounds on rho: sine-weighted QR (p=3 mod 4, explicit analytic bound (L)), QR half (p=1 mod 4: 2^{s-1}(1-1/sqrt p)),
# single frequency (4 sin^2(pi/p) F^s / p).  Report the worst max-of-bounds / need over real p in [29,1e9] and s>1.
import numpy as np, math
from math import gamma, lgamma, cos, pi, sqrt
def sigma(r): return math.exp(lgamma((r+1)/2)-lgamma(r/2+1))/math.sqrt(pi)
def b1(r): return 2*gamma(r+1)/(2**r*gamma((3+r)/2)*gamma((1+r)/2))
def k(r):
    if r>=1: return 1.0
    c=2*gamma(r+1)*cos(pi*r/2)/(pi*2**r); return b1(r)+c*2**(1+r)*((3-r)**(-1-r)+(3-r)**(-r)/(2*r))
def T(s):
    nmin=max(2,math.ceil(2*s-1-1e-12)); return (1+1/nmin)**(s-1)
worst=(9,None); fails=0
for p in np.unique(np.concatenate([np.arange(29,3000),np.geomspace(3000,1e9,3000)])):
    F=(p-1)/2; S1=0.5/math.tan(pi/(2*p)); x=1/math.sqrt(p)
    for s in np.concatenate([1+np.linspace(0,60,241)[1:]*x, np.linspace(1,8,281)[1:]]):
        r=1/s; need=T(s)*(1-sqrt(2)*x)
        if need<=1: continue
        sq=(2*S1/p)*2**(s-1)*(F/((F+0.5)*sigma(r)+1+k(r)/(2*x)))**s          # p=3 mod 4 construction
        single=4*math.sin(pi/p)**2*F**s/p
        best=max(sq,single)
        if best/need<worst[0]: worst=(best/need,(float(p),round(float(s),4)))
        if best<need: fails+=1
print("worst max(sine-QR, single freq)/need:",worst,"  #grid failures:",fails)
# where are the failures?
fp=[]
for p in np.arange(29,3000):
    F=(p-1)/2; S1=0.5/math.tan(pi/(2*p)); x=1/math.sqrt(p); bad=[]
    for s in np.linspace(1,8,701)[1:]:
        r=1/s; need=T(s)*(1-sqrt(2)*x)
        if need<=1: continue
        sq=(2*S1/p)*2**(s-1)*(F/((F+0.5)*sigma(r)+1+k(r)/(2*x)))**s
        single=4*math.sin(pi/p)**2*F**s/p
        if max(sq,single)<need: bad.append(round(float(s),3))
    if bad: fp.append((int(p),min(bad),max(bad)))
print("failing p (real p grid, integer) with s-range:",fp[:5],"...",fp[-3:],"  count",len(fp))
