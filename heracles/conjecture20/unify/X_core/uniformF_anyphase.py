# Fully analytic version of the p = 3 mod 4 bound.
#  K(r) <= k(r) := |b_1(r)| + c(r) 2^{1+r} [ (3-r)^{-1-r} + (3-r)^{-r}/(2r) ],
#  c(r) = 2 Gamma(r+1) cos(pi r/2)/(pi 2^r),   b_1(r) = 2 Gamma(r+1)/(2^r Gamma((3+r)/2) Gamma((1+r)/2)),
#  (Wendel: Gamma(a)/Gamma(a+r+1) <= a^{-1-r} for a>=1, applied with a=(j-r)/2, j>=3 odd).
#  Main inequality, x = 1/sqrt p, F = (p-1)/2, S1 = cot(pi/(2p))/2:
#  (2 S1/p) 2^{s-1} F^s >= T(s)(1-2x) [ (F+1/2) sigma_r + 1 + k(r)/(2x) ]^s .
import numpy as np, math
from math import gamma, lgamma, cos, pi, sqrt
def sigma(r): return math.exp(lgamma((r+1)/2)-lgamma(r/2+1))/math.sqrt(pi)
def b1(r): return 2*gamma(r+1)/(2**r*gamma((3+r)/2)*gamma((1+r)/2))
def k(r):
    if r>=1: return 1.0
    c=2*gamma(r+1)*cos(pi*r/2)/(pi*2**r)
    return b1(r)+c*2**(1+r)*((3-r)**(-1-r)+(3-r)**(-r)/(2*r))
def T(s):
    nmin=max(2,math.ceil(2*s-1-1e-12)); return (1+1/nmin)**(s-1)
for r in [1/3,0.5,2/3,0.9,0.99]: print(f"r={r:.3f}  analytic k(r)={k(r):.4f}")
worst=(9,None)
# continuous in p (no arithmetic left): scan p from 29 to 1e9 geometrically, s in (1,3]
for p in np.unique(np.concatenate([np.arange(29,2000),np.geomspace(2000,1e9,4000)])):
    F=(p-1)/2; S1=0.5/math.tan(pi/(2*p)); x=1/math.sqrt(p)
    for s in np.concatenate([1+np.linspace(0,60,241)[1:]/math.sqrt(p), np.linspace(1,3,201)[1:]]):
        if s>3: continue
        r=1/s; need=T(s)*(1-math.sqrt(2)*x)
        if need<=1: continue
        lhs=(2*S1/p)*2**(s-1)*F**s
        rhs=need*((F+0.5)*sigma(r)+1+k(r)/(2*x))**s
        if lhs/rhs<worst[0]: worst=(lhs/rhs,(float(p),round(float(s),5),round(float((s-1)*math.sqrt(p)),3)))
print("[depth = sqrt2, i.e. ANY phases with beta uniform on F] worst lhs/rhs over real p in [29,1e9], s in (1,3]:",worst)
