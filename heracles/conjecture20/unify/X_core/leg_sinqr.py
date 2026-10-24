# Legendre family, p = 3 mod 4: compression y_m = sin(2 pi m/p) (1 + chi(m)), m = 1..F.
# Exact facts (Gauss sum g = i sqrt p):
#   sum_m chi(m) sin(2 pi m k/p) = chi(k) sqrt(p)/2      (half range)
#   C1(delta) := sum_m sin_m cos(2 pi m delta/p) <= 0  for every delta  (closed form)
#   => coverage(delta) >= S1 := sum_m sin_m  for every delta.
# Cost: sum y^r = 2^r Q_r, Q_r = sum_{m<=F, chi(m)=1} sin_m^r = (P_r + B_r)/2,
#   P_r = sum sin_m^r <= (F+1/2) sigma_r + 1,  |B_r| = |sum chi(m) sin_m^r| <= (sqrt p/2) K(r),
#   K(r) = sum_{j odd} |b_j|, b_j = sine coefficients of sgn(sin)|sin|^r.
# Need rho >= T(s)(1-2/sqrt p) with rho >= (2 S1/p) 2^{s-1} (F/(P_r+B_r))^s.
import numpy as np, math
from math import gamma, lgamma, cos, pi, log, sqrt, exp
def chi_table(p):
    t=np.full(p,-1.0); t[0]=0
    for a in range(1,p): t[(a*a)%p]=1
    return t
def sigma(r): return math.exp(lgamma((r+1)/2)-lgamma(r/2+1))/math.sqrt(pi)   # E sin^r over a half period
def bj(r,j):
    # (2/pi) int_0^pi sin^r x sin(jx) dx = 2 Gamma(r+1) sin(j pi/2) / (2^r Gamma(1+(r+j)/2) Gamma(1+(r-j)/2))
    from scipy.special import gamma as G
    return 2*G(r+1)*math.sin(j*pi/2)/(2**r*G(1+(r+j)/2)*G(1+(r-j)/2))
def K(r,J=200001):
    from scipy.special import gammaln
    js=np.arange(1,J,2).astype(float)
    # |b_j| via reflection for j>=3 (avoid Gamma poles):  2 Gamma(r+1) cos(pi r/2)/(pi 2^r) * Gamma((j-r)/2)/Gamma((j+r)/2+1)
    if r==1: return 1.0
    c=2*gamma(r+1)*cos(pi*r/2)/(pi*2**r)
    terms=c*np.exp(gammaln((js-r)/2)-gammaln((js+r)/2+1))
    tail_c=c*2**(r+1)/(r)*( (J-r)/2 )**(-r)/2   # crude tail bound ~ c * sum_{j>=J} (j/2)^{-r-1}
    return float(np.abs(terms).sum()+tail_c)
def T(s):
    nmin=max(2,math.ceil(2*s-1-1e-12)); return (1+1/nmin)**(s-1)
# check K(r) against direct sine coefficients
for r in [1/3,0.5,2/3,0.9]:
    xs=np.linspace(0,np.pi,200001); f=np.sin(xs)**r
    b=[ (2/np.pi)*np.trapezoid(f*np.sin(j*xs),xs) for j in (1,3,5,7)]
    print(f"r={r:.3f} sigma={sigma(r):.5f} K={K(r):.4f}  b1..b7 direct={np.round(b,5)} formula={[round(bj(r,j),5) for j in (1,3,5,7)]}")
# exact check of the coverage claim and of the explicit bound at sample primes
worst=9; worst_at=None
for p in [p for p in range(29,20000) if p%4==3 and all(p%q for q in range(2,int(p**.5)+1))]:
    F=(p-1)//2; m=np.arange(1,F+1); sn=np.sin(2*np.pi*m/p); ch=chi_table(p)[m]
    S1=sn.sum()
    if p<3000:  # exact coverage check
        y=sn*(1+ch); v=np.zeros(p); v[1:F+1]=y; c=np.real(np.fft.fft(v))
        cov=(y.sum()-c[1:]).min(); assert cov>=S1-1e-9, (p,cov,S1)
    for s in np.linspace(1,3,81)[1:]:
        r=1/s; need=T(s)*(1-2/sqrt(p))
        if need<=1: continue
        Pr_bd=(F+0.5)*sigma(r)+1; Br_bd=sqrt(p)/2*K(r)
        rho_bd=(2*S1/p)*2**(s-1)*(F/(Pr_bd+Br_bd))**s
        if rho_bd/need<worst: worst=rho_bd/need; worst_at=(p,round(s,3))
print("worst (explicit lower bound on rho)/need over p=3 mod 4 in [29,20000], s in (1,3]:",round(worst,5),"at",worst_at)
