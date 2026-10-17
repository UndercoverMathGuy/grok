# Certified (interval-arithmetic) verification of the s-free 2-D inequality (S3') used for TARGET 2:
#   for all (b,y) in [0,1]^2 outside the analytic corners R0=[0,c0/2]x[0,c0] and R1=[b1c,1]x[y1c,1]:
#   exists lam' in [0,1], kap in [0,1-lam'] with
#        x0^{Lt(b,y)} >= sqrt(1 + c0^2 kap^2 - 2 c0 kap y) - c0 lam' b,
#   x0 = 3^{-1/2}, c0 = 1-x0,
#   Lt(b,y) = (1/(3 log 3)) sum_k nu_k log nu_k,  nu_k = 1+mu_k,  mu^3 - P mu - q = 0,
#   P = 2b^2+y^2,  q = max(-2 b^2 y, P-1).
# Monotonicity (proved in the note): Lt increasing in b and y; the right side decreasing in b and y.
# Hence on a box [b0,b1]x[y0,y1] it suffices that  x0^{Lt(b1,y1)} >= g(b0,y0).
import sys, numpy as np
from mpmath import iv, mp
from scipy.optimize import minimize
iv.dps=30
X0=iv.mpf(1)/iv.sqrt(iv.mpf(3)); C0=1-X0; LOG3=iv.log(iv.mpf(3))
c0f=1-3**-0.5
TWO_PI_3=2*iv.pi/3
def hsum_upper(nus):
    tot=iv.mpf(0)
    for nu in nus:
        lo=nu.a
        if lo<=0:      # nu in [0,hi]; h(nu)=nu log nu <= 0 on [0,1]
            hi=nu.b
            if hi<=iv.mpf(1)/iv.e: tot+=iv.mpf(0)
            else: raise ValueError("bad nu interval")
        else:
            tot+=nu*iv.log(nu)
    return tot
def Lt_upper(b,y):
    b=iv.mpf(b); y=iv.mpf(y)
    P=2*b*b+y*y
    q1=-2*b*b*y; q2=P-1
    cands=[]
    # branch 2 (q=P-1): roots -1, (1 +- sqrt(4P-3))/2  -> nu = 0, (3 +- sqrt(4P-3))/2
    if not (q2.b < q1.a):   # q2 may be the max
        d=iv.sqrt(4*P-3)
        cands.append([ (3+d)/2, (3-d)/2 ])
    if not (q1.b < q2.a):   # q1 may be the max
        q=q1
        z=(3*q/(2*P))*iv.sqrt(3/P)
        za=max(z.a,-1); zb=min(z.b,1)
        z=iv.mpf([za,zb])
        phi=iv.atan2(iv.sqrt(iv.mpf([max((1-z*z).a,0),(1-z*z).b])),z)
        R=2*iv.sqrt(P/3)
        mus=[R*iv.cos(phi/3-TWO_PI_3*k) for k in range(3)]
        cands.append([1+m for m in mus])
    best=None
    for nus in cands:
        u=hsum_upper(nus)/(3*LOG3)
        best=u if best is None else iv.mpf([min(best.a,u.a),max(best.b,u.b)])
    return best
def lhs_lower(b,y):
    L=Lt_upper(b,y)
    v=X0**iv.mpf(L.b)       # x0^L decreasing in L
    return iv.mpf(v.a)
def g_upper(b,y,lam,kap):
    b=iv.mpf(b); y=iv.mpf(y); lam=iv.mpf(lam); kap=iv.mpf(kap)
    v=iv.sqrt(1+C0*C0*kap*kap-2*C0*kap*y)-C0*lam*b
    return iv.mpf(v.b)
def best_lk(b,y):
    # numerically choose (lam',kap) minimizing g at (b,y) subject to kap<=1-lam'
    best=(9,0,0)
    for lam in np.linspace(0,1,101):
        kmax=1-lam
        k=min(max(y/c0f,0),kmax)     # minimizer of sqrt(1+c0^2k^2-2c0ky) is k=y/c0 (clipped)
        v=np.sqrt(1+c0f**2*k*k-2*c0f*k*y)-c0f*lam*b
        if v<best[0]: best=(v,lam,k)
    lam,k=best[1],best[2]
    k=min(k,1-lam)*(1-2.0**-40)   # guarantees kap < 1-lam' exactly
    return lam,k
def in_R0(b0,b1,y0,y1): return b1<=c0f/2-1e-12 and y1<=c0f-1e-12
B1C,Y1C=float(sys.argv[1]) if len(sys.argv)>1 else 0.99, float(sys.argv[2]) if len(sys.argv)>2 else 0.985
def in_R1(b0,b1,y0,y1): return b0>=B1C and y0>=Y1C
stats={'ok':0,'fail':0,'min_margin':1.0}
fails=[]
def check(b0,b1,y0,y1,depth):
    if in_R0(b0,b1,y0,y1) or in_R1(b0,b1,y0,y1): return
    lam,kap=best_lk(b0,y0)
    d=lhs_lower(b1,y1)-g_upper(b0,y0,lam,kap)   # interval; certified iff its lower end >= 0
    mg=float(d.a)
    if d.a>=0:
        stats['ok']+=1; stats['min_margin']=min(stats['min_margin'],mg); return
    if depth>=int(sys.argv[3]) if len(sys.argv)>3 else depth>=14:
        stats['fail']+=1; fails.append((b0,b1,y0,y1,mg)); return
    bm=(b0+b1)/2; ym=(y0+y1)/2
    for (a0,a1) in [(b0,bm),(bm,b1)]:
        for (e0,e1) in [(y0,ym),(ym,y1)]:
            check(a0,a1,e0,e1,depth+1)
N=int(sys.argv[4]) if len(sys.argv)>4 else 20
grid=np.linspace(0,1,N+1)
# make grid points exact binary fractions
for i in range(N):
    for j in range(N):
        check(float(grid[i]),float(grid[i+1]),float(grid[j]),float(grid[j+1]),0)
print("boxes verified:",stats['ok']," failed:",stats['fail']," min certified margin:",stats['min_margin'])
if fails: print("first fails:",fails[:10])
