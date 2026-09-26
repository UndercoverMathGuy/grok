# Rigorous (interval-arithmetic) certificate of the two-variable inequality behind (V3), r = 2/3.
# Domain: delta in [0,2], e1 in [1, sqrt3] minus the analytic corner [0,0.05] x [1,1.01].
# For each cell [da,db] x [ea,eb] prove  max(Cbound, Hbound) >= 3^{1-r} (1-da/2)^r   (lower bounds on sum x^r).
#  Cbound: any a,b,c>=0 with e1(abc)<=ea, e2(abc)<=E2, e3(abc)<=E3 gives sum x^r >= a^r+b^r+c^r, where
#          mu = min(eb, db + 2 sqrt(eb^2-1)), E2 = (1 - db*mu)/3, E3 = (1 - 3 db mu - db mu^2 - db^2 mu)/27.
#  Hbound: sum x^r >= 3^{-r} min{ sum lam^r : lam = ea + m, sum m = 0, |m| <= rho, lam >= 0 },
#          rho = sqrt(2(2 db^2 + (db + sqrt(2 eb (eb-1)))^2)); min over the extreme-point candidates.
from mpmath import iv, mpf
import numpy as np
from scipy.optimize import minimize
iv.dps=30
R=iv.mpf(2)/3
def ipow(x,e): return iv.exp(e*iv.log(x)) if x>0 else iv.mpf(0)
TGT=lambda da: ipow(iv.mpf(3),1-R)*ipow(1-iv.mpf(da)/2,R)
def cparams(da,db,ea,eb):
    db=iv.mpf(db); eb=iv.mpf(eb)
    mu=iv.sqrt(eb*eb-1)*2+db
    mu=mu if mu.b<eb.a else eb
    E2=(1-db*mu)/3; E3=(1-3*db*mu-db*mu*mu-db*db*mu)/27
    return mu,E2,E3
def cbound(da,db,ea,eb):
    mu,E2,E3=cparams(da,db,ea,eb)
    if E2.a<=0: return None
    E2f=float(E2.a); E3f=max(0.0,float(E3.a)); e1f=float(ea)
    best=None
    # exact factorisation candidate: reciprocal roots of 1 + e1 t + E2 t^2 + E3 t^3 (E3 clipped at 0), scaled by (1-1e-9)
    coeffs=[E3f,E2f,e1f,1.0] if E3f>0 else [E2f,e1f,1.0]
    roots=np.roots(coeffs); rec=[-1/t for t in roots]                  # p(t)=prod(1+rho t)
    reals=[z.real for z in rec if abs(z.imag)<1e-12]; cplx=[z for z in rec if z.imag>1e-12]
    sc=1-1e-9
    if len(cplx)==0 and all(x>=0 for x in reals):
        xs=[iv.mpf(float(x)*sc) for x in reals]+[iv.mpf(0)]*(3-len(reals))
        a,b,c=xs
        if (a+b+c).b<=ea and (a*b+b*c+c*a).b<=E2.a and (a*b*c).b<=max(E3.a,0):
            v=sum(ipow(t,R) for t in xs if t.a>0); best=v
    elif len(cplx)==1:
        z=cplx[0]; a=iv.mpf(max(reals[0],0)*sc) if reals else iv.mpf(0); Rr=iv.mpf(abs(z)*sc); th=iv.mpf(float(np.angle(z)))
        cth=iv.cos(th); s1=a+2*Rr*cth; s2=2*a*Rr*cth+Rr*Rr; s3=a*Rr*Rr
        if s1.b<=ea and s2.b<=E2.a and s3.b<=max(E3.a,0):
            v=(ipow(a,R) if a.a>0 else iv.mpf(0))+2*ipow(Rr,R)*iv.cos(R*th); best=v
    # type 1: three real factors (1+at)(1+bt)(1+ct)
    def neg1(z):
        a,b,c=np.abs(z); pen=max(0,a+b+c-e1f)+max(0,a*b+b*c+c*a-E2f)+max(0,a*b*c-E3f)
        return -(a**(2/3)+b**(2/3)+c**(2/3))+1e4*pen
    for z0 in ([1/3,1/3,1/3],[0.5,0.3,0.2],[0.6,0.2,0.2],[0.5,0.5,0.0]):
        z=np.abs(minimize(neg1,z0,method='Nelder-Mead',options={'xatol':1e-12,'fatol':1e-14,'maxiter':4000}).x)*(1-1e-9)
        a,b,c=[iv.mpf(float(t)) for t in z]
        if (a+b+c).b<=ea and (a*b+b*c+c*a).b<=E2.a and (a*b*c).b<=max(E3.a,0):
            v=ipow(a,R)+ipow(b,R)+ipow(c,R)
            if best is None or v.a>best.a: best=v
    # type 2: (1+at)(1+2R cos(th) t + R^2 t^2), value a^r + 2 R^r cos(r th), 0<=th<pi
    def neg2(z):
        a=abs(z[0]); Rr=abs(z[1]); th=min(abs(z[2]),np.pi-1e-6); c=np.cos(th)
        pen=max(0,a+2*Rr*c-e1f)+max(0,2*a*Rr*c+Rr*Rr-E2f)+max(0,a*Rr*Rr-E3f)
        return -(a**(2/3)+2*Rr**(2/3)*np.cos(2*th/3))+1e4*pen
    for z0 in ([1/3,1/3,0.0],[0.3,0.35,0.5],[0.2,0.4,1.0],[0.0,0.45,0.8]):
        z=minimize(neg2,z0,method='Nelder-Mead',options={'xatol':1e-12,'fatol':1e-14,'maxiter':4000}).x
        a=iv.mpf(abs(float(z[0]))*(1-1e-9)); Rr=iv.mpf(abs(float(z[1]))*(1-1e-9)); th=iv.mpf(min(abs(float(z[2])),float(np.pi)-1e-6))
        c=iv.cos(th)
        # dominance with c interval: use upper end of c where it increases a coefficient
        s1=a+2*Rr*c; s2=2*a*Rr*c+Rr*Rr; s3=a*Rr*Rr
        if s1.b<=ea and s2.b<=E2.a and s3.b<=max(E3.a,0):
            v=(ipow(a,R) if a>0 else iv.mpf(0))+2*ipow(Rr,R)*iv.cos(R*th)
            if best is None or v.a>best.a: best=v
    return best
def hbound(da,db,ea,eb):
    db=iv.mpf(db); eb=iv.mpf(eb); ea=iv.mpf(ea)
    sp=iv.sqrt(2*eb*(eb-1)) if eb>1 else iv.mpf(0)
    rho=iv.sqrt(2*(2*db*db+(db+sp)**2))
    c=ea; cands=[]
    k=rho/iv.sqrt(6)
    # two-equal candidates; if feasibility is uncertain include them with the possibly-zero term clipped to 0
    # (this only LOWERS the computed minimum, so the bound stays valid)
    if (c-2*k).b>0: cands.append(2*ipow(c+k,R)+(ipow(c-2*k,R) if (c-2*k).a>0 else iv.mpf(0)))
    else: cands.append(None)
    if (c-k).b>0: cands.append((2*ipow(c-k,R) if (c-k).a>0 else iv.mpf(0))+ipow(c+2*k,R))
    else: cands.append(None)
    # faces lam_j = 0 reachable?  lam=(x,y,0) with x+y=3c, (x-c)^2+(y-c)^2+c^2 <= rho^2 ; extreme: max spread
    if (rho*rho-6*c*c/4).b>=0 and (rho*rho - c*c).b>0:
        d2=(rho*rho-c*c-c*c/2)/2
        if d2.b>0:
            d=iv.sqrt(iv.mpf([0,d2.b]))
            x=c+c/2+d; y=c+c/2-d
            yv = y if y.a>0 else iv.mpf(0)
            cands.append(ipow(x,R)+(ipow(yv,R) if yv.a>0 else iv.mpf(0)))
    if (rho*rho).b>=6*c.a*c.a: cands.append(ipow(3*c,R))               # vertex (3c,0,0)
    if any(v is None for v in cands[:2]):
        # a two-equal candidate leaves the orthant: the minimum is then on a face; fall back to the face value if present
        cands=[v for v in cands if v is not None]
        if not cands: return None
    lo=min(v.a for v in cands)
    return iv.mpf(lo)/ipow(iv.mpf(3),R)
fails=[]; ncell=0
def check(da,db,ea,eb,depth=0):
    global ncell
    if db<=0.05 and eb<=1.01: return                       # analytic corner
    t=TGT(da)
    if ipow(iv.mpf(ea),R).a>=t.b: ncell+=1; return          # trivial: sum x^r >= e1^r >= ea^r
    hb=hbound(da,db,ea,eb)
    if hb is not None and hb.a>=t.b: ncell+=1; return
    cb=cbound(da,db,ea,eb)
    if cb is not None and cb.a>=t.b: ncell+=1; return
    if depth>=22: fails.append((da,db,ea,eb)); return
    if (db-da)/2 >= (eb-ea):
        m=(da+db)/2; check(da,m,ea,eb,depth+1); check(m,db,ea,eb,depth+1)
    else:
        m=(ea+eb)/2; check(da,db,ea,m,depth+1); check(da,db,m,eb,depth+1)
S3=float(iv.sqrt(3).b)
for (da,db) in [(0,0.05),(0.05,0.2),(0.2,0.5),(0.5,1.0),(1.0,1.5),(1.5,2.0)]:
    for (ea,eb) in [(1,1.01),(1.01,1.1),(1.1,1.3),(1.3,S3)]:
        check(da,db,ea,eb)
    print(f"delta block [{da},{db}] done: certified cells so far {ncell}, failures {len(fails)}",flush=True)
print("TOTAL certified cells:",ncell,"  failures:",len(fails))
for f in fails[:10]: print("FAIL",f)
