"""Monotone cell certificate for THEOREM DA (n=2): for every (d,w) outside the analytic corner R0,
   max(chain, peel, S^r) >= T(d)(1+tol),  S = sqrt(1+w^2), |A0| = 1, mu = 1-d.
Cell [d0,d1]x[w0,w1]: every tool is bounded BELOW using monotonicity (chain: e1=S(w0), E_k at (d1,w1);
peel: tails at (d1,w1), total S(w0); S^r at w0) and the target ABOVE by T(d0). Adaptive bisection.
usage: deptha_gridcert.py A r [dc wc]   (corner R0 = {d<=dc, w<=wc} is handled analytically in DEPTHA.md)"""
import numpy as np, sys, json, time
A=float(sys.argv[1]); r=float(sys.argv[2])
dc=float(sys.argv[3]) if len(sys.argv)>3 else 0.02; wc=float(sys.argv[4]) if len(sys.argv)>4 else 0.12
tol=1e-6; k0=3**((1-r)/r); dmax=1-1/k0; wmax=np.sqrt(k0*k0-1)
def LB(S,e2,e3):
    e2=max(e2,0.0); e3=max(e3,0.0); co=[e3,e2,S,1.0]
    while co[0]==0: co=co[1:]
    y=-1/np.roots(co); return float(np.sum((y.astype(complex)**r).real))
def cell_ok(d0,d1,w0,w1):
    S0=np.sqrt(1+w0*w0); S1=np.sqrt(1+w1*w1); T=3**(1-r)*(1-d0)**r*(1+tol)
    if S0**r>=T: return True
    mu=1-d1
    b1=(mu-S1*(1-A)**2)/((1+A)**2-(1-A)**2); b2=(mu-S1*(1-A*A))/((1+A)**2-(1-A*A))
    b1=min(max(b1,0),S0); b2=min(max(b2,0),b1)
    if (S0-b1)**r+(b1-b2)**r+b2**r>=T: return True
    D1=d1/A; D12=d1/A**2; u1=min(S1,D1+2*w1); u12=min(S1,D12+2*w1)
    E2=3-2*D1*u1-D12*u12; E3=1-2*D1*u1-D12*u12-D1*D1*u12-u1*u1*D12
    return LB(S0,E2/9,E3/27)>=T
# analytic corner conditions (polynomial, decreasing in d and w, so checking the far vertex suffices)
a1=2/A+1/A**2; a2=2/A**2+1/A**4
cB=6-dc*(3+a2)-2*wc*a1
cC=3-3*dc+dc*dc-dc*a2-2*wc*a1-(dc/A**2)*(dc/A**2+2*wc)-(dc/A+2*wc)**2/A**2
assert dc+2*wc*0<=1 and cB>=0 and cC>=0,("corner conditions fail",cB,cC)
t0=time.time(); stack=[(0.0,dmax,0.0,wmax)]; ncell=0; fail=[]; minsize=1
while stack:
    it=globals().get('it',0)+1
    if it>200000 or fail: fail.append('CAP/FAIL'); break
    d0,d1,w0,w1=stack.pop()
    if d1<=dc and w1<=wc: continue                      # inside analytic corner
    if cell_ok(d0,d1,w0,w1): ncell+=1; continue
    if max(d1-d0,w1-w0)<1e-5: fail.append((d0,d1,w0,w1)); continue
    minsize=min(minsize,max(d1-d0,w1-w0))
    dm=(d0+d1)/2; wm=(w0+w1)/2
    # split so that corner boundary lines are respected
    if d0<dc<d1: dm=dc
    if w0<wc<w1: wm=wc
    stack+= [(d0,dm,w0,wm),(dm,d1,w0,wm),(d0,dm,wm,w1),(dm,d1,wm,w1)]
res=dict(cornerB=cB,cornerC=cC,A=A,r=r,dc=dc,wc=wc,tol=tol,cells=ncell,failed=len(fail),min_cell=float(minsize),seconds=time.time()-t0,fail_examples=fail[:5])
print(res); json.dump(res,open('FINAL_RUNS/DEPTHA/gridcert_A%.3f_r%.4f.json'%(A,r),'w'))
