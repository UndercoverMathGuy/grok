"""Margin profile of certificate v2 at fixed A: for each d-band, min over S of val, with argmin and which tool wins."""
import numpy as np, sys, json
sys.argv=[sys.argv[0]]+sys.argv[1:]
A=float(sys.argv[1]); r=float(sys.argv[2]); G=int(sys.argv[3])
import importlib.util
k0=3**((1-r)/r)
def LB(S,e2,e3):
    e2=max(e2,0); e3=max(e3,0); co=[e3,e2,S,1.0]
    while co[0]==0: co=co[1:]
    y=-1/np.roots(co); return float(np.sum((y.astype(complex)**r).real))
def tools(A,d,S):
    mu=1-d; d1=d/A; d12=d/A**2; w=np.sqrt(max(S*S-1,0))
    u1=min(S,d1+2*w); u12=min(S,d12+2*w)
    E2=3-2*d1*u1-d12*u12; E3=1-2*d1*u1-d12*u12-d1*d1*u12-u1*u1*d12
    b1=(mu-S*(1-A)**2)/((1+A)**2-(1-A)**2); b2=(mu-S*(1-A*A))/((1+A)**2-(1-A*A))
    b1=min(max(b1,0),S); b2=min(max(b2,0),b1)
    p=float(np.sum(np.array([S-b1,b1-b2,b2])**r))
    return LB(S,E2/9,E3/27),p,S**r
rows=[]
for d in [0,0.005,0.01,0.02,0.03,0.05,0.07,0.09,0.11,0.13,0.15,0.2,0.25,0.3,0.35,0.4,1-1/k0]:
    mu=1-d; best=(9,)
    for S in np.linspace(1,max(k0*mu,1),G):
        c,p,s=tools(A,d,S); t=3**(1-r)*mu**r
        val=max(c,p,s)/t
        if val<best[0]: best=(val,S,c/t,p/t,s/t)
    rows.append((d,)+best); print("d=%.3f min val=%.5f at S=%.4f chain=%.4f peel=%.4f Sr=%.4f"%((d,)+best))
json.dump(rows,open('FINAL_RUNS/DEPTHA/cert4_A%.2f_r%.3f.json'%(A,r),'w'))
