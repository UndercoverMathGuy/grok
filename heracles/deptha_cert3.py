"""Certificate v2, margin map: min of val over region excluding corner {d<=dc, S<=Sc}; and A sweep."""
import numpy as np, sys, json
r=float(sys.argv[1]) if len(sys.argv)>1 else 2/3
G=int(sys.argv[2]) if len(sys.argv)>2 else 400
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
out={}
for A in [0.40,0.42,0.44,0.46,0.48,0.5,0.6,0.8,1.0]:
    worst=(9,None)
    for d in np.linspace(0,1-1/k0,G):
        mu=1-d
        for S in (np.linspace(1,k0*mu,G) if k0*mu>1 else [1.0]):
            if d<=0.02 and S<=1.01: continue
            c,p,s=tools(A,d,S); val=max(c,p,s)/(3**(1-r)*mu**r)
            if val<worst[0]: worst=(val,d,S,c,p,s)
    print(A,worst); out[A]=[float(z) for z in worst]
json.dump(out,open('FINAL_RUNS/DEPTHA/cert3_r%.4f.json'%r,'w'))
