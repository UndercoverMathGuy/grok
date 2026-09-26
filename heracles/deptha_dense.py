"""Dense brute-force scan of the certificate value in (d,w) outside the analytic corner (independent of the cell code)."""
import numpy as np, sys, json
A=float(sys.argv[1]); r=float(sys.argv[2]); G=int(sys.argv[3]); dc,wc=0.02,0.12
k0=3**((1-r)/r); dmax=1-1/k0; wmax=np.sqrt(k0*k0-1)
def LB(S,e2,e3):
    e2=max(e2,0.0); e3=max(e3,0.0); co=[e3,e2,S,1.0]
    while co[0]==0: co=co[1:]
    y=-1/np.roots(co); return float(np.sum((y.astype(complex)**r).real))
best=(9,)
for d in np.linspace(0,dmax,G):
    for w in np.linspace(0,wmax,G):
        if d<=dc and w<=wc: continue
        S=np.sqrt(1+w*w); mu=1-d; T=3**(1-r)*mu**r
        if S**r>=T: continue
        b1=(mu-S*(1-A)**2)/((1+A)**2-(1-A)**2); b2=(mu-S*(1-A*A))/((1+A)**2-(1-A*A))
        b1=min(max(b1,0),S); b2=min(max(b2,0),b1)
        p=(S-b1)**r+(b1-b2)**r+b2**r
        D1=d/A; D12=d/A**2; u1=min(S,D1+2*w); u12=min(S,D12+2*w)
        E2=3-2*D1*u1-D12*u12; E3=1-2*D1*u1-D12*u12-D1*D1*u12-u1*u1*D12
        val=max(p,LB(S,E2/9,E3/27))/T
        if val<best[0]: best=(val,d,w,S)
print(A,r,best); json.dump([float(z) for z in best],open('FINAL_RUNS/DEPTHA/dense_A%.2f_r%.3f.json'%(A,r),'w'))
