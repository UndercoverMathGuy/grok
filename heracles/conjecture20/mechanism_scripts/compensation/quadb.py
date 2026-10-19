# Quadratic bonus constant near IF, n=2:  sup bonus*|A| / sum_T |beta_T|^2   with sum_T|beta_T|^2 = eps^2 (a=3^{1-s}Q^s=1)
import numpy as np, sys
from scipy.optimize import minimize
from bonus2 import make, ifblock
s=float(sys.argv[1]); rng=np.random.default_rng(int(sys.argv[2]) if len(sys.argv)>2 else 0)
for eps in [0.3,0.1,0.03,0.01]:
    best=-np.inf
    for N in [3,4,5,6]:
        obj,eq,BT=make(s,N)
        ineq=lambda v: np.array([eps**2-np.sum(np.abs(BT(v))**2)])
        for t in range(30):
            v0=ifblock(s,N,rng,eps*(0.2+2*rng.random()))
            try:
                res=minimize(obj,v0,constraints=[{'type':'eq','fun':eq},{'type':'ineq','fun':ineq}],method='SLSQP',options={'maxiter':800,'ftol':1e-15})
            except Exception: continue
            v=res.x
            if ineq(v)[0]>-1e-12 and abs(eq(v)[0])<1e-10:
                A=abs(np.sum(v[:N]*np.exp(1j*v[N:2*N]))); bon=-obj(v)-1
                E=np.sum(np.abs(BT(v))**2); M=np.max(np.abs(BT(v)))**2
                val=bon*A/max(E,1e-30)
                if val>best: best=val; info=(N,bon,E,M,bon*1/M)
    print(f"s={s} eps={eps}: sup bonus*|A|/sum_T|B_T|^2 = {best:.5f}   (N={info[0]}, bonus={info[1]:.3e}, bonus*a/max_T^2={info[4]:.4f})  refs: (1-r)/3={(1-1/s)/3:.5f} (s-1)/3={(s-1)/3:.5f}",flush=True)
