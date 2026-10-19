import sys, numpy as np
sys.path.insert(0,'.')
from kstar_large import search, unpack, obj
for n,m in [(5,4),(6,4),(7,5),(8,6)]:
    s=(n+1)/2
    v,z,info=search(n,m,s,12,seed=1000*n+m)
    rho,al,th=unpack(z,m,n)
    b=np.exp(rho+1j*al); A=b.sum(); ph=np.angle(b/A)
    # canonicalise slot phases: for each slot t remove the phase of atom 0
    thc=(th-th[0:1,:])%(2*np.pi)
    print(f"n={n} m={m} k*={v:.4f}")
    order=np.argsort(-np.abs(b))
    for i in order:
        print(f"  |b|/L={abs(b[i])/np.abs(b).sum():.4f} arg(b/A)={ph[i]:+.4f}  slot phases rel. atom0 (deg): {np.round(np.degrees(thc[i]),1)}")
