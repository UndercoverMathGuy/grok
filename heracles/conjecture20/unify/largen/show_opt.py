import sys, numpy as np
sys.path.insert(0,'.')
from kstar_sym import make
for n in [int(v) for v in sys.argv[1].split(',')]:
    d=np.load(f"kstar_opt_n{n}_b2.0.npy"); m=int(d[0]); z=d[1:]
    rho,al,ph=z[:m],z[m:2*m],z[2*m:]
    f=make(n,(n+1)/2); v,_,info=f(z,m,False)
    b=np.exp(rho+1j*al); A=info['A']; c=np.angle(A)
    o=np.argsort(ph); ph0=ph[o]-ph[o].mean()
    print(f"n={n} m={m} k={v:.4f} |A|/L={abs(A)/np.abs(b).sum():.3f}")
    print("  phi (deg)   :",np.round(np.degrees(ph0),1))
    print("  dphi (deg)  :",np.round(np.degrees(np.diff(ph0)),1))
    print("  |b|/max     :",np.round(np.abs(b[o])/np.abs(b).max(),3))
    print("  arg(b/A)    :",np.round(np.angle(b[o]/A),3))
    mk=info['mk']; from scipy.special import comb
    print("  C(n,k)|m_k|^2/|A|^2:",np.round([comb(n,k)*abs(mk[k])**2/abs(A)**2 for k in range(n+1)],3))
