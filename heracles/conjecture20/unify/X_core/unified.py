# Unified compression: y_m(w0) = beta_m * (cos(2 pi m w0/p + phi_m))_+ , phases phi_m of U = chi.
# Claim: p=1 mod 4, w0=0 gives the QR half; p=3 mod 4, w0=+-1 gives sin_m (1+chi(m))/2 (up to normalisation).
import numpy as np
def chi_table(p):
    t=np.full(p,-1.0); t[0]=0
    for a in range(1,p): t[(a*a)%p]=1
    return t
for p in [13,29,37,7,11,19,23,31,43]:
    F=(p-1)//2; m=np.arange(1,F+1); w=np.arange(p); chi=chi_table(p)
    c=np.array([np.sum(chi*np.exp(-2j*np.pi*k*w/p))/p for k in m]); phi=np.angle(c)   # U(w)=sum 2|c_m| cos(2 pi m w/p + phi_m)
    res=[]
    for w0 in range(p):
        y=np.maximum(np.cos(2*np.pi*m*w0/p+phi),0)
        if p%4==1: tgt=(chi[m]>0).astype(float)
        else: tgt=np.sin(2*np.pi*m/p)*(1+chi[m])/2
        if np.allclose(y,tgt,atol=1e-9): res.append(w0)
    print(p, 'p%4=',p%4,' w0 reproducing the compression:',res)
